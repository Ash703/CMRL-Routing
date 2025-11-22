from utils import Network
import os, yaml, time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp

# ---------------------------
# Config
# ---------------------------
POLL_INTERVAL = 2.0                 # seconds between port/flow polls
ELEPHANT_BYTES = 2 * 1024 * 1024    # 2 MB promotion threshold
CAPACITY_Mbps = 1000.0              # normalize Mbps by this (1 Gbps)
GROUP_WEIGHT_SCALE = 100           # scale prob -> bucket weight

config_file = os.environ.get("NETWORK_CONFIG_FILE", "network_config3.yaml")
net = Network(config_file)

with open(config_file) as f:
    raw_cfg = yaml.safe_load(f)

HOST_TO_LEAF = {}
HOST_PORT = {}

for host in raw_cfg["hosts"]:
    ip = host["ip"]
    leaf_name = host["connected_to"]
    port = host["port"]

    # find leaf dpid using switch_mapping equivalent:
    leaf_dpid = next(
        sw["id"] for sw in raw_cfg["switches"] if sw["name"] == leaf_name
    )

    HOST_TO_LEAF[ip] = leaf_dpid
    HOST_PORT[ip] = port

SPINES = net.spines[:]

# Build LEAF_SPINE_PORTS aligned to SPINES ordering
LEAF_SPINE_PORTS = {}

for leaf in net.leaves:
    ports = []
    for spine in net.spines:
        # Look for leaf -> spine link
        port = None
        if (leaf, spine) in net.links:
            port = net.links[(leaf, spine)]["port"]
        ports.append(port)  # may be None if topology is missing a link
    LEAF_SPINE_PORTS[leaf] = ports

SPINE_TO_LEAF_PORTS = {}

for spine in net.spines:
    leaf_ports = {}
    for leaf in net.leaves:
        if (spine, leaf) in net.links:
            leaf_ports[leaf] = net.links[(spine, leaf)]["port"]
    SPINE_TO_LEAF_PORTS[spine] = leaf_ports

print(HOST_PORT,HOST_TO_LEAF,SPINES,LEAF_SPINE_PORTS,SPINE_TO_LEAF_PORTS)

class RLDCController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RLDCController, self).__init__(*args, **kwargs)

        # bookkeeping
        self.datapaths = {}              # dpid -> datapath
        self.port_stats = {}             # dpid -> port -> {'tx','rx','ts','util','tx_dropped',...}
        self.flow_stats_cache = {}       # dpid -> list of flow stats (latest)
        self.flow_prev_bytes = {}        # ((src,dst), dpid) -> previous byte_count
        self.flow_memory = {}            # (src,dst) -> meta for training
        self.groups_last_used = {}       # (dpid, group_id) -> last_used_ts
        # threads
        self.monitor_thread = hub.spawn(self._monitor)
        self.logger.info("RLDCController (final) initialized")

    # -------------------------
    # helpers
    # -------------------------
    def _get_host_port(self, leaf, host_ip):
        return HOST_PORT.get(host_ip)

    def _group_id_from_flow(self, flow_key):
        return (abs(hash(flow_key)) % (63556)) or 1

    # -------------------------
    # Table-miss
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto

        # send unmatched to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=inst)
        dp.send_msg(mod)
        self.logger.info("Installed table-miss on switch %s", dp.id)

        #for spine switches
        if dp.id in SPINES:
            # self.logger.info("spine: %s",dp.id)
            for host_ip, leaf in HOST_TO_LEAF.items():
                match = parser.OFPMatch(eth_type=0x0800, ipv4_dst=host_ip)
                actions = [parser.OFPActionOutput(SPINE_TO_LEAF_PORTS[dp.id][leaf])]
                inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
                mod = parser.OFPFlowMod(datapath=dp, priority=200, match=match, instructions=inst, idle_timeout=0)
                dp.send_msg(mod)
                self.logger.info("Installed Spine rule: host %s to leaf %s",host_ip, leaf)
        
        #for leaf switches
        if dp.id in LEAF_SPINE_PORTS:
            for dst in HOST_PORT.keys():
                if HOST_TO_LEAF[dst] == dp.id:
                    self._install_simple_flow(dp,dst=dst,out_port=HOST_PORT[dst],idle_timeout=0)
                else:
                    gid = self._group_id_from_flow(dst)
                    candidate_ports = LEAF_SPINE_PORTS[dp.id]
                    self._install_select_group(dp, gid, candidate_ports, [(GROUP_WEIGHT_SCALE // len(candidate_ports)) for _ in range(len(candidate_ports))])
                    self._install_flow_to_group(dp, dst, gid, idle_timeout=0)


    # -------------------------
    # State change
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
            self.port_stats.setdefault(dp.id, {})
            self.logger.info("Datapath %s connected", dp.id)
        else:
            if dp.id in self.datapaths:
                del self.datapaths[dp.id]
            if dp.id in self.port_stats:
                del self.port_stats[dp.id]
            self.logger.info("Datapath %s disconnected", dp.id)

    # -------------------------
    # Polling
    # -------------------------
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                try:
                    self._request_port_stats(dp)
                    self._request_flow_stats(dp)
                    self._request_group_stats(dp)
                except Exception as e:
                    self.logger.exception("Error requesting stats: %s", e)
            hub.sleep(POLL_INTERVAL)

    def _request_port_stats(self, datapath):
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        req = parser.OFPPortStatsRequest(datapath, 0, ofp.OFPP_ANY)
        datapath.send_msg(req)

    def _request_flow_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    def _request_group_stats(self, datapath):
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        req = parser.OFPGroupStatsRequest(datapath, 0, ofp.OFPG_ALL)
        datapath.send_msg(req)
    # -------------------------
    # Flow stats reply:
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        self.flow_stats_cache[dpid] = ev.msg.body

    @set_ev_cls(ofp_event.EventOFPGroupStatsReply, MAIN_DISPATCHER)
    def _group_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        # snapshot group stats indexed by group_id for O(1) lookup
        stats_snapshot = list(ev.msg.body)
        group_by_gid = {}
        for st in stats_snapshot:
            try:
                gid = int(getattr(st, "group_id", None))
                # some implementations provide byte_count/packet_count directly on st
                byte_count = int(getattr(st, "byte_count", 0) or 0)
                group_by_gid[gid] = byte_count
            except Exception:
                continue

    # -------------------------
    # Port stats reply
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        now = time.time()
        self.port_stats.setdefault(dpid, {})
        for stat in ev.msg.body:
            p = stat.port_no
            if p <= 0:
                continue
            prev = self.port_stats[dpid].get(p)
            if prev:
                interval = now - prev['ts']
                if interval <= 0:
                    util = prev.get('util', 0.0)
                    tx_dropped_delta = 0
                    rx_dropped_delta = 0
                else:
                    tx_delta = stat.tx_bytes - prev['tx']
                    rx_delta = stat.rx_bytes - prev['rx']
                    util = (tx_delta + rx_delta) * 8.0 / (interval * 1e6)  # Mbps
                    tx_dropped_delta = stat.tx_dropped - prev.get('tx_dropped', 0)
                    rx_dropped_delta = stat.rx_dropped - prev.get('rx_dropped', 0)
            else:
                util = 0.0
                tx_dropped_delta = 0
                rx_dropped_delta = 0

            # with open(f"{CHECKPOINT_DIR}/port_util_log.csv", "a", newline="") as f:
            #     writer = csv.writer(f)
            #     writer.writerow([time.time(), dpid, p, util, tx_dropped_delta, rx_dropped_delta])

            self.port_stats[dpid][p] = {
                'tx': stat.tx_bytes,
                'rx': stat.rx_bytes,
                'ts': now,
                'util': util,
                'tx_dropped': stat.tx_dropped,
                'rx_dropped': stat.rx_dropped,
                'tx_dropped_delta': tx_dropped_delta,
                'rx_dropped_delta': rx_dropped_delta
            }

    # -------------------------
    # Packet-in
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        msg = ev.msg
        dp_packetin = msg.datapath
        parser_pi = dp_packetin.ofproto_parser
        ofp_pi = dp_packetin.ofproto

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        # ---- ARP handling --------------------------------------------------
        if eth and eth.ethertype == 0x0806:  # ARP
            arp_pkt = pkt.get_protocol(arp.arp)
            if not arp_pkt:
                return

            target_ip = arp_pkt.dst_ip
            src_dp = dp_packetin
            src_dpid = src_dp.id
            in_port = msg.match.get('in_port')

            data = msg.data if msg.buffer_id == ofp_pi.OFP_NO_BUFFER else None

            # If we know destination leaf, send only to that leaf's host port(s)
            dst_leaf = HOST_TO_LEAF.get(target_ip)
            
            if dst_leaf and src_dpid == dst_leaf:
                # self.logger.info("Destination: %s, Source: %s",dst_leaf,src_dpid)
                dp_target = self.datapaths.get(dst_leaf)
                if dp_target:
                    host_ports = [HOST_PORT[ip] for ip, leaf in HOST_TO_LEAF.items() if leaf == dst_leaf]
                    # print(host_ports)
                    actions = [dp_target.ofproto_parser.OFPActionOutput(p) for p in host_ports]
                    out = dp_target.ofproto_parser.OFPPacketOut(
                        datapath=dp_target,
                        buffer_id=dp_target.ofproto.OFP_NO_BUFFER,
                        in_port=dp_target.ofproto.OFPP_CONTROLLER,
                        actions=actions,
                        data=data
                    )
                    dp_target.send_msg(out)
                return

            if src_dpid in SPINES:
                # print(SPINES)
                # self.logger.info("Destination: %s, Source: %s",dst_leaf,src_dpid)
                target_port = SPINE_TO_LEAF_PORTS[src_dpid][dst_leaf]
                actions = [src_dp.ofproto_parser.OFPActionOutput(target_port)]
                out = src_dp.ofproto_parser.OFPPacketOut(
                    datapath=src_dp,
                    buffer_id=src_dp.ofproto.OFP_NO_BUFFER,
                    in_port=src_dp.ofproto.OFPP_CONTROLLER,
                    actions=actions,
                    data=data
                )
                src_dp.send_msg(out)
                return
            
            #send to both spines
            # self.logger.info("Destination: %s, Source: %s",dst_leaf,src_dpid)
            target_ports = LEAF_SPINE_PORTS[src_dpid]
            actions = [src_dp.ofproto_parser.OFPActionOutput(target_port) for target_port in target_ports]
            out = src_dp.ofproto_parser.OFPPacketOut(
                datapath=src_dp,
                buffer_id=src_dp.ofproto.OFP_NO_BUFFER,
                in_port=src_dp.ofproto.OFPP_CONTROLLER,
                actions=actions,
                data=data
            )
            src_dp.send_msg(out)
            return

    # -------------------------
    # Flow & group install helpers
    # -------------------------
    def _install_simple_flow(self, datapath, dst, out_port, priority=100, idle_timeout=0):
        if datapath is None or out_port is None:
            return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_dst=dst)
        actions = [parser.OFPActionOutput(out_port)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst, idle_timeout=idle_timeout)
        datapath.send_msg(mod)

    def _install_select_group(self, datapath, group_id, out_ports, weights):
        if datapath is None:
            return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        buckets = []
        for p, w in zip(out_ports, weights):
            actions = [parser.OFPActionOutput(p)]
            buckets.append(parser.OFPBucket(weight=int(w), watch_port=ofp.OFPP_ANY, watch_group=ofp.OFPG_ANY, actions=actions))
        # Try MODIFY then ADD (some switches accept either)
        grp_mod = parser.OFPGroupMod(datapath=datapath, command=ofp.OFPGC_MODIFY, type_=ofp.OFPGT_SELECT, group_id=group_id, buckets=buckets)
        try:
            grp_add = parser.OFPGroupMod(datapath=datapath, command=ofp.OFPGC_ADD, type_=ofp.OFPGT_SELECT, group_id=group_id, buckets=buckets)
            datapath.send_msg(grp_add)
            datapath.send_msg(grp_mod)
        except Exception:
            pass
        self.groups_last_used[(datapath.id, group_id)] = time.time()

    def _install_flow_to_group(self, datapath, dst, group_id, priority=50, idle_timeout=0):
        if datapath is None:
            return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto

        match = parser.OFPMatch(eth_type=0x0800,
                                ipv4_dst=dst)

        actions = [parser.OFPActionGroup(group_id)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofp.OFPFC_ADD,
            priority=priority,        
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=0
        )
        datapath.send_msg(mod)
