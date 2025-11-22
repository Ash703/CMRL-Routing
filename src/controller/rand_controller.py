import time
import numpy as np
from collections import deque
from threading import Lock
import os
import yaml
import csv
from utils import Network

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
ELEPHANT_BYTES = 2 * 1024 * 1024    
CAPACITY_Mbps = 1000.0              # normalize Mbps by this (1 Gbps)
FLOW_IDLE_TIMEOUT = 30              # seconds idle timeout for per-flow rules

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
        self.transitions = deque(maxlen=5000)
        self.groups_last_used = {}       # (dpid, group_id) -> last_used_ts
        self.promoted_flows = set()      # set of ((src,dst), ingress_leaf) already promoted
        # new: per-promoted-flow metadata for cleanup
        # key: ((src,dst), ingress_leaf) -> { 'gid': int, 'last_bytes': int, 'missing': int, 'inactive': int, 'dpid': ingress_leaf }
        self.promoted_meta = {}

        self.lock = Lock()

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

    # -------------------------
    # Flow stats reply: update cache AND run elephant detection (polling-based)
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        with self.lock:
            self.flow_stats_cache[dpid] = ev.msg.body

    # -------------------------
    # Port stats reply
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        now = time.time()
        with self.lock:
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

        # ------------------ end ARP ------------------

        # handle IPv4 flows (RL-driven forwarding)
        ip = pkt.get_protocol(ipv4.ipv4)
        if not ip:
            return

        src = ip.src
        dst = ip.dst
        flow_key = (src, dst)

        ingress_leaf = HOST_TO_LEAF.get(src)
        dst_leaf = HOST_TO_LEAF.get(dst)

        if ingress_leaf is None:
            # unknown host mapping: flood
            actions = [parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)]
            data = msg.data if msg.buffer_id == ofp_pi.OFP_NO_BUFFER else None
            out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id, in_port=msg.match.get('in_port'), actions=actions, data=data)
            dp_packetin.send_msg(out)
            return

        candidate_ports = LEAF_SPINE_PORTS.get(ingress_leaf, [])
        if not candidate_ports:
            actions = [parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)]
            data = msg.data if msg.buffer_id == ofp_pi.OFP_NO_BUFFER else None
            out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id, in_port=msg.match.get('in_port'), actions=actions, data=data)
            dp_packetin.send_msg(out)
            return

        # ------------------ SAME-LEAF FAST PATH ------------------
        if dst_leaf is not None and dst_leaf == ingress_leaf:
            dp_ing = self.datapaths.get(ingress_leaf)
            if dp_ing is None:
                self.logger.warning("Ingress leaf %s not registered; flooding", ingress_leaf)
                out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id, in_port=msg.match.get('in_port'), actions=[parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)], data=msg.data)
                dp_packetin.send_msg(out)
                return

            out_port = self._get_host_port(ingress_leaf, dst)
            in_port = self._get_host_port(ingress_leaf, src)

            # install forward & reverse flows on the leaf
            self._install_simple_flow(dp_ing, src, dst, out_port, idle_timeout=FLOW_IDLE_TIMEOUT)
            self._install_simple_flow(dp_ing, dst, src, in_port, idle_timeout=FLOW_IDLE_TIMEOUT)

            self.logger.info("SAME-LEAF %s->%s on leaf %s ports %s<->%s", src, dst, ingress_leaf, in_port, out_port)

            # send PacketOut from ingress leaf (use controller as in_port)
            try:
                parser_ing = dp_ing.ofproto_parser
                ofp_ing = dp_ing.ofproto
                out = parser_ing.OFPPacketOut(datapath=dp_ing, buffer_id=ofp_ing.OFP_NO_BUFFER, in_port=ofp_ing.OFPP_CONTROLLER, actions=[parser_ing.OFPActionOutput(out_port)], data=msg.data)
                dp_ing.send_msg(out)
            except Exception as e:
                self.logger.debug("PacketOut same-leaf failed: %s", e)
            return
        # ------------------ end same-leaf ------------------

        # ensure dp_ing exists
        dp_ing = self.datapaths.get(ingress_leaf)
        if dp_ing is None:
            self.logger.warning("Ingress leaf %s not in datapaths; flooding", ingress_leaf)
            out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id, in_port=msg.match.get('in_port'), actions=[parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)], data=msg.data)
            dp_packetin.send_msg(out)
            return

        # chosen uplink port on ingress leaf
        chosen_idx = np.random.randint(len(candidate_ports))
        chosen_port = candidate_ports[chosen_idx]

        # install on ingress leaf
        self._install_simple_flow(dp_ing, src, dst, chosen_port, idle_timeout=FLOW_IDLE_TIMEOUT)
        actions_out = [dp_ing.ofproto_parser.OFPActionOutput(chosen_port)]

        # ---------- INSTALL spine + egress leaf rules ----------
        spine_dpid = SPINES[chosen_idx] if chosen_idx < len(SPINES) else SPINES[0]
        dp_spine = self.datapaths.get(spine_dpid)

        # install egress leaf -> host flow
        dp_dst = self.datapaths.get(dst_leaf)
        dst_host_port = self._get_host_port(dst_leaf, dst)
        if dp_dst is not None and dst_host_port is not None:
            self._install_simple_flow(dp_dst, src, dst, dst_host_port, idle_timeout=FLOW_IDLE_TIMEOUT)

        if dp_dst is not None and dp_spine is not None:
            uplinks = LEAF_SPINE_PORTS.get(dst_leaf, [])
            idx_of_spine = SPINES.index(spine_dpid) if spine_dpid in SPINES else None
            dst_uplink_port = None
            if idx_of_spine is not None and idx_of_spine < len(uplinks):
                dst_uplink_port = uplinks[idx_of_spine]
            if dst_uplink_port is not None:
                self._install_simple_flow(dp_dst, dst, src, dst_uplink_port, idle_timeout=FLOW_IDLE_TIMEOUT)

        # ingress leaf reverse to host
        src_host_port = self._get_host_port(ingress_leaf, src)
        if dp_ing is not None and src_host_port is not None:
            self._install_simple_flow(dp_ing, dst, src, src_host_port, idle_timeout=FLOW_IDLE_TIMEOUT)

        # send the current packet out from ingress leaf (use controller as in_port)
        try:
            parser_ing = dp_ing.ofproto_parser
            ofp_ing = dp_ing.ofproto
            out = parser_ing.OFPPacketOut(datapath=dp_ing, buffer_id=ofp_ing.OFP_NO_BUFFER, in_port=ofp_ing.OFPP_CONTROLLER, actions=actions_out, data=msg.data)
            dp_ing.send_msg(out)
        except Exception as e:
            self.logger.debug("PacketOut to ingress leaf failed: %s", e)
    # -------------------------
    # Flow install helper
    # -------------------------
    def _install_simple_flow(self, datapath, src, dst, out_port, priority=100, idle_timeout=0):
        if datapath is None or out_port is None:
            return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        actions = [parser.OFPActionOutput(out_port)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst, idle_timeout=idle_timeout)
        datapath.send_msg(mod)