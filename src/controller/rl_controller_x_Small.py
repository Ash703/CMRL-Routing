"""
Final patched Ryu controller for the YAML leaf-spine topology.
Features:
 - Network-wide ARP broadcast (sent via leaf uplinks -> spines -> target leaf)
 - SAME-LEAF fast path (direct host-to-host on leaf)
 - End-to-end flow install (ingress leaf -> chosen spine -> egress leaf -> host)
 - Reverse flows installed symmetrically
 - Elephant flows: SELECT group on ingress leaf + spine/egress flows
 - Uses OpenFlow port/flow stats for RL input
 - Idle timeouts for flows, group cleanup
 - Thread-safe (Lock) around shared state

Elephant detection and promotion is performed from flow-stats polling:
 - when flow stats are received for a leaf, compute delta vs saved prev bytes
 - if delta >= ELEPHANT_BYTES (or absolute >= ELEPHANT_BYTES) promote on ingress leaf
"""
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

# RL model: keep unchanged (controller expects policy(state_np) -> (probs, action_idx))
from src.rl.rl_model_x import ActorCritic

# ---------------------------
# Config
# ---------------------------
POLL_INTERVAL = 2.0                 # seconds between port/flow polls
ELEPHANT_BYTES = 2 * 1024 * 1024    # 2 MB promotion threshold
CAPACITY_Mbps = 1000.0              # normalize Mbps by this (1 Gbps)
GROUP_WEIGHT_SCALE = 100           # scale prob -> bucket weight
TRAIN_BATCH_SIZE = 8
DEVICE = 'cpu'
FLOW_IDLE_TIMEOUT = 30              # seconds idle timeout for per-flow rules
GROUP_IDLE_TIMEOUT = 300            # seconds to keep unused groups before deletion

CHECKPOINT_DIR = "checkpoints"
MODEL_SAVE_STEPS = 1


# Reward coefficients (alpha, beta, gamma, delta)
ALPHA = 1.0   # throughput gain weight
BETA  = 1.0   # utilization skew penalty
GAMMA = 1.0   # packet loss penalty
DELTA = 0.0   # latency penalty (0 by default)

config_file = os.environ.get("NETWORK_CONFIG_FILE", "network_config.yaml")
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

        # RL model
        num_paths = len(SPINES)
        #[util1,drop1,util2,drop2]
        self.model = ActorCritic(input_dim=num_paths*2, num_actions=num_paths, device=DEVICE)
        self.rl_update_step = 0
        SAVE_PATH = f"{CHECKPOINT_DIR}/rl_latest.pt"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        try:
            self.model.load_checkpoint(SAVE_PATH, map_location=DEVICE)
            self.logger.info("Loaded RL model from %s", SAVE_PATH)
        except Exception:
            self.logger.info("No saved RL model, starting fresh")


        # threads
        self.monitor_thread = hub.spawn(self._monitor)
        self.trainer_thread = hub.spawn(self._trainer)
        self.cleaner_thread = hub.spawn(self._group_cleaner)
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
    # Flow stats reply: update cache AND run elephant detection (polling-based)
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        with self.lock:
            self.flow_stats_cache[dpid] = ev.msg.body

        # If this is a leaf, check flow entries on this datapath for elephant promotion.
        # The logic: for each IPv4 flow entry (ipv4_src, ipv4_dst) compute delta vs flow_prev_bytes
        # If delta >= ELEPHANT_BYTES or absolute >= ELEPHANT_BYTES => promote on ingress leaf.
        try:
            if dpid in LEAF_SPINE_PORTS:
                # iterate snapshot (do not hold lock across heavy ops)
                stats_snapshot = list(ev.msg.body)
                for s in stats_snapshot:
                    try:
                        m = getattr(s, 'match', {}) or {}
                        # only consider IPv4 flows with both src and dst match
                        src = m.get('ipv4_src')
                        dst = m.get('ipv4_dst')
                        if not src or not dst:
                            continue
                        byte_count = getattr(s, 'byte_count', 0) or 0

                        # We'll treat the ingress leaf as the leaf where the source host resides.
                        ingress_leaf = HOST_TO_LEAF.get(src)
                        if ingress_leaf is None:
                            # don't know the host mapping
                            continue


                        # key uses ingress_leaf (promote on the ingress leaf)
                        key_prev = ((src, dst), ingress_leaf)
                        prev_bytes = self.flow_prev_bytes.get(key_prev)
                        if prev_bytes is None:
                            # initialize; next poll will produce delta
                            self.flow_prev_bytes[key_prev] = byte_count
                            continue

                        delta = max(0, byte_count - prev_bytes)

                        # self.logger.info("delta: %s, promoted: %s",delta, self.promoted_flows)

                        # update prev here so we don't repeatedly re-promote on the same delta
                        self.flow_prev_bytes[key_prev] = byte_count

                        # condition for promotion
                        if (delta >= ELEPHANT_BYTES): #or byte_count >= ELEPHANT_BYTES):
                            # only promote once per (flow, ingress_leaf)
                            if key_prev in self.promoted_flows:
                                continue

                            # perform promotion on the ingress leaf datapath
                            dp_ing = self.datapaths.get(ingress_leaf)
                            if dp_ing is None:
                                self.logger.warning("Promotion skipped: ingress leaf datapath %s not found", ingress_leaf)
                                continue

                            candidate_ports = LEAF_SPINE_PORTS.get(ingress_leaf, [])
                            if not candidate_ports:
                                self.logger.warning("Promotion skipped: no uplinks for leaf %s", ingress_leaf)
                                continue

                            # try to get policy probs for weights; fall back to uniform
                            probs = None
                            try:
                                # build a minimal state for policy: per-uplink util normalized
                                state_list = []
                                with self.lock:
                                    for p in candidate_ports:
                                        entry = self.port_stats.get(ingress_leaf, {}).get(p, {})
                                        util = entry.get('util', 0.0)
                                        drop = float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0)) 
                                        loss = min(drop / 1000.0, 1.0)
                                        state_list.append(min(util / CAPACITY_Mbps, 1.0))
                                        state_list.append(loss)
                                state_np = np.array(state_list, dtype=np.float32)
                                probs, _ = self.model.policy(state_np)
                                flow_key = (src,dst)
                                chosen_idx = int(np.argmax(probs))
                                meta = {
                                    'flow_key': flow_key,
                                    'dpid': ingress_leaf,
                                    'state': state_np,
                                    'action_probs': probs.copy() if isinstance(probs, np.ndarray) else np.array(probs, dtype=np.float32),
                                    'chosen_idx': chosen_idx,
                                    'time': time.time(),
                                    'type': 'elephant',
                                    'candidate_ports': candidate_ports
                                }
                                with self.lock:
                                    self.flow_memory[flow_key] = meta

                            except Exception:
                                probs = None

                            if probs is None:
                                probs = np.ones(len(candidate_ports), dtype=float) / float(len(candidate_ports))

                            weights = (probs * GROUP_WEIGHT_SCALE).astype(int)
                            # weights = [max(1, int(w)) for w in weights_np]
                            weights = np.maximum(weights, 1)

                            gid = self._group_id_from_flow((src, dst))
                            # install select group & flow -> group on ingress leaf
                            try:
                                self.logger.info("%s , %s , %s , %s ", dp_ing.id, gid, candidate_ports, weights)
                                self._install_select_group(dp_ing, gid, candidate_ports, weights)
                                self._install_flow_to_group(dp_ing, src, dst, gid, idle_timeout=FLOW_IDLE_TIMEOUT)
                            except Exception as e:
                                self.logger.exception("Error installing group for promotion: %s", e)
                                continue

                            # bookkeeping
                            with self.lock:
                                self.promoted_flows.add(key_prev)
                                self.promoted_meta[key_prev] = {
                                    'gid': int(gid),
                                    'last_bytes': int(byte_count),
                                    'missing': 0,
                                    'inactive': 0,
                                    'dpid': ingress_leaf
                                }
                                self.groups_last_used[(ingress_leaf, gid)] = time.time()
                            self.logger.info("PROMOTED flow %s->%s on leaf %s (delta=%d bytes, abs=%d) gid=%s weights=%s",
                                             src, dst, ingress_leaf, delta, byte_count, gid, weights.tolist())
                    except Exception:
                        # skip this stat entry on any unexpected issue
                        continue
                
        except Exception:
            self.logger.exception("Error during polling-based promotion handling for dpid %s", dpid)

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
        # --- cleanup loop for currently promoted flows (on this leaf) ---
        # parameters:
        MISSING_LIMIT = 2      # number of consecutive polls a flow can be absent before removal
        INACTIVITY_LIMIT = 3   # number of consecutive polls with zero delta before removal

        # iterate only promoted flows relevant to this dpid
        with self.lock:
            promoted_keys = [k for k in self.promoted_meta.keys() if k[1] == dpid]

        for flow_id_key in promoted_keys:
            try:
                meta = None
                with self.lock:
                    meta = self.promoted_meta.get(flow_id_key)
                if meta is None:
                    continue

                gid = int(meta.get("gid"))
                current_bytes = 0
                # attempt to read group byte_count directly by gid
                if gid in group_by_gid:
                    current_bytes = group_by_gid[gid]
                    # found -> reset missing
                    with self.lock:
                        self.promoted_meta[flow_id_key]['missing'] = 0
                else:
                    # group not present in this reply (maybe switch didn't include it)
                    with self.lock:
                        self.promoted_meta[flow_id_key]['missing'] = self.promoted_meta[flow_id_key].get('missing', 0) + 1
                        missing_cnt = self.promoted_meta[flow_id_key]['missing']
                    if missing_cnt >= MISSING_LIMIT:
                        self.logger.info("Missing Promoted flow %s missing for %d polls -> removing promotion",
                                        flow_id_key, missing_cnt)
                        self._remove_promotion(flow_id_key)
                    # done with this flow for this poll
                    continue

                # Now we have current_bytes (from group stats). Compute delta vs last_bytes.
                last_bytes = 0
                with self.lock:
                    last_bytes = int(self.promoted_meta[flow_id_key].get('last_bytes', 0))

                self.logger.info("currrent bytes: %s, last bytes: %s, promoted meta: %s",current_bytes,last_bytes,self.promoted_meta)

                delta_bytes = max(0, int(current_bytes) - int(last_bytes))
                # update last_bytes to the latest observed group byte_count
                self.promoted_meta[flow_id_key]['last_bytes'] = int(current_bytes)

                if delta_bytes == 0:
                    with self.lock:
                        self.promoted_meta[flow_id_key]['inactive'] = self.promoted_meta[flow_id_key].get('inactive', 0) + 1
                else:
                    with self.lock:
                        self.promoted_meta[flow_id_key]['inactive'] = 0

                # if it has been inactive for too long, remove promotion
                with self.lock:
                    inactive_cnt = self.promoted_meta[flow_id_key].get('inactive', 0)
                if inactive_cnt >= INACTIVITY_LIMIT:
                    self.logger.info("Inactive Promoted flow %s inactive for %d polls -> removing promotion",
                                    flow_id_key, inactive_cnt)
                    self._remove_promotion(flow_id_key)

            except Exception:
                self.logger.exception("Error while cleaning promoted flow %s", flow_id_key)

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
    # (left mostly unchanged; uses _is_elephant only as a hint, promotion handled by polling)
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
                    print(host_ports)
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

        # build state vector (one util per candidate port)
        with self.lock:
            state = []
            for p in candidate_ports:
                entry = self.port_stats.get(ingress_leaf, {}).get(p, {})
                util = entry.get('util', 0.0)
                drop = float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0)) 
                loss = min(drop / 1000.0, 1.0)
                state.append(min(util / CAPACITY_Mbps, 1.0))
                state.append(loss)
            state_np = np.array(state, dtype=np.float32)

        # elephant detection hint (legacy). Promotion is performed by polling.
        # is_elephant = self._is_elephant(flow_key, ingress_leaf)

        # policy (keeps the RL model unchanged)
        probs, action_idx = self.model.policy(state_np)
        self.logger.info("probs: %s",probs)

        # ensure dp_ing exists
        dp_ing = self.datapaths.get(ingress_leaf)
        if dp_ing is None:
            self.logger.warning("Ingress leaf %s not in datapaths; flooding", ingress_leaf)
            out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id, in_port=msg.match.get('in_port'), actions=[parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)], data=msg.data)
            dp_packetin.send_msg(out)
            return

        # chosen uplink port on ingress leaf
        chosen_idx = int(np.argmax(probs))
        chosen_port = candidate_ports[chosen_idx]

        # install on ingress leaf
        # if not is_elephant:
            # mice: install direct flow to chosen uplink port
        self._install_simple_flow(dp_ing, src, dst, chosen_port, idle_timeout=FLOW_IDLE_TIMEOUT)
        flow_type = 'mice'
        self.logger.info("MICE %s->%s chosen port %s at leaf %s", src, dst, chosen_port, ingress_leaf)
        actions_out = [dp_ing.ofproto_parser.OFPActionOutput(chosen_port)]
        # else:
            # elephant path (packet_in-level fallback) - group install still allowed, but main promotion done by polling
            # weights_np = (probs * GROUP_WEIGHT_SCALE).astype(int)
            # weights = [max(1, int(w)) for w in weights_np]
            # # weights = np.maximum(weights, 1)
            # gid = self._group_id_from_flow(flow_key)
            # self._install_select_group(dp_ing, gid, candidate_ports, weights)
            # self._install_flow_to_group(dp_ing, src, dst, gid, idle_timeout=FLOW_IDLE_TIMEOUT)
            # flow_type = 'elephant'
            # self.logger.info("ELEPHANT (packet_in) %s->%s group %s ports %s weights %s at leaf %s", src, dst, gid, candidate_ports, weights.tolist(), ingress_leaf)
            # actions_out = [dp_ing.ofproto_parser.OFPActionGroup(gid)]
            # with self.lock:
            #     self.groups_last_used[(ingress_leaf, gid)] = time.time()

        # ---------- INSTALL spine + egress leaf rules ----------
        spine_dpid = SPINES[chosen_idx] if chosen_idx < len(SPINES) else SPINES[0]
        dp_spine = self.datapaths.get(spine_dpid)
        if dp_spine is not None:
            spine_out_port = SPINE_TO_LEAF_PORTS.get(spine_dpid, {}).get(dst_leaf)
            if spine_out_port is not None:
                self._install_simple_flow(dp_spine, src, dst, spine_out_port, idle_timeout=FLOW_IDLE_TIMEOUT)

        # install egress leaf -> host flow
        dp_dst = self.datapaths.get(dst_leaf)
        dst_host_port = self._get_host_port(dst_leaf, dst)
        if dp_dst is not None and dst_host_port is not None:
            self._install_simple_flow(dp_dst, src, dst, dst_host_port, idle_timeout=FLOW_IDLE_TIMEOUT)

        # also install reverse path entries:
        if dp_spine is not None:
            spine_back_port = SPINE_TO_LEAF_PORTS.get(spine_dpid, {}).get(ingress_leaf)
            if spine_back_port is not None:
                self._install_simple_flow(dp_spine, dst, src, spine_back_port, idle_timeout=FLOW_IDLE_TIMEOUT)

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

        # store metadata for training/reward
        meta = {
            'flow_key': flow_key,
            'dpid': ingress_leaf,
            'state': state_np,
            'action_probs': probs.copy() if isinstance(probs, np.ndarray) else np.array(probs, dtype=np.float32),
            'chosen_idx': chosen_idx,
            'time': time.time(),
            'type': flow_type,
            'candidate_ports': candidate_ports
        }
        with self.lock:
            self.flow_memory[flow_key] = meta

        # send the current packet out from ingress leaf (use controller as in_port)
        try:
            parser_ing = dp_ing.ofproto_parser
            ofp_ing = dp_ing.ofproto
            out = parser_ing.OFPPacketOut(datapath=dp_ing, buffer_id=ofp_ing.OFP_NO_BUFFER, in_port=ofp_ing.OFPP_CONTROLLER, actions=actions_out, data=msg.data)
            dp_ing.send_msg(out)
        except Exception as e:
            self.logger.debug("PacketOut to ingress leaf failed: %s", e)

    # -------------------------
    # Flow detection helper
    # -------------------------
    def _is_elephant(self, flow_key, dpid):
        stats = []
        with self.lock:
            stats = list(self.flow_stats_cache.get(dpid, []))
        src, dst = flow_key
        byte_count = None
        for s in stats:
            try:
                m = getattr(s, 'match', {}) or {}
                if m.get('ipv4_src') == src and m.get('ipv4_dst') == dst:
                    byte_count = getattr(s, 'byte_count', 0) or 0
                    break
            except Exception:
                continue
        if byte_count is None:
            return False
        key = ((src, dst), dpid)
        prev = self.flow_prev_bytes.get(key)
        if prev is None:
            self.flow_prev_bytes[key] = byte_count
        return byte_count >= ELEPHANT_BYTES

    # -------------------------
    # Flow & group install helpers
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
        with self.lock:
            self.groups_last_used[(datapath.id, group_id)] = time.time()

    def _install_flow_to_group(self, datapath, src, dst, group_id, priority=300, idle_timeout=0):
        if datapath is None:
            return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto

        match = parser.OFPMatch(eth_type=0x0800,
                                ipv4_src=src,
                                ipv4_dst=dst)

        actions = [parser.OFPActionGroup(group_id)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofp.OFPFC_ADD,     # ensure replace via priority
            priority=priority,         # higher than mice flow
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=0
        )
        datapath.send_msg(mod)
    # -------------------------
    # Reward computation & trainer
    # -------------------------
    def _compute_reward(self, flow_key, meta):
        dpid = meta['dpid']
        candidate_ports = meta['candidate_ports']
        src_ip, dst_ip = meta.get('flow_key', (None, None))
        if src_ip is None:
            return None

        with self.lock:
            stats = list(self.flow_stats_cache.get(dpid, []))
        byte_count = None
        for s in stats:
            m = getattr(s, 'match', {}) or {}
            if m.get('ipv4_src') == src_ip and m.get('ipv4_dst') == dst_ip:
                byte_count = getattr(s, 'byte_count', 0) or 0
                break

        key_prev = ((src_ip, dst_ip), dpid)
        prev_bytes = self.flow_prev_bytes.get(key_prev, None)
        if prev_bytes is None:
            if byte_count is not None:
                self.flow_prev_bytes[key_prev] = byte_count
            return None
        if byte_count is None:
            return None

        interval = POLL_INTERVAL
        bytes_delta = max(0, byte_count - prev_bytes)
        throughput_mbps = (bytes_delta * 8.0) / (interval * 1e6)
        flow_throughput_norm = min(throughput_mbps / CAPACITY_Mbps, 1.0)
        self.flow_prev_bytes[key_prev] = byte_count

        with self.lock:
            utils = [min(self.port_stats.get(dpid, {}).get(p, {}).get('util', 0.0) / CAPACITY_Mbps, 1.0) for p in candidate_ports]
        if not utils:
            return None
        util_skew = float(np.std(utils))

        with self.lock:
            drop_sum = 0.0
            for p in candidate_ports:
                entry = self.port_stats.get(dpid, {}).get(p, {})
                drop_sum += float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0))
        packet_loss_norm = min(drop_sum / 1000.0, 1.0)

        latency_penalty = 0.0

        reward = (ALPHA * flow_throughput_norm) - (BETA * util_skew) - (GAMMA * packet_loss_norm) - (DELTA * latency_penalty)
        reward = max(-1.0, min(1.0, reward))
        with open("rl_reward_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), reward])
        return reward

    def _trainer(self):
        while True:
            with self.lock:
                keys = list(self.flow_memory.keys())
            for k in keys:
                with self.lock:
                    meta = self.flow_memory.get(k)
                if not meta:
                    continue
                if time.time() - meta['time'] >= max(1.5, POLL_INTERVAL):
                    r = self._compute_reward(k, meta)
                    if r is not None:
                        s = meta['state']
                        # a = meta['action_idx']
                        a_probs = meta['action_probs']
                        # Convert to a deterministic dtype (plain numpy float32) so the queue stores serializable objects
                        if a_probs is None:
                            # fallback: build one-hot from chosen_idx
                            chosen_idx = int(meta.get('chosen_idx', 0))
                            a_probs = np.zeros(len(meta['candidate_ports']), dtype=np.float32)
                            a_probs[chosen_idx] = 1.0
                        else:
                            a_probs = np.asarray(a_probs, dtype=np.float32)
                        dpid = meta['dpid']
                        candidate_ports = meta['candidate_ports']
                        with self.lock:
                            next_state = []#[min(self.port_stats.get(dpid, {}).get(p, {}).get('util', 0.0) / CAPACITY_Mbps, 1.0) for p in candidate_ports]
                            for p in candidate_ports:
                                        entry = self.port_stats.get(dpid, {}).get(p, {})
                                        util = entry.get('util', 0.0)
                                        drop = float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0)) 
                                        loss = min(drop / 1000.0, 1.0)
                                        next_state.append(min(util / CAPACITY_Mbps, 1.0))
                                        next_state.append(loss)
                        next_state_np = np.array(next_state, dtype=np.float32)
                        done = False
                        logp_placeholder = None
                        self.transitions.append((s, a_probs, r, next_state_np, done, logp_placeholder))
                        with self.lock:
                            try:
                                del self.flow_memory[k]
                            except KeyError:
                                pass
            if len(self.transitions) >= TRAIN_BATCH_SIZE:
                batch = [self.transitions.popleft() for _ in range(min(len(self.transitions), TRAIN_BATCH_SIZE))]
                res = self.model.update(batch)
                if res is not None:
                    try:
                        actor_loss, critic_loss = res
                        self.logger.info("RL update: actor_loss=%.4f critic_loss=%.4f", actor_loss, critic_loss)
                        with open("rl_training_log.csv", "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([time.time(), actor_loss, critic_loss])

                    except Exception:
                        pass

                    # ---- SAVE MODEL ----
                    try:
                        self.rl_update_step += 1
                        if self.rl_update_step % MODEL_SAVE_STEPS == 0:
                            self.model.save_checkpoint(f"{CHECKPOINT_DIR}/rl_latest.pt")
                            self.logger.info("[RL] Model saved to checkpoints/rl_latest.pt")
                    except Exception as e:
                        self.logger.error("Failed to save RL model: %s", e)
                    
            hub.sleep(POLL_INTERVAL)

    # -------------------------
    # Group cleanup
    # -------------------------
    def _group_cleaner(self):
        while True:
            now = time.time()
            stale = []
            with self.lock:
                for (dpid, gid), ts in list(self.groups_last_used.items()):
                    if now - ts > GROUP_IDLE_TIMEOUT:
                        stale.append((dpid, gid))
            for dpid, gid in stale:
                dp = self.datapaths.get(dpid)
                if dp is None:
                    with self.lock:
                        if (dpid, gid) in self.groups_last_used:
                            del self.groups_last_used[(dpid, gid)]
                    continue
                parser = dp.ofproto_parser
                ofp = dp.ofproto
                try:
                    grp_del = parser.OFPGroupMod(datapath=dp, command=ofp.OFPGC_DELETE, type_=ofp.OFPGT_ALL, group_id=gid, buckets=[])
                    dp.send_msg(grp_del)
                except Exception:
                    pass
                with self.lock:
                    if (dpid, gid) in self.groups_last_used:
                        del self.groups_last_used[(dpid, gid)]
            hub.sleep(GROUP_IDLE_TIMEOUT / 4.0)

    def _remove_promotion(self, flow_id_key):
        """Tear down group and flow->group for a promoted flow, and clean bookkeeping.
        flow_id_key = ((src, dst), ingress_leaf)
        """
        with self.lock:
            meta = self.promoted_meta.get(flow_id_key)
            if not meta:
                # maybe already removed
                if flow_id_key in self.promoted_flows:
                    self.promoted_flows.discard(flow_id_key)
                return

            gid = int(meta.get('gid'))
            dpid = meta.get('dpid')
            # remove from sets/maps
            self.promoted_flows.discard(flow_id_key)
            self.flow_prev_bytes[flow_id_key] = None
            try:
                del self.promoted_meta[flow_id_key]
            except KeyError:
                pass
            if (dpid, gid) in self.groups_last_used:
                try:
                    del self.groups_last_used[(dpid, gid)]
                except KeyError:
                    pass

        # send Group DELETE and Flow DELETE outside lock
        dp = self.datapaths.get(dpid)
        if dp is None:
            self.logger.info("Remove promotion: datapath %s not present (already gone)", dpid)
            return

        parser = dp.ofproto_parser
        ofp = dp.ofproto

        # delete group
        try:
            grp_del = parser.OFPGroupMod(datapath=dp, command=ofp.OFPGC_DELETE,
                                        type_=ofp.OFPGT_ALL, group_id=gid, buckets=[])
            dp.send_msg(grp_del)
            self.logger.info("Deleted group %s on dpid %s for promotion removal", gid, dpid)
        except Exception:
            self.logger.exception("Failed to delete group %s on dpid %s", gid, dpid)

        # delete flow -> group rule (match ipv4 src/dst)
        try:
            src_dst = flow_id_key[0]
            src_ip, dst_ip = src_dst
            match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip, ipv4_dst=dst_ip)
            # Use command DELETE to remove matching flows
            fm = parser.OFPFlowMod(datapath=dp,
                                command=ofp.OFPFC_DELETE,
                                out_port=ofp.OFPP_ANY,
                                out_group=ofp.OFPG_ANY,
                                match=match,
                                table_id=0)
            dp.send_msg(fm)
            self.logger.info("Deleted flow->group entry for %s on dpid %s", src_dst, dpid)
        except Exception:
            self.logger.exception("Failed to delete flow->group entry for %s on dpid %s", src_dst, dpid)

