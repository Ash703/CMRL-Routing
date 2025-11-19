# src/controller/rl_ryu_controller_final.py
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
"""
import time
import numpy as np
from collections import deque
from threading import Lock

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp

# RL model: ActorCritic expects policy(state_np) -> (probs, action_idx)
from src.rl.rl_model import ActorCritic

# ---------------------------
# Config (tweak as required)
# ---------------------------
POLL_INTERVAL = 2.0                 # seconds between port/flow polls
ELEPHANT_BYTES = 2 * 1024 * 1024    # 2 MB promotion threshold
CAPACITY_Mbps = 1000.0              # normalize Mbps by this (1 Gbps)
GROUP_WEIGHT_SCALE = 1000           # scale prob -> bucket weight
TRAIN_BATCH_SIZE = 8
DEVICE = 'cpu'
FLOW_IDLE_TIMEOUT = 30              # seconds idle timeout for per-flow rules (mice)
GROUP_IDLE_TIMEOUT = 300            # seconds to keep unused groups before deletion
FLOW_INACTIVE_TIMEOUT = 3

# Reward coefficients (alpha, beta, gamma, delta)
ALPHA = 1.0   # throughput gain weight
BETA  = 1.0   # utilization skew penalty
GAMMA = 1.0   # packet loss penalty
DELTA = 0.0   # latency penalty (0 by default)

# Topology mapping (from your YAML)
LEAF_SPINE_PORTS = {21: [1, 2], 22: [1, 2], 23: [1, 2]}   # leaf -> [uplink ports to spines]
HOST_TO_LEAF = {
    "10.1.1.1": 21, "10.1.1.2": 21,
    "10.1.1.3": 22, "10.1.1.4": 22,
    "10.1.1.5": 23, "10.1.1.6": 23,
}
HOST_PORT = {
    "10.1.1.1": 3, "10.1.1.2": 4,
    "10.1.1.3": 3, "10.1.1.4": 4,
    "10.1.1.5": 3, "10.1.1.6": 4,
}

# Spine ordering: candidate_ports order -> spine dpids
SPINES = [11, 12]

# SPINE -> leaf port mapping (from your YAML links)
SPINE_TO_LEAF_PORTS = {
    11: {21: 1, 22: 2, 23: 3},
    12: {21: 1, 22: 2, 23: 3},
}

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
        self.promoted_flows = set()      # set of ((src,dst), dpid) promoted to elephant

        self.lock = Lock()

        # RL model: input_dim = 6 for B ordering:
        # [ util_up1, loss_up1, util_up2, loss_up2, flow_size, last_action ]
        num_paths = len(SPINES)
        self.model = ActorCritic(input_dim=6, num_actions=num_paths, device=DEVICE)

        # threads
        self.monitor_thread = hub.spawn(self._monitor)
        self.trainer_thread = hub.spawn(self._trainer)
        self.cleaner_thread = hub.spawn(self._group_cleaner)
        self.logger.info("RLDCController (patched) initialized")

    # -------------------------
    # helpers
    # -------------------------
    def _get_host_port(self, leaf, host_ip):
        return HOST_PORT.get(host_ip)

    def _group_id_from_flow(self, flow_key):
        return (abs(hash(flow_key)) % (2**31 - 1)) or 1

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

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        with self.lock:
            # store raw OFPFlowStats entries for later analysis
            self.flow_stats_cache[dpid] = ev.msg.body

        # optional debug: don't spam in normal runs
        # self.logger.debug("FlowStatsReply: dpid=%s entries=%d", dpid, len(ev.msg.body))

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
                # print("Destination:",dst_leaf, "Source:",src_dpid)
                dp_target = self.datapaths.get(dst_leaf)
                if dp_target:
                    host_ports = [HOST_PORT[ip] for ip, leaf in HOST_TO_LEAF.items() if leaf == dst_leaf]
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

            if src_dpid in SPINE_TO_LEAF_PORTS:
                # print("Destination:",dst_leaf, "Source:",src_dpid)
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
            # print("Destination:",dst_leaf, "Source:",src_dpid)
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

            # install forward & reverse flows on the leaf (mice)
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

        # -------------------------
        # Build 6-dim state vector (ordering B):
        # [ util_up1, loss_up1, util_up2, loss_up2, flow_size, last_action ]
        # -------------------------
        # compute flow_size (bytes seen so far) from flow_stats_cache
        flow_size_bytes = 0
        with self.lock:
            stats_for_leaf = list(self.flow_stats_cache.get(ingress_leaf, []))
        for s in stats_for_leaf:
            try:
                m = getattr(s, 'match', {}) or {}
                if m.get('ipv4_src') == src and m.get('ipv4_dst') == dst:
                    flow_size_bytes = getattr(s, 'byte_count', 0) or 0
                    break
            except Exception:
                continue

        # last_action from previous meta if exists (per-flow); normalized
        prev_action_idx = 0
        with self.lock:
            prev_meta = self.flow_memory.get(flow_key)
            if prev_meta:
                prev_action_idx = prev_meta.get('action_idx', 0)

        # build state in correct order
        state = []
        with self.lock:
            for p in candidate_ports:
                entry = self.port_stats.get(ingress_leaf, {}).get(p, {})
                util = min(entry.get('util', 0.0) / CAPACITY_Mbps, 1.0)
                drop = float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0))
                loss = min(drop / 1000.0, 1.0)
                # group per port: util then loss
                state.append(util)
                state.append(loss)

        # flow_size normalized (scale by ELEPHANT_BYTES)
        flow_size_norm = min(float(flow_size_bytes) / float(ELEPHANT_BYTES), 1.0)
        state.append(flow_size_norm)

        # last_action normalized across actions (if only one path, 0.0)
        if len(candidate_ports) > 1:
            last_action_norm = float(prev_action_idx) / float(len(candidate_ports) - 1)
        else:
            last_action_norm = 0.0
        state.append(last_action_norm)

        state_np = np.array(state, dtype=np.float32)

        # policy (model uses the 6-dim state)
        probs, action_idx = self.model.policy(state_np)

        self.logger.info("RL STATE %s -> %s leaf=%s state=%s", src, dst, ingress_leaf, state_np.tolist())
        self.logger.info("RL POLICY probs=%s action_idx=%s", probs.tolist(), action_idx)

        # ensure dp_ing exists
        dp_ing = self.datapaths.get(ingress_leaf)
        if dp_ing is None:
            self.logger.warning("Ingress leaf %s not in datapaths; flooding", ingress_leaf)
            out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id, in_port=msg.match.get('in_port'), actions=[parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)], data=msg.data)
            dp_packetin.send_msg(out)
            return

        # chosen uplink port on ingress leaf (for mice) or action_idx used for group weights if later promoted
        chosen_idx = int(np.argmax(probs))
        chosen_port = candidate_ports[chosen_idx]
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

        # install on ingress leaf (mice path)
        self._install_simple_flow(dp_ing, src, dst, chosen_port, idle_timeout=FLOW_IDLE_TIMEOUT)
        flow_type = 'mice'
        self.logger.info("MICE %s->%s chosen port %s at leaf %s", src, dst, chosen_port, ingress_leaf)
        actions_out = [dp_ing.ofproto_parser.OFPActionOutput(chosen_port)]

        # store metadata for training/reward
        meta = {
            'flow_key': flow_key,
            'dpid': ingress_leaf,
            'state': state_np,
            'action_idx': chosen_idx,
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
    # Flow detection helper (kept for reference; trainer uses delta logic)
    # -------------------------
    def _is_elephant(self, flow_key, dpid):
        """Legacy check: returns True if absolute byte_count >= ELEPHANT_BYTES.
           The trainer uses more robust delta-based promotion logic instead."""
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
        return byte_count >= ELEPHANT_BYTES

    # -------------------------
    # Flow & group install helpers (fixed, safe versions)
    # -------------------------
    def _install_simple_flow(self, datapath, src, dst, out_port, priority=100, idle_timeout=0):
        """Install a stable per-flow forwarding rule (no hard_timeout)."""
        if datapath is None or out_port is None:
            return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        actions = [parser.OFPActionOutput(out_port)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        # hard_timeout set to 0 to keep flow stable (RL training relies on stable flows)
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                instructions=inst, idle_timeout=idle_timeout, hard_timeout=0)
        datapath.send_msg(mod)

    def _install_select_group(self, datapath, group_id, out_ports, weights):
        """Add (then modify) SELECT group with proper watch ports and guards."""
        if datapath is None:
            return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto

        buckets = []
        for p, w in zip(out_ports, weights):
            actions = [parser.OFPActionOutput(p)]
            # SELECT groups should use OFPP_ANY / OFPG_ANY for watch fields
            buckets.append(parser.OFPBucket(weight=int(w),
                                            watch_port=ofp.OFPP_ANY,
                                            watch_group=ofp.OFPG_ANY,
                                            actions=actions))
        # Always attempt ADD first, then MODIFY to update if it already exists.
        # Some switches/OVS respond via error messages for invalid MODIFY-first patterns.
        try:
            grp_add = parser.OFPGroupMod(datapath=datapath,
                                         command=ofp.OFPGC_ADD,
                                         type_=ofp.OFPGT_SELECT,
                                         group_id=group_id,
                                         buckets=buckets)
            datapath.send_msg(grp_add)
        except Exception:
            # in Ryu sending msg rarely raises; still attempt MODIFY afterwards
            pass

        # Also send MODIFY to update existing group (some switches accept both).
        try:
            grp_mod = parser.OFPGroupMod(datapath=datapath,
                                         command=ofp.OFPGC_MODIFY,
                                         type_=ofp.OFPGT_SELECT,
                                         group_id=group_id,
                                         buckets=buckets)
            datapath.send_msg(grp_mod)
        except Exception:
            pass

        with self.lock:
            self.groups_last_used[(datapath.id, group_id)] = time.time()

    def _install_flow_to_group(self, datapath, src, dst, group_id, priority=200, idle_timeout=0):
        """Install flow that forwards to a group (no hard_timeout)."""
        if datapath is None:
            return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, [parser.OFPActionGroup(group_id)])]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                instructions=inst, idle_timeout=idle_timeout, hard_timeout=0)
        datapath.send_msg(mod)

    # -------------------------
    # Reward computation & trainer (delta-based elephant promotion + RL training)
    # -------------------------
    def _compute_reward(self, flow_key, meta):
        """
        Compute reward for a single flow based on delta throughput (computed using
        flow_prev_bytes), port utils skew and packet drops.
        IMPORTANT: this function DOES NOT overwrite flow_prev_bytes except to
        initialize it when absent. The trainer is responsible for updating
        flow_prev_bytes after reading current byte counts.
        Returns a float reward in [-1, 1] or None if insufficient data.
        """
        dpid = meta['dpid']
        candidate_ports = meta['candidate_ports']
        src_ip, dst_ip = meta.get('flow_key', (None, None))
        if src_ip is None:
            return None

        # read current byte_count from latest flow_stats snapshot
        with self.lock:
            stats = list(self.flow_stats_cache.get(dpid, []))
        byte_count = None
        for s in stats:
            try:
                m = getattr(s, 'match', {}) or {}
                if m.get('ipv4_src') == src_ip and m.get('ipv4_dst') == dst_ip:
                    byte_count = getattr(s, 'byte_count', 0) or 0
                    break
            except Exception:
                continue

        if byte_count is None:
            # no stats yet
            return None

        # prev_bytes is read but NOT overwritten here
        key_prev = ((src_ip, dst_ip), dpid)
        prev_bytes = self.flow_prev_bytes.get(key_prev, None)
        if prev_bytes is None:
            # initialize prev_bytes so next poll will produce a delta; do not compute reward now
            self.flow_prev_bytes[key_prev] = byte_count
            return None

        # compute delta throughput over the poll interval
        bytes_delta = max(0, byte_count - prev_bytes)
        interval = POLL_INTERVAL
        throughput_mbps = (bytes_delta * 8.0) / (interval * 1e6)
        flow_throughput_norm = min(throughput_mbps / CAPACITY_Mbps, 1.0)

        # utilization skew across candidate uplinks (normalized)
        with self.lock:
            utils = [min(self.port_stats.get(dpid, {}).get(p, {}).get('util', 0.0) / CAPACITY_Mbps, 1.0)
                    for p in candidate_ports]
        if not utils:
            return None
        util_skew = float(np.std(utils))

        # packet-drop based loss signal (normalized)
        with self.lock:
            drop_sum = 0.0
            for p in candidate_ports:
                entry = self.port_stats.get(dpid, {}).get(p, {})
                drop_sum += float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0))
        packet_loss_norm = min(drop_sum / 1000.0, 1.0)

        # combine into reward (clamped)
        latency_penalty = 0.0  
        reward = (ALPHA * flow_throughput_norm) - (BETA * util_skew) - (GAMMA * packet_loss_norm) - (DELTA * latency_penalty)
        reward = max(-1.0, min(1.0, reward))
        return reward


    def _trainer(self):
        """
        Trainer loop:
        - iterate over flows in flow_memory after at least POLL_INTERVAL
        - compute reward (uses flow_prev_bytes for delta but does not update it)
        - read current byte_count and update flow_prev_bytes HERE (single source of truth)
        - detect elephant promotion based on delta/current bytes
        - build next_state and push transition
        - perform batched RL updates and log actor/critic losses
        """
        while True:
            with self.lock:
                keys = list(self.flow_memory.keys())

            for k in keys:
                with self.lock:
                    meta = self.flow_memory.get(k)
                if not meta:
                    continue

                # wait so flow_prev_bytes had a chance to be initialized by earlier polls
                if time.time() - meta['time'] < POLL_INTERVAL:
                    continue

                # compute reward (returns None if insufficient data)
                r = self._compute_reward(k, meta)
                self.logger.info("Reward: %s",r)
                if r is None:
                    # either no stats yet or prev initialization just happened
                    continue

                s = meta['state']                   # state is 6-dim np array
                a = meta['action_idx']
                dpid = meta['dpid']
                candidate_ports = meta['candidate_ports']
                src_ip, dst_ip = meta.get('flow_key', (None, None))

                # ---- read latest byte_count and update prev_bytes HERE ----
                current_bytes = None
                with self.lock:
                    stats = list(self.flow_stats_cache.get(dpid, []))
                for sfs in stats:
                    try:
                        m = getattr(sfs, 'match', {}) or {}
                        if m.get('ipv4_src') == src_ip and m.get('ipv4_dst') == dst_ip:
                            current_bytes = getattr(sfs, 'byte_count', 0) or 0
                            break
                    except Exception:
                        continue

                key_prev = ((src_ip, dst_ip), dpid)
                # if we couldn't read current_bytes, skip updating and skip transition
                if current_bytes is None:
                    continue

                # compute delta for promotion decision (use prev if present)
                prev = self.flow_prev_bytes.get(key_prev, None)
                delta_bytes = 0
                if prev is not None:
                    delta_bytes = max(0, current_bytes - prev)

                # now update prev_bytes so next trainer iteration will compute new delta correctly
                self.flow_prev_bytes[key_prev] = current_bytes

                # ---------- check for elephant promotion (delta-based) ----------
                flow_id_key = ((src_ip, dst_ip), dpid)
                self.logger.info("flow: %s , %s , %s , %s",delta_bytes,current_bytes,flow_id_key,self.promoted_flows)
                if ((delta_bytes >= ELEPHANT_BYTES) or (current_bytes >= ELEPHANT_BYTES)) and (flow_id_key not in self.promoted_flows):
                    self.logger.info("PROMOTING flow %s on dpid %s delta=%d current=%s", (src_ip, dst_ip), dpid, delta_bytes, str(current_bytes))
                    dp_ing = self.datapaths.get(dpid)
                    if dp_ing is not None:
                        probs_placeholder = None
                        try:
                            state_for_policy = meta.get('state', None)
                            if state_for_policy is not None:
                                probs_placeholder, _ = self.model.policy(state_for_policy)
                        except Exception:
                            probs_placeholder = None

                        if probs_placeholder is None:
                            probs_placeholder = np.ones(len(candidate_ports), dtype=float) / float(len(candidate_ports))

                        weights = (probs_placeholder * GROUP_WEIGHT_SCALE).astype(int)
                        weights = np.maximum(weights, 1)
                        self.logger.info("weights: %s",weights)
                        gid = self._group_id_from_flow((src_ip, dst_ip))
                        self._install_select_group(dp_ing, gid, candidate_ports, weights)
                        self._install_flow_to_group(dp_ing, src_ip, dst_ip, gid, idle_timeout=FLOW_IDLE_TIMEOUT)
                        with self.lock:
                            self.groups_last_used[(dp_ing.id, gid)] = time.time()
                            self.promoted_flows.add(flow_id_key)
                            # update stored meta type
                            meta['type'] = 'elephant'
                    else:
                        self.logger.warning("Cannot promote: datapath %s not found", dpid)

                # ---------- build next_state (6-dim) ----------
                next_state_list = []
                with self.lock:
                    for p in candidate_ports:
                        entry = self.port_stats.get(dpid, {}).get(p, {})
                        util = min(entry.get('util', 0.0) / CAPACITY_Mbps, 1.0)
                        drop = float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0))
                        loss = min(drop / 1000.0, 1.0)
                        next_state_list.append(util)
                        next_state_list.append(loss)

                    # flow_size for next_state (from latest view we already read)
                    flow_size_norm_ns = min(float(current_bytes) / float(ELEPHANT_BYTES), 1.0)
                    next_state_list.append(flow_size_norm_ns)

                    # last_action for next_state = the action we took (a), normalized
                    if len(candidate_ports) > 1:
                        last_action_norm_ns = float(a) / float(len(candidate_ports) - 1)
                    else:
                        last_action_norm_ns = 0.0
                    next_state_list.append(last_action_norm_ns)

                next_state_np = np.array(next_state_list, dtype=np.float32)

                # done flag and placeholder maintained for compatibility
                done = False
                logp_placeholder = None

                # append transition for training
                self.logger.info("Next state: %s",next_state_np)
                self.transitions.append((s, a, r, next_state_np, done, logp_placeholder))

                # # remove meta so we don't re-train on the same snapshot
                # with self.lock:
                #     try:
                #         del self.flow_memory[k]
                #     except KeyError:
                #         pass
                # DO NOT delete here. Only update meta['time']
                # with self.lock:
                #     meta['time'] = time.time()
                # ----- FLOW INACTIVE TIMEOUT HANDLING -----
                with self.lock:
                    # initialize inactivity counter if missing
                    if 'inactive' not in meta:
                        meta['inactive'] = 0

                    if delta_bytes == 0:
                        meta['inactive'] += 1
                    else:
                        meta['inactive'] = 0

                    # if flow remained inactive for N consecutive polls → remove it
                    if meta['inactive'] >= FLOW_INACTIVE_TIMEOUT:
                        self.logger.info("FLOW INACTIVE: removing %s after %d empty polls",
                                        (src_ip, dst_ip), meta['inactive'])
                        try:
                            del self.flow_memory[k]
                        except KeyError:
                            pass
                        continue  # skip building transition

                    # flow still active → update timestamp for next trainer cycle
                    meta['time'] = time.time()
                    self.flow_memory[k] = meta



            # perform batched RL update
            if len(self.transitions) >= TRAIN_BATCH_SIZE:
                batch = [self.transitions.popleft() for _ in range(min(len(self.transitions), TRAIN_BATCH_SIZE))]
                res = self.model.update(batch)
                if res is not None:
                    try:
                        actor_loss, critic_loss = res
                        self.logger.info("[RL-UPDATE] actor_loss=%.6f critic_loss=%.6f", actor_loss, critic_loss)
                    except Exception:
                        pass

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
