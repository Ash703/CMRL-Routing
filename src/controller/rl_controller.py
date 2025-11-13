# src/controller/rl_ryu_controller_config_patched.py
"""
Patched Ryu controller (fixed datapath installs, deltas, timeouts, cleanup).
Designed for the YAML topology (s11/s12 = 11/12 spines, s21/s22/s23 = 21/22/23 leaves).
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
from ryu.lib.packet import packet, ethernet, ipv4

# Import ActorCritic implementation — keep the same import you used earlier.
# Adjust if your PYTHONPATH differs.
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
FLOW_IDLE_TIMEOUT = 30              # seconds idle timeout for per-flow rules
GROUP_IDLE_TIMEOUT = 300            # seconds to keep unused groups before deletion

# Reward coefficients (alpha, beta, gamma, delta)
ALPHA = 1.0   # throughput gain weight
BETA  = 1.0   # utilization skew penalty
GAMMA = 1.0   # packet loss penalty
DELTA = 0.0   # latency penalty (0 by default)

# Topology mapping derived from your YAML config
LEAF_SPINE_PORTS = {21: [1, 2], 22: [1, 2], 23: [1, 2]}
HOST_TO_LEAF = {
    "10.1.1.1": 21, "10.1.1.2": 21,
    "10.1.1.3": 22, "10.1.1.4": 22,
    "10.1.1.5": 23, "10.1.1.6": 23,
}

HOST_PORT = {
    "10.1.1.1": 3,
    "10.1.1.2": 4,
    "10.1.1.3": 3,
    "10.1.1.4": 4,
    "10.1.1.5": 3,
    "10.1.1.6": 4
}



class RLDCController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RLDCController, self).__init__(*args, **kwargs)
        # bookkeeping
        self.datapaths = {}              # dpid -> datapath
        self.port_stats = {}             # dpid -> port -> {'tx','rx','ts','util','tx_dropped','rx_dropped','tx_dropped_delta','rx_dropped_delta'}
        self.flow_stats_cache = {}       # dpid -> list of flow stats (latest)
        self.flow_prev_bytes = {}        # ((src,dst), dpid) -> previous byte_count (for throughput delta)
        # memory for flows waiting reward computation
        self.flow_memory = {}            # (src,dst) -> metadata (contains 'flow_key')
        self.transitions = deque(maxlen=5000)

        # groups bookkeeping for cleanup: (dpid, group_id) -> last_used_ts
        self.groups_last_used = {}

        # thread lock for shared structures
        self.lock = Lock()

        # RL model: input_dim = number of candidate paths (we use 2 spines)
        num_paths = 2
        self.model = ActorCritic(input_dim=num_paths, num_actions=num_paths, device=DEVICE)

        # spawn monitor & trainer threads & group cleaner
        self.monitor_thread = hub.spawn(self._monitor)
        self.trainer_thread = hub.spawn(self._trainer)
        self.cleaner_thread = hub.spawn(self._group_cleaner)
        self.logger.info("RLDCController (patched) initialized")

    
    def _get_host_port(self, leaf, host_ip):
        return HOST_PORT[host_ip]

    # -------------------------
    # Table-miss: install on switch features
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        # Table-miss flow entry: send unmatched packets to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=inst)
        dp.send_msg(mod)

        self.logger.info("Installed table-miss on switch %s", dp.id)

    # -------------------------
    # OF event: switches connect/disconnect
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
    # Polling for port & flow stats
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
            self.flow_stats_cache[dpid] = ev.msg.body

    # -------------------------
    # Packet-in: decide path and install flows/groups
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        msg = ev.msg
        dp_packetin = msg.datapath           # datapath that sent this PacketIn
        parser_pi = dp_packetin.ofproto_parser
        ofp_pi = dp_packetin.ofproto

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        # ARP: flood (fast path) so hosts can learn each other's MACs
        if eth and eth.ethertype == 0x0806:  # ARP
            self.logger.info("ARP %s -> broadcasting across all switches", dp_packetin.id)

            data = msg.data if msg.buffer_id == ofp_pi.OFP_NO_BUFFER else None

            # Broadcast to ALL datapaths (all switches)
            for dpid, dp in list(self.datapaths.items()):
                parser_dp = dp.ofproto_parser
                ofp_dp = dp.ofproto

                ports = list(self.port_stats.get(dpid, {}).keys())
                if ports:
                    actions = [parser_dp.OFPActionOutput(p) for p in ports if p > 0]
                else:
                    # fallback flood
                    actions = [parser_dp.OFPActionOutput(ofp_dp.OFPP_FLOOD)]

                out = parser_dp.OFPPacketOut(
                    datapath=dp,
                    buffer_id=ofp_dp.OFP_NO_BUFFER,
                    in_port=ofp_dp.OFPP_CONTROLLER,
                    actions=actions,
                    data=data
                )
                dp.send_msg(out)

            return
        
        ip = pkt.get_protocol(ipv4.ipv4)
        if not ip:
            return

        src = ip.src
        dst = ip.dst
        flow_key = (src, dst)

        # Determine ingress leaf dpid for this src
        ingress_leaf = HOST_TO_LEAF.get(src)
        if ingress_leaf is None:
            # unknown host mapping (out of our YAML); fallback flood
            actions = [parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)]
            data = msg.data if msg.buffer_id == ofp_pi.OFP_NO_BUFFER else None
            out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id,
                                         in_port=msg.match['in_port'], actions=actions, data=data)
            dp_packetin.send_msg(out)
            return

        candidate_ports = LEAF_SPINE_PORTS.get(ingress_leaf, [])
        if not candidate_ports:
            actions = [parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)]
            data = msg.data if msg.buffer_id == ofp_pi.OFP_NO_BUFFER else None
            out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id,
                                         in_port=msg.match['in_port'], actions=actions, data=data)
            dp_packetin.send_msg(out)
            return
        
        dst_leaf = HOST_TO_LEAF.get(dst)
        if dst_leaf is not None and dst_leaf == ingress_leaf:
            # get datapath for ingress leaf
            dp_ing = self.datapaths.get(ingress_leaf)
            if dp_ing is None:
                # fallback: flood via packet-in datapath
                self.logger.warning("Ingress leaf %s not in datapaths for same-leaf case; flooding", ingress_leaf)
                actions = [parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)]
                out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id,
                                            in_port=msg.match['in_port'], actions=actions, data=msg.data)
                dp_packetin.send_msg(out)
                return

            # find host ports (helper _get_host_port defined below)
            out_port = self._get_host_port(ingress_leaf, dst)
            in_port = self._get_host_port(ingress_leaf, src)

            # install forwarding both directions on the leaf with idle timeout
            self._install_simple_flow(dp_ing, src, dst, out_port, idle_timeout=FLOW_IDLE_TIMEOUT)
            self._install_simple_flow(dp_ing, dst, src, in_port, idle_timeout=FLOW_IDLE_TIMEOUT)

            self.logger.info("SAME-LEAF %s->%s installed on leaf %s ports %s<->%s",
                            src, dst, ingress_leaf, in_port, out_port)

            # immediately forward the current packet to the out_port using the packet-in datapath
            actions_out = [parser_pi.OFPActionOutput(out_port)]
            data = msg.data if msg.buffer_id == ofp_pi.OFP_NO_BUFFER else None
            out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id,
                                        in_port=msg.match['in_port'], actions=actions_out, data=data)
            dp_packetin.send_msg(out)
            return

        # Build state using ingress_leaf port utils — ensure we read under lock
        with self.lock:
            state = []
            for p in candidate_ports:
                util = self.port_stats.get(ingress_leaf, {}).get(p, {}).get('util', 0.0)
                state.append(min(util / CAPACITY_Mbps, 1.0))
        state_np = np.array(state, dtype=np.float32)

        # Elephant detection using FlowStats cache on ingress_leaf datapath
        is_elephant = self._is_elephant(flow_key, ingress_leaf)

        # Query policy (actor-critic)
        probs, action_idx = self.model.policy(state_np)

        # Use the ingress leaf datapath object for installs
        dp_ing = self.datapaths.get(ingress_leaf)
        if dp_ing is None:
            # fallback: flood via packet-in datapath and log
            self.logger.warning("Ingress leaf %s not in datapaths; flooding packet", ingress_leaf)
            actions = [parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)]
            out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id,
                                         in_port=msg.match['in_port'], actions=actions, data=msg.data)
            dp_packetin.send_msg(out)
            return


        if not is_elephant:
            # Mice -> single path (argmax)
            chosen_idx = int(np.argmax(probs))
            chosen_port = candidate_ports[chosen_idx]
            # install flow on ingress leaf datapath with idle_timeout
            self._install_simple_flow(dp_ing, src, dst, chosen_port, idle_timeout=FLOW_IDLE_TIMEOUT)
            flow_type = 'mice'
            self.logger.info("MICE %s->%s chosen port %s at leaf %s", src, dst, chosen_port, ingress_leaf)
            # send packet out through packet-in datapath to chosen port (could also send via dp_ing)
            actions_out = [parser_pi.OFPActionOutput(chosen_port)]
        else:
            # Elephant -> install SELECT group with weights proportional to probs
            weights = (probs * GROUP_WEIGHT_SCALE).astype(int)
            weights = np.maximum(weights, 1)
            gid = self._group_id_from_flow(flow_key)
            # install group on ingress leaf datapath (use parser_ing)
            self._install_select_group(dp_ing, gid, candidate_ports, weights)
            # install flow that forwards to group (on ingress)
            self._install_flow_to_group(dp_ing, src, dst, gid, idle_timeout=FLOW_IDLE_TIMEOUT)
            flow_type = 'elephant'
            self.logger.info("ELEPHANT %s->%s group %s ports %s weights %s at leaf %s",
                             src, dst, gid, candidate_ports, weights.tolist(), ingress_leaf)
            actions_out = [parser_pi.OFPActionGroup(gid)]
            # mark group last used now
            with self.lock:
                self.groups_last_used[(ingress_leaf, gid)] = time.time()

        # store metadata for training/reward — detach logp tensor to avoid graph issues
        meta = {
            'flow_key': flow_key,
            'dpid': ingress_leaf,
            'state': state_np,
            'action_idx': action_idx,
            # 'logp': logp.detach() if hasattr(logp, 'detach') else logp,
            'time': time.time(),
            'type': flow_type,
            'candidate_ports': candidate_ports
        }
        with self.lock:
            self.flow_memory[flow_key] = meta

        # send the current packet out using packet-in datapath
        data = msg.data if msg.buffer_id == ofp_pi.OFP_NO_BUFFER else None
        out = parser_pi.OFPPacketOut(datapath=dp_packetin,
                                     buffer_id=msg.buffer_id,
                                     in_port=msg.match['in_port'],
                                     actions=actions_out,
                                     data=data)
        dp_packetin.send_msg(out)

    # -------------------------
    # Flow detection helper
    # -------------------------
    def _is_elephant(self, flow_key, dpid):
        """Check cached FlowStats for flow (src,dst) on dpid. If byte_count >= threshold -> elephant.
           Also initializes flow_prev_bytes for future throughput calculation (but does not overwrite if missing)."""
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
            # record initial byte count and wait for next interval to compute throughput
            self.flow_prev_bytes[key] = byte_count
        # decide elephant based on absolute byte_count (not delta) — you can change policy
        return byte_count >= ELEPHANT_BYTES

    # -------------------------
    # Flow & group install helpers
    # -------------------------
    def _install_simple_flow(self, datapath, src, dst, out_port, priority=200, idle_timeout=0):
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        actions = [parser.OFPActionOutput(out_port)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                instructions=inst, idle_timeout=idle_timeout)
        datapath.send_msg(mod)

    def _install_select_group(self, datapath, group_id, out_ports, weights):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        buckets = []
        for p, w in zip(out_ports, weights):
            actions = [parser.OFPActionOutput(p)]
            buckets.append(parser.OFPBucket(weight=int(w), watch_port=p, watch_group=ofp.OFPG_ANY, actions=actions))
        # try modify then add
        grp_mod = parser.OFPGroupMod(datapath=datapath, command=ofp.OFPGC_MODIFY,
                                     type_=ofp.OFPGT_SELECT, group_id=group_id, buckets=buckets)
        try:
            datapath.send_msg(grp_mod)
        except Exception:
            grp_mod = parser.OFPGroupMod(datapath=datapath, command=ofp.OFPGC_ADD,
                                         type_=ofp.OFPGT_SELECT, group_id=group_id, buckets=buckets)
            datapath.send_msg(grp_mod)
        # update last used timestamp
        with self.lock:
            self.groups_last_used[(datapath.id, group_id)] = time.time()

    def _install_flow_to_group(self, datapath, src, dst, group_id, priority=200, idle_timeout=0):
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, [parser.OFPActionGroup(group_id)])]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                instructions=inst, idle_timeout=idle_timeout)
        datapath.send_msg(mod)

    def _group_id_from_flow(self, flow_key):
        return (abs(hash(flow_key)) % (2**31 - 1)) or 1

    # -------------------------
    # Reward computation & trainer
    # -------------------------
    def _compute_reward(self, flow_key, meta):
        """
        reward = alpha * throughput_gain
                 - beta * util_skew
                 - gamma * packet_loss
                 - delta * latency_penalty
        All sub-terms normalized to [0,1]; reward clipped to [-1,1].
        """
        dpid = meta['dpid']
        candidate_ports = meta['candidate_ports']

        # get (src,dst)
        src_ip, dst_ip = meta.get('flow_key', (None, None))
        if src_ip is None:
            return None

        # find flow byte_count from flow_stats_cache
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
            # store and wait
            if byte_count is not None:
                self.flow_prev_bytes[key_prev] = byte_count
            return None
        if byte_count is None:
            return None

        # compute throughput (Mbps) over POLL_INTERVAL
        interval = POLL_INTERVAL
        bytes_delta = max(0, byte_count - prev_bytes)
        throughput_mbps = (bytes_delta * 8.0) / (interval * 1e6)
        flow_throughput_norm = min(throughput_mbps / CAPACITY_Mbps, 1.0)
        # update prev
        self.flow_prev_bytes[key_prev] = byte_count

        # Link utilization skew
        with self.lock:
            utils = [min(self.port_stats.get(dpid, {}).get(p, {}).get('util', 0.0) / CAPACITY_Mbps, 1.0) for p in candidate_ports]
        if not utils:
            return None
        util_skew = float(np.std(utils))  # already approx 0..1

        # packet_loss: use deltas stored in port_stats
        with self.lock:
            drop_sum = 0.0
            for p in candidate_ports:
                entry = self.port_stats.get(dpid, {}).get(p, {})
                drop_sum += float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0))
        packet_loss_norm = min(drop_sum / 1000.0, 1.0)

        latency_penalty = 0.0  # optional probing can set this

        reward = (ALPHA * flow_throughput_norm) - (BETA * util_skew) - (GAMMA * packet_loss_norm) - (DELTA * latency_penalty)
        reward = max(-1.0, min(1.0, reward))
        return reward

    def _trainer(self):
        """Periodically evaluate stored flows and create transitions for training"""
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
                        a = meta['action_idx']
                        logp = None #ignored by ActorCritic now
                        dpid = meta['dpid']
                        candidate_ports = meta['candidate_ports']
                        # next state sample
                        with self.lock:
                            next_state = [min(self.port_stats.get(dpid, {}).get(p, {}).get('util', 0.0) / CAPACITY_Mbps, 1.0)
                                          for p in candidate_ports]
                        next_state_np = np.array(next_state, dtype=np.float32)
                        done = False
                        # append transition
                        self.transitions.append((s, a, r, next_state_np, done, logp))
                        # remove meta
                        with self.lock:
                            try:
                                del self.flow_memory[k]
                            except KeyError:
                                pass
            # train if enough transitions
            if len(self.transitions) >= TRAIN_BATCH_SIZE:
                batch = [self.transitions.popleft() for _ in range(min(len(self.transitions), TRAIN_BATCH_SIZE))]
                res = self.model.update(batch)
                if res is not None:
                    try:
                        actor_loss, critic_loss = res
                        self.logger.info("RL update: actor_loss=%.4f critic_loss=%.4f", actor_loss, critic_loss)
                    except Exception:
                        pass
            hub.sleep(POLL_INTERVAL)

    # -------------------------
    # Group cleanup thread
    # -------------------------
    def _group_cleaner(self):
        """Periodically remove groups that haven't been used for GROUP_IDLE_TIMEOUT seconds."""
        while True:
            now = time.time()
            stale = []
            with self.lock:
                for (dpid, gid), ts in list(self.groups_last_used.items()):
                    if now - ts > GROUP_IDLE_TIMEOUT:
                        stale.append((dpid, gid))
            for dpid, gid in stale:
                dp = self.datapaths.get(dpid)
                if not dp:
                    with self.lock:
                        del self.groups_last_used[(dpid, gid)]
                    continue
                parser = dp.ofproto_parser
                ofp = dp.ofproto
                try:
                    grp_del = parser.OFPGroupMod(datapath=dp, command=ofp.OFPGC_DELETE,
                                                 type_=ofp.OFPGT_ALL, group_id=gid, buckets=[])
                    dp.send_msg(grp_del)
                except Exception:
                    # ignore delete errors
                    pass
                with self.lock:
                    if (dpid, gid) in self.groups_last_used:
                        del self.groups_last_used[(dpid, gid)]
            hub.sleep(GROUP_IDLE_TIMEOUT / 4.0)
