# src/controller/rl_ryu_controller_config.py
"""
Ryu controller for your YAML config:
- spines: s11 (11), s12 (12)
- leaves: s21 (21), s22 (22), s23 (23)
- hosts h1..h6 connected as in your YAML
Behavior:
- use only OpenFlow stats (PortStats & FlowStats)
- input vector: per-candidate-path utils + drop rates
- output: policy probabilities over candidate paths
- mice -> single path (argmax); elephant -> SELECT group with weights
- reward = alpha*throughput - beta*util_skew - gamma*packet_loss - delta*latency_penalty
"""

import time
import numpy as np
from collections import deque

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4

# Import ActorCritic implementation (from previous file)
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

# Reward coefficients (alpha, beta, gamma, delta)
ALPHA = 1.0   # throughput gain weight
BETA  = 1.0   # utilization skew penalty
GAMMA = 1.0   # packet loss penalty
DELTA = 0.0   # latency penalty (0 by default; enable optional probing to compute)

# Topology mapping derived from your YAML config
# Leaves and spines dpids (from your YAML)
SPINES = [11, 12]
LEAVES = [21, 22, 23]

# leaf -> spine-facing ports on that leaf (from your links section)
# according to your YAML:
# s11 -> s21 (s21 port 1), s12 -> s21 (s21 port 2) etc.
LEAF_SPINE_PORTS = {
    21: [1, 2],  # s21 ports that lead to spines (to s11 and s12)
    22: [1, 2],  # s22
    23: [1, 2],  # s23
}

# host -> leaf mapping (from your YAML)
HOST_TO_LEAF = {
    "10.1.1.1": 21,  # h1
    "10.1.1.2": 21,  # h2
    "10.1.1.3": 22,  # h3
    "10.1.1.4": 22,  # h4
    "10.1.1.5": 23,  # h5
    "10.1.1.6": 23,  # h6
}

class RLDCController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RLDCController, self).__init__(*args, **kwargs)
        # bookkeeping
        self.datapaths = {}              # dpid -> datapath
        self.port_stats = {}             # dpid -> port -> {'tx','rx','ts','util','tx_dropped','rx_dropped'}
        self.flow_stats_cache = {}       # dpid -> list of flow stats (latest)
        self.flow_prev_bytes = {}        # (src,dst,dpid) -> previous byte_count (for throughput delta)
        # memory for flows waiting reward computation
        self.flow_memory = {}            # (src,dst) -> metadata
        self.transitions = deque(maxlen=5000)

        # RL model: input_dim = number of candidate paths (we use 2 spines) -> one util per candidate
        num_paths = 2
        self.model = ActorCritic(input_dim=num_paths, num_actions=num_paths, device=DEVICE)
        # spawn monitor & trainer threads
        self.monitor_thread = hub.spawn(self._monitor)
        self.trainer_thread = hub.spawn(self._trainer)
        self.logger.info("RLDCController initialized for YAML topology")
    
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

        self.logger.info(f"Installed table-miss on switch {dp.id}")


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
                del self.port_stats[dp.id]
                self.logger.info("Datapath %s disconnected", dp.id)

    # -------------------------
    # Polling for port & flow stats
    # -------------------------
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                self._request_port_stats(dp)
                self._request_flow_stats(dp)
            hub.sleep(POLL_INTERVAL)

    def _request_port_stats(self, datapath):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
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
                else:
                    tx_delta = stat.tx_bytes - prev['tx']
                    rx_delta = stat.rx_bytes - prev['rx']
                    util = (tx_delta + rx_delta) * 8.0 / (interval * 1e6)  # Mbps
                tx_dropped = stat.tx_dropped - prev.get('tx_dropped', 0)
                rx_dropped = stat.rx_dropped - prev.get('rx_dropped', 0)
            else:
                util = 0.0
                tx_dropped = 0
                rx_dropped = 0
            self.port_stats[dpid][p] = {
                'tx': stat.tx_bytes, 'rx': stat.rx_bytes, 'ts': now,
                'util': util, 'tx_dropped': stat.tx_dropped, 'rx_dropped': stat.rx_dropped
            }

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        self.flow_stats_cache[dpid] = ev.msg.body

    # -------------------------
    # Packet-in: decide path and install flows/groups
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        dpid = dp.id
        parser = dp.ofproto_parser
        ofp = dp.ofproto

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        eth = pkt.get_protocol(ethernet.ethernet)
        if eth.ethertype == 0x0806:  # ARP packet
            actions = [parser.OFPActionOutput(ofp.OFPP_FLOOD)]
            out = parser.OFPPacketOut(datapath=dp,
                                    buffer_id=msg.buffer_id,
                                    in_port=msg.match['in_port'],
                                    actions=actions,
                                    data=msg.data)
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
            actions = [parser.OFPActionOutput(ofp.OFPP_FLOOD)]
            data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
            out = parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=msg.match['in_port'], actions=actions, data=data)
            dp.send_msg(out)
            return

        # Candidate out_ports = leaf's spine-facing ports
        candidate_ports = LEAF_SPINE_PORTS.get(ingress_leaf, [])
        if not candidate_ports:
            # no alternate paths, fallback
            actions = [parser.OFPActionOutput(ofp.OFPP_FLOOD)]
            data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
            out = parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=msg.match['in_port'], actions=actions, data=data)
            dp.send_msg(out)
            return

        # Build state vector (one util per candidate port, normalized)
        state = []
        for p in candidate_ports:
            util = self.port_stats.get(ingress_leaf, {}).get(p, {}).get('util', 0.0)
            state.append(min(util / CAPACITY_Mbps, 1.0))
        state_np = np.array(state, dtype=np.float32)

        # Elephant detection using FlowStats cache on ingress_leaf datapath
        is_elephant = self._is_elephant(flow_key, ingress_leaf)

        # Query policy (actor-critic)
        probs, action_idx, logp = self.model.policy(state_np)

        # Mice -> single path (argmax)
        if not is_elephant:
            chosen_idx = int(np.argmax(probs))
            chosen_port = candidate_ports[chosen_idx]
            self._install_simple_flow(dp, src, dst, chosen_port)
            flow_type = 'mice'
            self.logger.info("MICE %s->%s chosen port %s", src, dst, chosen_port)
        else:
            # Elephant -> install SELECT group with weights proportional to probs
            weights = (probs * GROUP_WEIGHT_SCALE).astype(int)
            weights = np.maximum(weights, 1)
            gid = self._group_id_from_flow(flow_key)
            self._install_select_group(dp, gid, candidate_ports, weights)
            self._install_flow_to_group(dp, src, dst, gid)
            flow_type = 'elephant'
            self.logger.info("ELEPHANT %s->%s group %s ports %s weights %s", src, dst, gid, candidate_ports, weights.tolist())

        # store metadata for training/reward
        self.flow_memory[flow_key] = {
            'dpid': ingress_leaf,
            'state': state_np,
            'action_idx': action_idx,
            'logp': logp,
            'time': time.time(),
            'type': flow_type,
            'candidate_ports': candidate_ports
        }

        # send the packet through chosen path / group
        if flow_type == 'mice':
            actions = [parser.OFPActionOutput(chosen_port)]
        else:
            actions = [parser.OFPActionGroup(gid)]
        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=msg.match['in_port'], actions=actions, data=data)
        dp.send_msg(out)

    # -------------------------
    # Flow detection helper
    # -------------------------
    def _is_elephant(self, flow_key, dpid):
        """Check cached FlowStats for flow (src,dst) on dpid. If byte_count >= threshold -> elephant"""
        stats = self.flow_stats_cache.get(dpid, [])
        src, dst = flow_key
        for s in stats:
            # match might be in s.match; keys differ by Ryu/DPDK; guard access
            try:
                m = s.match
                if m.get('ipv4_src') == src and m.get('ipv4_dst') == dst:
                    byte_count = getattr(s, 'byte_count', 0) or 0
                    # store previous for throughput calc later
                    key = (flow_key, dpid)
                    prev = self.flow_prev_bytes.get(key, byte_count)
                    self.flow_prev_bytes[key] = byte_count
                    if byte_count >= ELEPHANT_BYTES:
                        return True
            except Exception:
                continue
        return False

    # -------------------------
    # Flow & group install helpers
    # -------------------------
    def _install_simple_flow(self, datapath, src, dst, out_port, priority=200):
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        actions = [parser.OFPActionOutput(out_port)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)

    def _install_select_group(self, datapath, group_id, out_ports, weights):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        buckets = []
        for p, w in zip(out_ports, weights):
            actions = [parser.OFPActionOutput(p)]
            buckets.append(parser.OFPBucket(weight=int(w), watch_port=p, watch_group=ofp.OFPG_ANY, actions=actions))
        grp_mod = parser.OFPGroupMod(datapath=datapath, command=ofp.OFPGC_MODIFY, type_=ofp.OFPGT_SELECT, group_id=group_id, buckets=buckets)
        try:
            datapath.send_msg(grp_mod)
        except Exception:
            grp_mod = parser.OFPGroupMod(datapath=datapath, command=ofp.OFPGC_ADD, type_=ofp.OFPGT_SELECT, group_id=group_id, buckets=buckets)
            datapath.send_msg(grp_mod)

    def _install_flow_to_group(self, datapath, src, dst, group_id, priority=200):
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, [parser.OFPActionGroup(group_id)])]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)

    def _group_id_from_flow(self, flow_key):
        return (abs(hash(flow_key)) % (2**31 - 1)) or 1

    # -------------------------
    # Reward computation & trainer
    # -------------------------
    def _compute_reward(self, flow_key, meta):
        """
        reward = alpha * throughput_gain
                 - beta * link_utilization_skew
                 - gamma * packet_loss
                 - delta * latency_penalty
        All terms normalized to [0,1] ranges.
        """
        dpid = meta['dpid']
        candidate_ports = meta['candidate_ports']

        # Throughput gain: compute flow throughput (bytes delta / interval -> Mbps / CAPACITY)
        # Look up latest flow stats and previous bytes
        flow_throughput_norm = 0.0
        # find byte_count in flow_stats_cache
        stats = self.flow_stats_cache.get(dpid, [])
        src, dst = flow_key = (list(meta.get('state', [])),) if False else (meta,)  # dummy to avoid lint
        # Real compute: find flow stat with matching src/dst
        # We'll try to find a matching stat entry
        src_ip, dst_ip = flow_key = (None, None)
        if 'state' in meta:  # earlier we stored state, but we need real src/dst; meta lacks them: reconstruct from flow_memory keys
            pass
        # Instead, iterate flow_stats_cache to find any entry with the same match fields
        # We'll find any s where ipv4_src & ipv4_dst match one of flow_memory keys (search backwards)
        # The controller stored flow_memory keyed by (src,dst). We'll compute reward only if we can identify the exact (src,dst)
        # Find original (src,dst) by matching meta against flow_memory entries
        target_key = None
        for k, v in list(self.flow_memory.items()):
            if v is meta:
                target_key = k
                break
        if target_key is None:
            # cannot compute
            return None
        src_ip, dst_ip = target_key
        # find flow stat entry
        byte_count = None
        for s in stats:
            m = s.match if hasattr(s, 'match') else {}
            if m.get('ipv4_src') == src_ip and m.get('ipv4_dst') == dst_ip:
                byte_count = getattr(s, 'byte_count', 0) or 0
                break
        key_prev = (target_key, dpid)
        prev_bytes = self.flow_prev_bytes.get(key_prev, None)
        if prev_bytes is None:
            # store and wait for next interval
            if byte_count is not None:
                self.flow_prev_bytes[key_prev] = byte_count
            return None
        if byte_count is None:
            return None
        # compute throughput (Mbps)
        interval = POLL_INTERVAL
        bytes_delta = max(0, byte_count - prev_bytes)
        throughput_mbps = (bytes_delta * 8.0) / (interval * 1e6)
        flow_throughput_norm = min(throughput_mbps / CAPACITY_Mbps, 1.0)
        # update prev
        self.flow_prev_bytes[key_prev] = byte_count

        # Link utilization skew: compute std dev or max-min across candidate ports (normalized)
        utils = []
        for p in candidate_ports:
            u = self.port_stats.get(dpid, {}).get(p, {}).get('util', 0.0)
            utils.append(min(u / CAPACITY_Mbps, 1.0))
        if not utils:
            return None
        util_skew = float(np.std(utils))  # already normalized to 0..1 scale approx

        # packet_loss: compute dropped packets delta across candidate ports normalized
        drop_sum = 0.0
        for p in candidate_ports:
            entry = self.port_stats.get(dpid, {}).get(p, {})
            drop_sum += float(entry.get('tx_dropped', 0) + entry.get('rx_dropped', 0))
        # normalize drop: if large, cap; here assume 1000 drops per interval corresponds to loss=1.0
        packet_loss_norm = min(drop_sum / 1000.0, 1.0)

        # latency_penalty: placeholder 0.0 (you can add active probing later)
        latency_penalty = 0.0

        # full reward
        reward = (ALPHA * flow_throughput_norm) - (BETA * util_skew) - (GAMMA * packet_loss_norm) - (DELTA * latency_penalty)
        # clip to [-1,1] for stability
        reward = max(-1.0, min(1.0, reward))
        return reward

    def _trainer(self):
        """Periodically evaluate stored flows and create transitions for training"""
        while True:
            keys = list(self.flow_memory.keys())
            for k in keys:
                meta = self.flow_memory.get(k)
                if not meta:
                    continue
                # wait at least one poll interval
                if time.time() - meta['time'] >= max(1.5, POLL_INTERVAL):
                    r = self._compute_reward(k, meta)
                    if r is not None:
                        # build transition: (s,a,r,s',done,logp)
                        s = meta['state']
                        a = meta['action_idx']
                        logp = meta['logp']
                        # next state: sample current utils
                        dpid = meta['dpid']
                        candidate_ports = meta['candidate_ports']
                        next_state = []
                        for p in candidate_ports:
                            util = self.port_stats.get(dpid, {}).get(p, {}).get('util', 0.0)
                            next_state.append(min(util / CAPACITY_Mbps, 1.0))
                        next_state_np = np.array(next_state, dtype=np.float32)
                        done = False
                        # append
                        self.transitions.append((s, a, r, next_state_np, done, logp))
                        # remove meta so we don't double-count
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

