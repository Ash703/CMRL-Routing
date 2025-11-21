import time
import numpy as np
import json
import os
import yaml
import random
from collections import deque
from threading import Lock

from utils import Network
from src.rl.rl_model_x import ActorCritic

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp

# ======================================================================
# CONFIGURATION
# ======================================================================
ROUTING_ALGO = "RL"  # "RL", "ECMP", "STATIC", "GREEDY", "RANDOM", "RR"

POLL_INTERVAL = 2.0
TRAFFIC_LOG_DIR = "./traffic_logs"
HANDSHAKE_FILE = "active_flow_config.json"
CHECKPOINT_DIR = "checkpoints4"
MODEL_SAVE_STEPS = 20
DEVICE = 'cpu'

# Thresholds & Params
ELEPHANT_BYTES = 2 * 1024 * 1024 
CAPACITY_Mbps = 1000.0
GROUP_WEIGHT_SCALE = 100
TRAIN_BATCH_SIZE = 8
FLOW_IDLE_TIMEOUT = 30
GROUP_IDLE_TIMEOUT = 300

# Reward Weights
ALPHA = 1.0   # Throughput
BETA  = 1.0   # Load Balance
GAMMA = 1.0   # Packet Loss
DELTA = 0.0

# ======================================================================
# TOPOLOGY SETUP
# ======================================================================
config_file = os.environ.get("NETWORK_CONFIG_FILE", "network_config.yaml")
if not os.path.exists(config_file):
    # Fallback for testing without mininet script wrapper
    config_file = "network_config.yaml"

net = Network(config_file)

with open(config_file) as f:
    raw_cfg = yaml.safe_load(f)

HOST_TO_LEAF = {}
HOST_PORT = {}
NAME_TO_IP = {}

for h_conf in raw_cfg["hosts"]:
    ip = h_conf["ip"]
    name = h_conf["name"]
    leaf_name = h_conf["connected_to"]
    port = h_conf["port"]
    NAME_TO_IP[name] = ip
    
    try:
        leaf_dpid = next(sw["id"] for sw in raw_cfg["switches"] if sw["name"] == leaf_name)
        HOST_TO_LEAF[ip] = leaf_dpid
        HOST_PORT[ip] = port
    except StopIteration:
        pass

SPINES = net.spines[:]
LEAF_SPINE_PORTS = {}
for leaf in net.leaves:
    ports = []
    for spine in net.spines:
        # Find link leaf->spine
        p = None
        if (leaf, spine) in net.links:
            p = net.links[(leaf, spine)]["port"]
        ports.append(p)
    LEAF_SPINE_PORTS[leaf] = ports

SPINE_TO_LEAF_PORTS = {}
for spine in net.spines:
    leaf_ports = {}
    for leaf in net.leaves:
        if (spine, leaf) in net.links:
            leaf_ports[leaf] = net.links[(spine, leaf)]["port"]
    SPINE_TO_LEAF_PORTS[spine] = leaf_ports

# ======================================================================
# CONTROLLER
# ======================================================================
class RLDCController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RLDCController, self).__init__(*args, **kwargs)
        
        # Core structures
        self.datapaths = {}
        self.port_stats = {}
        self.flow_stats_cache = {}
        self.flow_prev_bytes = {}
        self.flow_memory = {}
        self.transitions = deque(maxlen=5000)
        self.groups_last_used = {}
        self.promoted_flows = set()
        self.promoted_meta = {}
        
        # Dynamic monitoring
        self.primary_flow_key = None 
        self.iperf3_stats = {'throughput_mbps': 0.0, 'ts': 0.0}
        self.iperf3_log_path = None
        
        self.lock = Lock()
        self.rr_counter = 0
        
        # Initialize RL
        num_paths = len(SPINES)
        self.model = ActorCritic(input_dim=num_paths*2, num_actions=num_paths, device=DEVICE)
        self.rl_update_step = 0
        
        path = f"{CHECKPOINT_DIR}/rl_latest.pt"
        if os.path.exists(path):
            self.model.load_checkpoint(path, map_location=DEVICE)
        
        # Spawn threads
        self.monitor_thread = hub.spawn(self._monitor)
        self.trainer_thread = hub.spawn(self._trainer)
        self.cleaner_thread = hub.spawn(self._group_cleaner)
        self.flow_thread = hub.spawn(self._dynamic_flow_monitor)
        
        self.logger.info("RL Controller Running. Strategy: %s", ROUTING_ALGO)

    # ------------------------------------------------------------------
    # MONITORS
    # ------------------------------------------------------------------
    def _dynamic_flow_monitor(self):
        """Reads handshake file to find which flow is the 'Target'."""
        handshake_path = os.path.join(TRAFFIC_LOG_DIR, HANDSHAKE_FILE)
        while True:
            # 1. Discovery
            if self.primary_flow_key is None:
                if os.path.exists(handshake_path):
                    try:
                        with open(handshake_path, 'r') as f:
                            cfg = json.load(f)
                            src, dst = cfg.get("src_ip"), cfg.get("dst_ip")
                            log = cfg.get("log_path")
                            if src and dst and log:
                                with self.lock:
                                    self.primary_flow_key = (src, dst)
                                    self.iperf3_log_path = log
                                self.logger.info("Target Flow Discovered: %s -> %s", src, dst)
                    except: pass
            
            # 2. Reading Stats
            if self.iperf3_log_path and os.path.exists(self.iperf3_log_path):
                try:
                    with open(self.iperf3_log_path, 'r') as f:
                        data = json.load(f)
                        if 'intervals' in data and data['intervals']:
                            last = data['intervals'][-1]['sum']
                            mbps = last.get('bits_per_second', 0) / 1e6
                            with self.lock:
                                self.iperf3_stats = {'throughput_mbps': mbps, 'ts': time.time()}
                except: pass
            
            hub.sleep(POLL_INTERVAL)

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                parser = dp.ofproto_parser
                dp.send_msg(parser.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY))
                dp.send_msg(parser.OFPFlowStatsRequest(dp))
            hub.sleep(POLL_INTERVAL)

    # ------------------------------------------------------------------
    # DECISION LOGIC
    # ------------------------------------------------------------------
    def _get_routing_decision(self, candidate_ports, ingress_leaf):
        n = len(candidate_ports)
        if n == 0: return None, None, None
        
        # Build State
        utils = []
        state_list = []
        with self.lock:
            for p in candidate_ports:
                entry = self.port_stats.get(ingress_leaf, {}).get(p, {})
                u = entry.get('util', 0.0)
                d = float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0))
                utils.append(u)
                state_list.append(min(u/CAPACITY_Mbps, 1.0))
                state_list.append(min(d/1000.0, 1.0))
        
        state_np = np.array(state_list, dtype=np.float32)
        probs = np.zeros(n, dtype=np.float32)
        idx = 0
        
        if ROUTING_ALGO == "RL":
            probs, _ = self.model.policy(state_np)
            idx = int(np.argmax(probs))
        elif ROUTING_ALGO == "ECMP":
            probs[:] = 1.0/n
            idx = random.randint(0, n-1)
        elif ROUTING_ALGO == "GREEDY":
            idx = int(np.argmin(utils))
            probs[idx] = 1.0
        elif ROUTING_ALGO == "STATIC":
            idx = 0
            probs[idx] = 1.0
        elif ROUTING_ALGO == "RANDOM":
            idx = random.randint(0, n-1)
            probs[idx] = 1.0
        elif ROUTING_ALGO == "RR":
            with self.lock:
                idx = self.rr_counter % n
                self.rr_counter += 1
            probs[idx] = 1.0
            
        return probs, idx, state_np

    # ------------------------------------------------------------------
    # PACKET PROCESSING
    # ------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        if eth.ethertype == 0x0806:
            self._handle_arp(msg)
            return
        
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if not ip_pkt: return
        
        src, dst = ip_pkt.src, ip_pkt.dst
        ing_leaf = HOST_TO_LEAF.get(src)
        dst_leaf = HOST_TO_LEAF.get(dst)
        
        if not ing_leaf: return
        
        # Local switching
        if ing_leaf == dst_leaf:
            self._handle_same_leaf(dp, src, dst, msg)
            return
            
        # Spine routing
        candidates = LEAF_SPINE_PORTS.get(ing_leaf, [])
        probs, idx, state = self._get_routing_decision(candidates, ing_leaf)
        
        if probs is None: return
        
        # INSTALL BIDIRECTIONAL RULES (MICE)
        self._install_path_rules(src, dst, ing_leaf, dst_leaf, idx)
        
        # Store for RL
        if ROUTING_ALGO == "RL":
            meta = {
                'flow_key': (src, dst), 'dpid': ing_leaf, 'state': state,
                'action_probs': probs, 'chosen_idx': idx, 'time': time.time(),
                'type': 'mice', 'candidate_ports': candidates
            }
            with self.lock: self.flow_memory[(src, dst)] = meta
            
        # Packet Out
        port = candidates[idx]
        out = dp.ofproto_parser.OFPPacketOut(
            datapath=dp, buffer_id=dp.ofproto.OFP_NO_BUFFER,
            in_port=dp.ofproto.OFPP_CONTROLLER,
            actions=[dp.ofproto_parser.OFPActionOutput(port)],
            data=msg.data
        )
        dp.send_msg(out)

    # ------------------------------------------------------------------
    # HELPERS & INSTALLATION
    # ------------------------------------------------------------------
    def _install_path_rules(self, src, dst, ing_dpid, dst_dpid, path_idx):
        """
        Installs Forward AND Reverse path flows to ensure connectivity.
        Path: HostA -> IngLeaf -> Spine[idx] -> EgrLeaf -> HostB
        """
        spine_dpid = SPINES[path_idx % len(SPINES)]
        
        # --- FORWARD PATH (src -> dst) ---
        
        # 1. Ingress Leaf -> Spine
        p_ing_spine = LEAF_SPINE_PORTS[ing_dpid][path_idx % len(SPINES)]
        self._install_simple_flow(self.datapaths.get(ing_dpid), src, dst, p_ing_spine)
        
        # 2. Spine -> Egress Leaf
        p_spine_egr = SPINE_TO_LEAF_PORTS[spine_dpid].get(dst_dpid)
        self._install_simple_flow(self.datapaths.get(spine_dpid), src, dst, p_spine_egr)
        
        # 3. Egress Leaf -> Host
        p_egr_host = HOST_PORT.get(dst)
        self._install_simple_flow(self.datapaths.get(dst_dpid), src, dst, p_egr_host)
        
        # --- REVERSE PATH (dst -> src) ---
        # We use the SAME spine for symmetry, preventing out-of-order TCP packets
        
        # 1. Egress Leaf -> Spine (Uplink)
        # Find port on dst_dpid connected to spine_dpid
        # We assume symmetric topology: if Ing->Spine is index I, Dst->Spine is likely index I too
        # But safely, we just need *a* port to that specific spine.
        p_egr_spine = LEAF_SPINE_PORTS[dst_dpid][path_idx % len(SPINES)]
        self._install_simple_flow(self.datapaths.get(dst_dpid), dst, src, p_egr_spine)
        
        # 2. Spine -> Ingress Leaf
        p_spine_ing = SPINE_TO_LEAF_PORTS[spine_dpid].get(ing_dpid)
        self._install_simple_flow(self.datapaths.get(spine_dpid), dst, src, p_spine_ing)
        
        # 3. Ingress Leaf -> Host
        p_ing_host = HOST_PORT.get(src)
        self._install_simple_flow(self.datapaths.get(ing_dpid), dst, src, p_ing_host)

    def _install_simple_flow(self, dp, src, dst, out_port):
        if not dp or not out_port: return
        parser = dp.ofproto_parser
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        actions = [parser.OFPActionOutput(out_port)]
        inst = [parser.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=100, match=match, instructions=inst, idle_timeout=FLOW_IDLE_TIMEOUT)
        dp.send_msg(mod)

    def _handle_same_leaf(self, dp, src, dst, msg):
        port = HOST_PORT.get(dst)
        if port:
            self._install_simple_flow(dp, src, dst, port)
            # Reverse
            r_port = HOST_PORT.get(src)
            self._install_simple_flow(dp, dst, src, r_port)
            
            out = dp.ofproto_parser.OFPPacketOut(datapath=dp, buffer_id=dp.ofproto.OFP_NO_BUFFER,
                                                 in_port=dp.ofproto.OFPP_CONTROLLER,
                                                 actions=[dp.ofproto_parser.OFPActionOutput(port)],
                                                 data=msg.data)
            dp.send_msg(out)

    def _handle_arp(self, msg):
        dp = msg.datapath
        parser = dp.ofproto_parser
        out = parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id,
                                  in_port=msg.match['in_port'],
                                  actions=[parser.OFPActionOutput(dp.ofproto.OFPP_FLOOD)],
                                  data=msg.data)
        dp.send_msg(out)
        
    # ------------------------------------------------------------------
    # BOILERPLATE & STATS
    # ------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        with self.lock: self.flow_stats_cache[dpid] = ev.msg.body
        
        if dpid not in LEAF_SPINE_PORTS: return
        
        # Elephant Promotion Logic
        for stat in ev.msg.body:
            match = stat.match
            if 'ipv4_src' not in match: continue
            src, dst = match['ipv4_src'], match['ipv4_dst']
            
            ing = HOST_TO_LEAF.get(src)
            if ing != dpid: continue
            
            cnt = stat.byte_count
            key = ((src,dst), dpid)
            prev = self.flow_prev_bytes.get(key, 0)
            delta = max(0, cnt - prev)
            self.flow_prev_bytes[key] = cnt
            
            if delta > ELEPHANT_BYTES and key not in self.promoted_flows:
                cand = LEAF_SPINE_PORTS.get(dpid, [])
                probs, idx, state = self._get_routing_decision(cand, dpid)
                if probs is None: continue
                
                # Install Group
                weights = (probs * GROUP_WEIGHT_SCALE).astype(int)
                weights = np.maximum(weights, 1)
                gid = (abs(hash((src,dst))) % 60000) + 1
                
                self._install_select_group(self.datapaths[dpid], gid, cand, weights)
                self._install_flow_to_group(self.datapaths[dpid], src, dst, gid)
                
                with self.lock:
                    self.promoted_flows.add(key)
                    self.groups_last_used[(dpid, gid)] = time.time()
                    self.promoted_meta[key] = {'gid': gid, 'dpid': dpid, 'last_bytes': cnt}

                self.logger.info("Elephant Promoted: %s->%s", src, dst)
                
                if ROUTING_ALGO == "RL":
                    meta = {
                        'flow_key': (src, dst), 'dpid': dpid, 'state': state,
                        'action_probs': probs, 'chosen_idx': idx, 'time': time.time(),
                        'type': 'elephant', 'candidate_ports': cand
                    }
                    with self.lock: self.flow_memory[(src, dst)] = meta

    def _install_select_group(self, dp, gid, ports, weights):
        buckets = []
        for p, w in zip(ports, weights):
            buckets.append(dp.ofproto_parser.OFPBucket(weight=int(w), actions=[dp.ofproto_parser.OFPActionOutput(p)]))
        try:
            dp.send_msg(dp.ofproto_parser.OFPGroupMod(dp, dp.ofproto.OFPGC_ADD, dp.ofproto.OFPGT_SELECT, gid, buckets))
        except:
            dp.send_msg(dp.ofproto_parser.OFPGroupMod(dp, dp.ofproto.OFPGC_MODIFY, dp.ofproto.OFPGT_SELECT, gid, buckets))

    def _install_flow_to_group(self, dp, src, dst, gid):
        match = dp.ofproto_parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        actions = [dp.ofproto_parser.OFPActionGroup(gid)]
        inst = [dp.ofproto_parser.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = dp.ofproto_parser.OFPFlowMod(datapath=dp, priority=200, match=match, instructions=inst, idle_timeout=FLOW_IDLE_TIMEOUT)
        dp.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        dp.send_msg(dp.ofproto_parser.OFPFlowMod(datapath=dp, priority=0, match=dp.ofproto_parser.OFPMatch(), instructions=[dp.ofproto_parser.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, [dp.ofproto_parser.OFPActionOutput(dp.ofproto.OFPP_CONTROLLER, dp.ofproto.OFPCML_NO_BUFFER)])]))
    
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change(self, ev):
        if ev.state == MAIN_DISPATCHER: self.datapaths[ev.datapath.id] = ev.datapath

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        for stat in ev.msg.body:
            if stat.port_no > 1000: continue
            old = self.port_stats.get(dpid, {}).get(stat.port_no, {})
            now = time.time()
            delta_t = now - old.get('ts', 0)
            bw = (stat.tx_bytes - old.get('tx_bytes', 0)) * 8 / delta_t if delta_t > 0 else 0
            self.port_stats.setdefault(dpid, {})[stat.port_no] = {
                'util': bw, 'tx_bytes': stat.tx_bytes, 'ts': now,
                'tx_dropped_delta': stat.tx_dropped - old.get('tx_dropped', 0),
                'rx_dropped_delta': stat.rx_dropped - old.get('rx_dropped', 0),
                'tx_dropped': stat.tx_dropped, 'rx_dropped': stat.rx_dropped
            }

    def _compute_reward(self, flow_key, meta):
        is_primary = (self.primary_flow_key and flow_key == self.primary_flow_key)
        if is_primary:
            with self.lock: stats = self.iperf3_stats
            if stats['ts'] > meta['time']: tp = min(stats['throughput_mbps']/CAPACITY_Mbps, 1.0)
            else: return None
        else: tp = 0.1

        utils = []
        drops = 0.0
        with self.lock:
            for p in meta['candidate_ports']:
                e = self.port_stats.get(meta['dpid'], {}).get(p, {})
                utils.append(e.get('util', 0.0))
                drops += e.get('tx_dropped_delta', 0)
        
        if not utils: return None
        skew = float(np.std(utils))/CAPACITY_Mbps
        loss = min(drops/1000.0, 1.0)
        r = (ALPHA*tp) - (BETA*skew) - (GAMMA*loss)
        r = max(-1.0, min(1.0, r))
        
        if is_primary:
            with open(f"{CHECKPOINT_DIR}/rl_reward_log.csv", "a") as f: f.write(f"{time.time()},{r}\n")
        return r

    def _trainer(self):
        while True:
            if ROUTING_ALGO == "RL":
                batch = []
                with self.lock:
                    keys = list(self.flow_memory.keys())
                    for k in keys:
                        meta = self.flow_memory[k]
                        if time.time() - meta['time'] > 2.0:
                            r = self._compute_reward(k, meta)
                            if r is not None:
                                self.transitions.append((meta['state'], meta['action_probs'], r, meta['state'], False, None))
                                del self.flow_memory[k]
                
                if len(self.transitions) >= TRAIN_BATCH_SIZE:
                    b = [self.transitions.popleft() for _ in range(TRAIN_BATCH_SIZE)]
                    loss = self.model.update(b)
                    if loss: 
                        self.rl_update_step += 1
                        if self.rl_update_step % MODEL_SAVE_STEPS == 0:
                            self.model.save_checkpoint(f"{CHECKPOINT_DIR}/rl_latest.pt")
            hub.sleep(POLL_INTERVAL)

    def _group_cleaner(self):
        while True: hub.sleep(GROUP_IDLE_TIMEOUT)