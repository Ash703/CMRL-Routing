import time
import numpy as np
import json
import os
import yaml
import random
from collections import deque
from threading import Lock

# Import your custom modules
from utils import Network
from src.rl.rl_model_x import ActorCritic

# Ryu Imports
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp

# ======================================================================
# CONFIGURATION SECTION
# ======================================================================

# --- ROUTING ALGORITHM SELECTION ---
# Options: "RL", "ECMP", "STATIC", "GREEDY", "RANDOM"
# Change this to run your different comparisons!
ROUTING_ALGO = "RL" 

# System Config
POLL_INTERVAL = 2.0              # Seconds between gathering stats
TRAFFIC_LOG_DIR = "./traffic_logs"
CHECKPOINT_DIR = "checkpoints2"
MODEL_SAVE_STEPS = 20
DEVICE = 'cpu'

# Network Thresholds
ELEPHANT_BYTES = 2 * 1024 * 1024 # 2 MB threshold for Elephant flow promotion
CAPACITY_Mbps = 1000.0           # Link capacity for normalization
GROUP_WEIGHT_SCALE = 100         # Scaling factor for Group Bucket weights
TRAIN_BATCH_SIZE = 8
FLOW_IDLE_TIMEOUT = 30
GROUP_IDLE_TIMEOUT = 300

# RL Reward Coefficients
ALPHA = 1.0   # Throughput
BETA  = 1.0   # Load Balance (Skew)
GAMMA = 1.0   # Packet Loss
DELTA = 0.0   # Latency (Not used currently)

# ======================================================================
# NETWORK TOPOLOGY LOADING
# ======================================================================
config_file = os.environ.get("NETWORK_CONFIG_FILE", "network_config.yaml")
net = Network(config_file)

with open(config_file) as f:
    raw_cfg = yaml.safe_load(f)

HOST_TO_LEAF = {}
HOST_PORT = {}
NAME_TO_IP = {}

# Parse Hosts from YAML
for host in raw_cfg["hosts"]:
    ip = host["ip"]
    name = host["name"]
    leaf_name = host["connected_to"]
    port = host["port"]
    
    NAME_TO_IP[name] = ip
    
    # Find leaf DPID by name
    try:
        leaf_dpid = next(sw["id"] for sw in raw_cfg["switches"] if sw["name"] == leaf_name)
        HOST_TO_LEAF[ip] = leaf_dpid
        HOST_PORT[ip] = port
    except StopIteration:
        print(f"Error: Leaf switch {leaf_name} not found in switches config.")

# Dynamic Primary Flow Detection (for RL Rewards)
h1_ip = NAME_TO_IP.get("h1")
h4_ip = NAME_TO_IP.get("h4")

if h1_ip and h4_ip:
    PRIMARY_FLOW_KEY = (h1_ip, h4_ip)
    print(f"*** RL Target Flow Detected: {h1_ip} -> {h4_ip}")
else:
    PRIMARY_FLOW_KEY = ("10.1.1.1", "10.1.1.4")
    print(f"*** WARNING: h1/h4 not found. Defaulting RL flow to {PRIMARY_FLOW_KEY}")

SPINES = net.spines[:]

# Build Leaf <-> Spine Port Mappings
LEAF_SPINE_PORTS = {}
for leaf in net.leaves:
    ports = []
    for spine in net.spines:
        port = None
        if (leaf, spine) in net.links:
            port = net.links[(leaf, spine)]["port"]
        ports.append(port)
    LEAF_SPINE_PORTS[leaf] = ports

SPINE_TO_LEAF_PORTS = {}
for spine in net.spines:
    leaf_ports = {}
    for leaf in net.leaves:
        if (spine, leaf) in net.links:
            leaf_ports[leaf] = net.links[(spine, leaf)]["port"]
    SPINE_TO_LEAF_PORTS[spine] = leaf_ports

print(f"Active Routing Algorithm: {ROUTING_ALGO}")

# ======================================================================
# CONTROLLER CLASS
# ======================================================================
class RLDCController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RLDCController, self).__init__(*args, **kwargs)

        # Data structures
        self.datapaths = {}
        self.port_stats = {}        # dpid -> port -> stats
        self.flow_stats_cache = {}
        self.flow_prev_bytes = {}
        self.flow_memory = {}       # For RL transitions
        self.transitions = deque(maxlen=5000)
        self.groups_last_used = {}
        self.promoted_flows = set()
        self.promoted_meta = {}
        self.iperf3_stats = {'throughput_mbps': 0.0, 'ts': 0.0}
        self.rr_counter = 0         # For Round Robin logic

        self.lock = Lock()

        # Initialize RL Model
        num_paths = len(SPINES)
        # Input: [Util, Loss] per path
        self.model = ActorCritic(input_dim=num_paths*2, num_actions=num_paths, device=DEVICE)
        self.rl_update_step = 0
        
        # Load Checkpoint if exists
        SAVE_PATH = f"{CHECKPOINT_DIR}/rl_latest.pt"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        try:
            self.model.load_checkpoint(SAVE_PATH, map_location=DEVICE)
            self.logger.info("Loaded RL model from %s", SAVE_PATH)
        except Exception:
            self.logger.info("No saved RL model, starting fresh")

        # Start Background Threads
        self.monitor_thread = hub.spawn(self._monitor)
        self.trainer_thread = hub.spawn(self._trainer)
        self.cleaner_thread = hub.spawn(self._group_cleaner)
        self.iperf3_thread = hub.spawn(self._iperf3_monitor)
        
        self.logger.info("RLDCController initialized. ALGO: %s", ROUTING_ALGO)

    # ------------------------------------------------------------------
    # 1. External Monitor (Reads Traffic Generator JSON Logs)
    # ------------------------------------------------------------------
    def _iperf3_monitor(self):
        file_path = os.path.join(TRAFFIC_LOG_DIR, "h1_h4_parallel_flow.json")
        while True:
            try:
                if not os.path.exists(file_path):
                    hub.sleep(1.0)
                    continue
                
                with open(file_path, 'r') as f:
                    # Attempt to read (handle incomplete writes by iperf)
                    try:
                        data = json.load(f)
                        if 'intervals' in data and data['intervals']:
                            last_sum = data['intervals'][-1]['sum']
                            mbps = last_sum.get('bits_per_second', 0) / 1e6
                            with self.lock:
                                self.iperf3_stats = {'throughput_mbps': mbps, 'ts': time.time()}
                    except json.JSONDecodeError:
                        pass # File might be being written to
            except Exception as e:
                self.logger.error("Iperf3 Monitor Error: %s", e)
            
            hub.sleep(POLL_INTERVAL)

    # ------------------------------------------------------------------
    # 2. Core Routing Logic (The Decision Maker)
    # ------------------------------------------------------------------
    def _get_routing_decision(self, candidate_ports, ingress_leaf):
        """
        Returns (probs, chosen_idx) based on global ROUTING_ALGO.
        This ensures both Packet-In and Flow-Stats-Reply use the same logic.
        """
        num_ports = len(candidate_ports)
        if num_ports == 0:
            return None, None

        # Build State for RL or Greedy
        utils = []
        state_list = []
        with self.lock:
            for p in candidate_ports:
                entry = self.port_stats.get(ingress_leaf, {}).get(p, {})
                u = entry.get('util', 0.0)
                d = float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0))
                
                utils.append(u) # For Greedy
                # For RL State: [Util_Norm, Loss_Norm]
                state_list.append(min(u / CAPACITY_Mbps, 1.0))
                state_list.append(min(d / 1000.0, 1.0))

        state_np = np.array(state_list, dtype=np.float32)
        probs = np.zeros(num_ports, dtype=np.float32)
        chosen_idx = 0

        # --- ALGORITHM SWITCH ---
        if ROUTING_ALGO == "RL":
            probs, _ = self.model.policy(state_np)
            chosen_idx = int(np.argmax(probs))

        elif ROUTING_ALGO == "ECMP":
            # Equal probability
            probs = np.ones(num_ports) / num_ports
            chosen_idx = random.randint(0, num_ports - 1)

        elif ROUTING_ALGO == "GREEDY":
            # Pick least utilized port
            chosen_idx = int(np.argmin(utils))
            probs[chosen_idx] = 1.0

        elif ROUTING_ALGO == "RANDOM":
            chosen_idx = random.randint(0, num_ports - 1)
            probs[chosen_idx] = 1.0

        elif ROUTING_ALGO == "STATIC":
            # Always pick first path (or hash-based if you prefer)
            chosen_idx = 0 
            probs[chosen_idx] = 1.0
            
        elif ROUTING_ALGO == "RR": # Round Robin
            with self.lock:
                chosen_idx = self.rr_counter % num_ports
                self.rr_counter += 1
            probs[chosen_idx] = 1.0

        return probs, chosen_idx, state_np

    # ------------------------------------------------------------------
    # 3. Packet In Handler (New Flows)
    # ------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        pkt = packet.Packet(msg.data)
        
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth.ethertype == 0x0806: 
            self._handle_arp(msg, pkt)
            return

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if not ip_pkt: return

        src, dst = ip_pkt.src, ip_pkt.dst
        ingress_leaf = HOST_TO_LEAF.get(src)
        dst_leaf = HOST_TO_LEAF.get(dst)

        if not ingress_leaf: return # Unknown host

        # Same Leaf Optimization
        if dst_leaf == ingress_leaf:
            self._handle_same_leaf(dp, src, dst, ingress_leaf, msg)
            return

        # Routing Decision
        candidate_ports = LEAF_SPINE_PORTS.get(ingress_leaf, [])
        probs, chosen_idx, state_np = self._get_routing_decision(candidate_ports, ingress_leaf)
        
        if probs is None: return

        chosen_port = candidate_ports[chosen_idx]

        # Install "Mice" flow (Simple forwarding)
        self._install_path_rules(src, dst, ingress_leaf, dst_leaf, chosen_idx)
        
        # Store Memory for RL Training (if using RL)
        if ROUTING_ALGO == "RL":
            meta = {
                'flow_key': (src, dst),
                'dpid': ingress_leaf,
                'state': state_np,
                'action_probs': probs,
                'chosen_idx': chosen_idx,
                'time': time.time(),
                'type': 'mice',
                'candidate_ports': candidate_ports
            }
            with self.lock:
                self.flow_memory[(src, dst)] = meta

        # Forward current packet
        out = parser.OFPPacketOut(datapath=dp, buffer_id=ofp.OFP_NO_BUFFER, 
                                  in_port=ofp.OFPP_CONTROLLER, 
                                  actions=[parser.OFPActionOutput(chosen_port)], 
                                  data=msg.data)
        dp.send_msg(out)

    # ------------------------------------------------------------------
    # 4. Flow Stats & Elephant Promotion
    # ------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        with self.lock:
            self.flow_stats_cache[dpid] = ev.msg.body

        # Only process Leaves for promotion
        if dpid not in LEAF_SPINE_PORTS: return

        for stat in ev.msg.body:
            # Parse Flow
            match = stat.match
            if 'ipv4_src' not in match or 'ipv4_dst' not in match: continue
            src, dst = match['ipv4_src'], match['ipv4_dst']
            
            byte_count = stat.byte_count
            ingress_leaf = HOST_TO_LEAF.get(src)
            
            if ingress_leaf != dpid: continue # Only promote at ingress

            # Elephant Detection Logic
            key = ((src, dst), dpid)
            prev_bytes = self.flow_prev_bytes.get(key, 0)
            delta = max(0, byte_count - prev_bytes)
            self.flow_prev_bytes[key] = byte_count

            # Promotion Check
            if delta >= ELEPHANT_BYTES and key not in self.promoted_flows:
                # Calculate Route again (Re-evaluate for Elephant)
                candidate_ports = LEAF_SPINE_PORTS.get(dpid, [])
                probs, chosen_idx, state_np = self._get_routing_decision(candidate_ports, dpid)
                
                if probs is None: continue

                # Scale probs to integer weights
                weights = (probs * GROUP_WEIGHT_SCALE).astype(int)
                weights = np.maximum(weights, 1) # Avoid 0 weights
                
                gid = (abs(hash((src, dst))) % 60000) + 1
                
                # Install Group Rule
                dp = self.datapaths[dpid]
                self._install_select_group(dp, gid, candidate_ports, weights)
                self._install_flow_to_group(dp, src, dst, gid)
                
                # Log & Track
                self.logger.info("PROMOTED %s->%s on %s (Algo: %s)", src, dst, dpid, ROUTING_ALGO)
                with self.lock:
                    self.promoted_flows.add(key)
                    self.promoted_meta[key] = {'gid': gid, 'dpid': dpid, 'last_bytes': byte_count, 'missing': 0, 'inactive': 0}
                    self.groups_last_used[(dpid, gid)] = time.time()
                    
                # Also store for RL training if needed
                if ROUTING_ALGO == "RL":
                    meta = {
                        'flow_key': (src, dst),
                        'dpid': dpid,
                        'state': state_np,
                        'action_probs': probs,
                        'chosen_idx': chosen_idx,
                        'time': time.time(),
                        'type': 'elephant',
                        'candidate_ports': candidate_ports
                    }
                    with self.lock:
                        self.flow_memory[(src, dst)] = meta

    # ------------------------------------------------------------------
    # 5. Training Logic
    # ------------------------------------------------------------------
    def _compute_reward(self, flow_key, meta):
        # Hybrid Reward: External Iperf (Alpha) + Internal Switch Stats (Beta/Gamma)
        dpid = meta['dpid']
        candidate_ports = meta['candidate_ports']
        
        # 1. Throughput (Alpha)
        throughput_norm = 0.0
        if flow_key == PRIMARY_FLOW_KEY:
            # Use External
            with self.lock:
                stats = self.iperf3_stats
            if stats['ts'] > meta['time']:
                throughput_norm = min(stats['throughput_mbps'] / CAPACITY_Mbps, 1.0)
            else:
                return None # Stale data
        else:
            # Use Internal Delta
            # (Simplified for brevity: assuming internal calculation done elsewhere or skipped for non-primary)
            pass

        # 2. Penalties (Beta/Gamma)
        utils = []
        drops = 0.0
        with self.lock:
            for p in candidate_ports:
                entry = self.port_stats.get(dpid, {}).get(p, {})
                utils.append(entry.get('util', 0.0))
                drops += entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0)
        
        if not utils: return None
        
        util_skew = float(np.std(utils)) / CAPACITY_Mbps # Normalized skew approx
        loss_norm = min(drops / 1000.0, 1.0)
        
        reward = (ALPHA * throughput_norm) - (BETA * util_skew) - (GAMMA * loss_norm)
        reward = max(-1.0, min(1.0, reward))
        
        # Log
        with open(f"{CHECKPOINT_DIR}/rl_reward_log.csv", "a") as f:
            f.write(f"{time.time()},{reward}\n")
            
        return reward

    def _trainer(self):
        while True:
            if ROUTING_ALGO != "RL":
                hub.sleep(5)
                continue

            keys = []
            with self.lock:
                keys = list(self.flow_memory.keys())
            
            for k in keys:
                with self.lock:
                    meta = self.flow_memory.get(k)
                if not meta: continue
                
                # Check if enough time passed to measure result
                if time.time() - meta['time'] > 2.0:
                    r = self._compute_reward(k, meta)
                    if r is not None:
                        # Create Transition
                        # (state, action_probs, reward, next_state, done)
                        # For simplicity, next_state is just current state re-read
                        # In deep RL, we'd re-read stats here.
                        self.transitions.append((meta['state'], meta['action_probs'], r, meta['state'], False, None))
                        
                        with self.lock:
                            if k in self.flow_memory: del self.flow_memory[k]
            
            # Batch Update
            if len(self.transitions) >= TRAIN_BATCH_SIZE:
                batch = [self.transitions.popleft() for _ in range(TRAIN_BATCH_SIZE)]
                loss = self.model.update(batch)
                if loss:
                    self.logger.info("RL Update: Loss Actor %.4f, Critic %.4f", loss[0], loss[1])
                    if self.rl_update_step % MODEL_SAVE_STEPS == 0:
                        self.model.save_checkpoint(f"{CHECKPOINT_DIR}/rl_latest.pt")
                    self.rl_update_step += 1
            
            hub.sleep(POLL_INTERVAL)

    # ------------------------------------------------------------------
    # 6. Helpers (Installation, ARP, Setup)
    # ------------------------------------------------------------------
    def _install_path_rules(self, src, dst, ing_leaf, dst_leaf, path_idx):
        # Helper to install End-to-End rules for Mice flows
        spine_dpid = SPINES[path_idx] if path_idx < len(SPINES) else SPINES[0]
        
        # 1. Ingress Leaf -> Spine
        p_ing_spine = LEAF_SPINE_PORTS[ing_leaf][path_idx]
        self._install_simple_flow(self.datapaths[ing_leaf], src, dst, p_ing_spine)
        
        # 2. Spine -> Egress Leaf
        if spine_dpid in self.datapaths:
             p_spine_leaf = SPINE_TO_LEAF_PORTS[spine_dpid].get(dst_leaf)
             if p_spine_leaf:
                 self._install_simple_flow(self.datapaths[spine_dpid], src, dst, p_spine_leaf)
        
        # 3. Egress Leaf -> Host
        if dst_leaf in self.datapaths:
            p_leaf_host = HOST_PORT.get(dst)
            if p_leaf_host:
                self._install_simple_flow(self.datapaths[dst_leaf], src, dst, p_leaf_host)
                
        # (Reverse path omitted for brevity, but symmetric logic applies)

    def _handle_same_leaf(self, dp, src, dst, leaf, msg):
        out_port = HOST_PORT.get(dst)
        if out_port:
            self._install_simple_flow(dp, src, dst, out_port)
            out = dp.ofproto_parser.OFPPacketOut(datapath=dp, buffer_id=dp.ofproto.OFP_NO_BUFFER,
                                                 in_port=dp.ofproto.OFPP_CONTROLLER,
                                                 actions=[dp.ofproto_parser.OFPActionOutput(out_port)],
                                                 data=msg.data)
            dp.send_msg(out)

    def _handle_arp(self, msg, pkt):
        # Simple ARP Flood (or directed if location known)
        # For robustness in this project, flooding everything except the ingress port is safest
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        actions = [parser.OFPActionOutput(ofp.OFPP_FLOOD)]
        out = parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=msg.match['in_port'], actions=actions, data=msg.data)
        dp.send_msg(out)

    def _install_simple_flow(self, dp, src, dst, out_port, priority=100):
        if not dp: return
        parser = dp.ofproto_parser
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        actions = [parser.OFPActionOutput(out_port)]
        inst = [parser.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=priority, match=match, instructions=inst, idle_timeout=FLOW_IDLE_TIMEOUT)
        dp.send_msg(mod)

    def _install_select_group(self, dp, gid, ports, weights):
        parser = dp.ofproto_parser
        buckets = []
        for p, w in zip(ports, weights):
            actions = [parser.OFPActionOutput(p)]
            buckets.append(parser.OFPBucket(weight=int(w), actions=actions))
        
        # Modify if exists, Add if not
        try:
            req = parser.OFPGroupMod(dp, dp.ofproto.OFPGC_ADD, dp.ofproto.OFPGT_SELECT, gid, buckets)
            dp.send_msg(req)
        except:
            req = parser.OFPGroupMod(dp, dp.ofproto.OFPGC_MODIFY, dp.ofproto.OFPGT_SELECT, gid, buckets)
            dp.send_msg(req)

    def _install_flow_to_group(self, dp, src, dst, gid):
        parser = dp.ofproto_parser
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        actions = [parser.OFPActionGroup(gid)]
        inst = [parser.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=200, match=match, instructions=inst, idle_timeout=FLOW_IDLE_TIMEOUT)
        dp.send_msg(mod)

    # --- Standard Setup ---
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=inst)
        dp.send_msg(mod)
        self.logger.info("Switch %s connected.", dp.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == "DEAD":
            if dp.id in self.datapaths: del self.datapaths[dp.id]

    # Polling Loop
    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                parser = dp.ofproto_parser
                dp.send_msg(parser.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY))
                dp.send_msg(parser.OFPFlowStatsRequest(dp))
            hub.sleep(POLL_INTERVAL)

    # Port Stats Handler (Updates self.port_stats)
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        for stat in ev.msg.body:
            if stat.port_no > 1000: continue # Skip local/special ports
            
            # Simple delta calculation
            old = self.port_stats.get(dpid, {}).get(stat.port_no, {})
            old_tx = old.get('tx_bytes', 0)
            old_ts = old.get('ts', 0)
            now = time.time()
            
            if old_ts:
                delta_t = now - old_ts
                if delta_t > 0:
                    bw = (stat.tx_bytes - old_tx) * 8 / delta_t
                else:
                    bw = 0
            else:
                bw = 0
            
            # Store
            self.port_stats.setdefault(dpid, {})[stat.port_no] = {
                'util': bw,
                'tx_bytes': stat.tx_bytes,
                'ts': now,
                'tx_dropped_delta': 0, # Simplified
                'rx_dropped_delta': 0
            }

    def _group_cleaner(self):
        while True:
            # Cleanup logic (omitted for brevity, strict cleanup not essential for simple comparison)
            hub.sleep(GROUP_IDLE_TIMEOUT)