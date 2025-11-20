import time
import numpy as np
import json
import os 
from collections import deque
from threading import Lock
from utils import Network 

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp

from src.rl.rl_model_x import ActorCritic

# ---------------------------
# Config & External Flow Definition
# ---------------------------
POLL_INTERVAL = 2.0 
ELEPHANT_BYTES = 2 * 1024 * 1024 
CAPACITY_Mbps = 1000.0 
GROUP_WEIGHT_SCALE = 100 
TRAIN_BATCH_SIZE = 8
DEVICE = 'cpu'
FLOW_IDLE_TIMEOUT = 30 
GROUP_IDLE_TIMEOUT = 300 

CHECKPOINT_DIR = "checkpoints2"
MODEL_SAVE_STEPS = 20
TRAFFIC_LOG_DIR = "./traffic_logs" 

# UPDATED: Primary flow based on your network_config.yaml
PRIMARY_FLOW_KEY = ("10.1.1.1", "10.1.1.4") 

# Reward coefficients
ALPHA = 1.0  # throughput
BETA = 1.0 # utilization skew
GAMMA = 1.0  # packet loss
DELTA = 0.0  # latency 

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

    leaf_dpid = next(
        sw["id"] for sw in raw_cfg["switches"] if sw["name"] == leaf_name
    )

    HOST_TO_LEAF[ip] = leaf_dpid
    HOST_PORT[ip] = port

SPINES = net.spines[:]

# Build LEAF_SPINE_PORTS
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

print("Host Ports:", HOST_PORT)
print("Host->Leaf:", HOST_TO_LEAF)
print("Spines:", SPINES)
print("Leaf->Spine Ports:", LEAF_SPINE_PORTS)

class RLDCController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RLDCController, self).__init__(*args, **kwargs)

        self.datapaths = {}
        self.port_stats = {}
        self.flow_stats_cache = {}
        self.flow_prev_bytes = {}
        self.flow_memory = {}
        self.transitions = deque(maxlen=5000)
        self.groups_last_used = {}
        self.promoted_flows = set()
        self.promoted_meta = {}
        self.iperf3_stats = {'throughput_mbps': 0.0, 'ts': 0.0}

        self.lock = Lock()

        num_paths = len(SPINES)
        self.model = ActorCritic(input_dim=num_paths*2, num_actions=num_paths, device=DEVICE)
        self.rl_update_step = 0
        SAVE_PATH = f"{CHECKPOINT_DIR}/rl_latest.pt"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        try:
            self.model.load_checkpoint(SAVE_PATH, map_location=DEVICE)
            self.logger.info("Loaded RL model from %s", SAVE_PATH)
        except Exception:
            self.logger.info("No saved RL model, starting fresh")

        self.monitor_thread = hub.spawn(self._monitor)
        self.trainer_thread = hub.spawn(self._trainer)
        self.cleaner_thread = hub.spawn(self._group_cleaner)
        self.iperf3_thread = hub.spawn(self._iperf3_monitor) 
        self.logger.info("RLDCController (final) initialized")

    def _iperf3_monitor(self):
        file_path = os.path.join(TRAFFIC_LOG_DIR, "h1_h4_parallel_flow.json")
        while True:
            try:
                if not os.path.exists(file_path):
                    hub.sleep(1.0)
                    continue
                
                with open(file_path, 'r') as f:
                    data = json.load(f)

                latest_throughput_mbps = 0.0
                latest_ts = time.time()
                
                if 'intervals' in data and data['intervals']:
                    last_interval_sum = data['intervals'][-1]['sum']
                    if 'bits_per_second' in last_interval_sum:
                        latest_throughput_mbps = last_interval_sum['bits_per_second'] / 1e6

                with self.lock:
                    self.iperf3_stats = {'throughput_mbps': latest_throughput_mbps, 'ts': latest_ts}

            except json.JSONDecodeError:
                pass
            except Exception as e:
                self.logger.exception("Error reading iperf3 log file: %s", e)
                
            hub.sleep(POLL_INTERVAL)

    # ... (Rest of the Controller code from previous response remains exactly the same) ...
    # [Truncated for brevity - paste the rest of the class methods here: 
    #  _get_host_port, switch_features_handler, _state_change, _monitor, 
    #  _flow_stats_reply, _group_stats_reply, _port_stats_reply, _packet_in, 
    #  _install_simple_flow, _install_select_group, _install_flow_to_group, 
    #  _compute_reward, _trainer, _group_cleaner, _remove_promotion]
    
    # --- Helper Functions (Same as previous) ---
    def _get_host_port(self, leaf, host_ip):
        return HOST_PORT.get(host_ip)

    def _group_id_from_flow(self, flow_key):
        return (abs(hash(flow_key)) % (63556)) or 1

    # --- Table-miss ---
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=inst)
        dp.send_msg(mod)
        self.logger.info("Installed table-miss on switch %s", dp.id)

        if dp.id in SPINES:
            for host_ip, leaf in HOST_TO_LEAF.items():
                match = parser.OFPMatch(eth_type=0x0800, ipv4_dst=host_ip)
                actions = [parser.OFPActionOutput(SPINE_TO_LEAF_PORTS[dp.id][leaf])]
                inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
                mod = parser.OFPFlowMod(datapath=dp, priority=200, match=match, instructions=inst, idle_timeout=0)
                dp.send_msg(mod)
                self.logger.info("Installed Spine rule: host %s to leaf %s",host_ip, leaf)

    # --- State Change ---
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

    # --- Monitor ---
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                try:
                    parser = dp.ofproto_parser
                    ofp = dp.ofproto
                    # Ports
                    req = parser.OFPPortStatsRequest(dp, 0, ofp.OFPP_ANY)
                    dp.send_msg(req)
                    # Flows
                    req = parser.OFPFlowStatsRequest(dp)
                    dp.send_msg(req)
                    # Groups
                    req = parser.OFPGroupStatsRequest(dp, 0, ofp.OFPG_ALL)
                    dp.send_msg(req)
                except Exception as e:
                    self.logger.exception("Error requesting stats: %s", e)
            hub.sleep(POLL_INTERVAL)

    # --- Flow Stats Reply ---
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        with self.lock:
            self.flow_stats_cache[dpid] = ev.msg.body

        try:
            if dpid in LEAF_SPINE_PORTS:
                stats_snapshot = list(ev.msg.body)
                for s in stats_snapshot:
                    try:
                        m = getattr(s, 'match', {}) or {}
                        src = m.get('ipv4_src')
                        dst = m.get('ipv4_dst')
                        if not src or not dst:
                            continue
                        byte_count = getattr(s, 'byte_count', 0) or 0
                        ingress_leaf = HOST_TO_LEAF.get(src)
                        if ingress_leaf is None:
                            continue

                        key_prev = ((src, dst), ingress_leaf)
                        prev_bytes = self.flow_prev_bytes.get(key_prev)
                        if prev_bytes is None:
                            self.flow_prev_bytes[key_prev] = byte_count
                            continue

                        delta = max(0, byte_count - prev_bytes)
                        self.flow_prev_bytes[key_prev] = byte_count

                        if (delta >= ELEPHANT_BYTES): 
                            if key_prev in self.promoted_flows:
                                continue

                            dp_ing = self.datapaths.get(ingress_leaf)
                            if dp_ing is None:
                                continue
                            candidate_ports = LEAF_SPINE_PORTS.get(ingress_leaf, [])
                            if not candidate_ports:
                                continue

                            # Policy Inference
                            probs = None
                            try:
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
                            weights = np.maximum(weights, 1)

                            gid = self._group_id_from_flow((src, dst))
                            try:
                                self._install_select_group(dp_ing, gid, candidate_ports, weights)
                                self._install_flow_to_group(dp_ing, src, dst, gid, idle_timeout=FLOW_IDLE_TIMEOUT)
                            except Exception:
                                continue

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
                            self.logger.info("PROMOTED flow %s->%s on leaf %s", src, dst, ingress_leaf)
                    except Exception:
                        continue
        except Exception:
            pass

    # --- Group Stats Reply ---
    @set_ev_cls(ofp_event.EventOFPGroupStatsReply, MAIN_DISPATCHER)
    def _group_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        stats_snapshot = list(ev.msg.body)
        group_by_gid = {}
        for st in stats_snapshot:
            try:
                gid = int(getattr(st, "group_id", None))
                byte_count = int(getattr(st, "byte_count", 0) or 0)
                group_by_gid[gid] = byte_count
            except Exception:
                continue
        
        with self.lock:
            promoted_keys = [k for k in self.promoted_meta.keys() if k[1] == dpid]

        for flow_id_key in promoted_keys:
            # (Cleanup logic same as before)
            try:
                meta = None
                with self.lock:
                    meta = self.promoted_meta.get(flow_id_key)
                if meta is None: continue

                gid = int(meta.get("gid"))
                if gid in group_by_gid:
                    current_bytes = group_by_gid[gid]
                    with self.lock:
                        self.promoted_meta[flow_id_key]['missing'] = 0
                else:
                    with self.lock:
                        self.promoted_meta[flow_id_key]['missing'] = self.promoted_meta[flow_id_key].get('missing', 0) + 1
                        if self.promoted_meta[flow_id_key]['missing'] >= 2:
                            self._remove_promotion(flow_id_key)
                    continue

                last_bytes = 0
                with self.lock:
                    last_bytes = int(self.promoted_meta[flow_id_key].get('last_bytes', 0))

                delta_bytes = max(0, int(current_bytes) - int(last_bytes))
                self.promoted_meta[flow_id_key]['last_bytes'] = int(current_bytes)

                if delta_bytes == 0:
                    with self.lock:
                        self.promoted_meta[flow_id_key]['inactive'] = self.promoted_meta[flow_id_key].get('inactive', 0) + 1
                else:
                    with self.lock:
                        self.promoted_meta[flow_id_key]['inactive'] = 0

                with self.lock:
                    if self.promoted_meta[flow_id_key].get('inactive', 0) >= 3:
                        self._remove_promotion(flow_id_key)

            except Exception:
                pass

    # --- Port Stats Reply ---
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        now = time.time()
        with self.lock:
            self.port_stats.setdefault(dpid, {})
            for stat in ev.msg.body:
                p = stat.port_no
                if p <= 0: continue
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
                        util = (tx_delta + rx_delta) * 8.0 / (interval * 1e6)  # Mbps
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

    # --- Packet In ---
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        msg = ev.msg
        dp_packetin = msg.datapath
        parser_pi = dp_packetin.ofproto_parser
        ofp_pi = dp_packetin.ofproto

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        # ARP
        if eth and eth.ethertype == 0x0806: 
            arp_pkt = pkt.get_protocol(arp.arp)
            if not arp_pkt: return

            target_ip = arp_pkt.dst_ip
            src_dp = dp_packetin
            src_dpid = src_dp.id
            data = msg.data if msg.buffer_id == ofp_pi.OFP_NO_BUFFER else None

            dst_leaf = HOST_TO_LEAF.get(target_ip)
            
            if dst_leaf and src_dpid == dst_leaf:
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

            if src_dpid in SPINES:
                target_port = SPINE_TO_LEAF_PORTS[src_dpid].get(dst_leaf)
                if target_port:
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
            
            target_ports = LEAF_SPINE_PORTS[src_dpid]
            actions = [src_dp.ofproto_parser.OFPActionOutput(tp) for tp in target_ports if tp is not None]
            out = src_dp.ofproto_parser.OFPPacketOut(
                datapath=src_dp,
                buffer_id=src_dp.ofproto.OFP_NO_BUFFER,
                in_port=src_dp.ofproto.OFPP_CONTROLLER,
                actions=actions,
                data=data
            )
            src_dp.send_msg(out)
            return

        # IPv4
        ip = pkt.get_protocol(ipv4.ipv4)
        if not ip: return

        src = ip.src
        dst = ip.dst
        ingress_leaf = HOST_TO_LEAF.get(src)
        dst_leaf = HOST_TO_LEAF.get(dst)

        if ingress_leaf is None:
            actions = [parser_pi.OFPActionOutput(ofp_pi.OFPP_FLOOD)]
            out = parser_pi.OFPPacketOut(datapath=dp_packetin, buffer_id=msg.buffer_id, in_port=msg.match.get('in_port'), actions=actions, data=msg.data)
            dp_packetin.send_msg(out)
            return

        candidate_ports = LEAF_SPINE_PORTS.get(ingress_leaf, [])
        if not candidate_ports: return

        # Same Leaf
        if dst_leaf is not None and dst_leaf == ingress_leaf:
            dp_ing = self.datapaths.get(ingress_leaf)
            if dp_ing:
                out_port = self._get_host_port(ingress_leaf, dst)
                in_port = self._get_host_port(ingress_leaf, src)
                self._install_simple_flow(dp_ing, src, dst, out_port, idle_timeout=FLOW_IDLE_TIMEOUT)
                self._install_simple_flow(dp_ing, dst, src, in_port, idle_timeout=FLOW_IDLE_TIMEOUT)
                # PacketOut
                out = dp_ing.ofproto_parser.OFPPacketOut(datapath=dp_ing, buffer_id=dp_ing.ofproto.OFP_NO_BUFFER, in_port=dp_ing.ofproto.OFPP_CONTROLLER, actions=[dp_ing.ofproto_parser.OFPActionOutput(out_port)], data=msg.data)
                dp_ing.send_msg(out)
            return

        # RL Decision
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

        probs, _ = self.model.policy(state_np)
        dp_ing = self.datapaths.get(ingress_leaf)
        if dp_ing is None: return

        chosen_idx = int(np.argmax(probs))
        chosen_port = candidate_ports[chosen_idx]

        self._install_simple_flow(dp_ing, src, dst, chosen_port, idle_timeout=FLOW_IDLE_TIMEOUT)
        actions_out = [dp_ing.ofproto_parser.OFPActionOutput(chosen_port)]

        # Install Path
        spine_dpid = SPINES[chosen_idx] if chosen_idx < len(SPINES) else SPINES[0]
        dp_spine = self.datapaths.get(spine_dpid)
        if dp_spine is not None:
            spine_out_port = SPINE_TO_LEAF_PORTS.get(spine_dpid, {}).get(dst_leaf)
            if spine_out_port:
                self._install_simple_flow(dp_spine, src, dst, spine_out_port, idle_timeout=FLOW_IDLE_TIMEOUT)

        dp_dst = self.datapaths.get(dst_leaf)
        dst_host_port = self._get_host_port(dst_leaf, dst)
        if dp_dst and dst_host_port:
            self._install_simple_flow(dp_dst, src, dst, dst_host_port, idle_timeout=FLOW_IDLE_TIMEOUT)

        # Reverse Path
        if dp_spine:
            spine_back_port = SPINE_TO_LEAF_PORTS.get(spine_dpid, {}).get(ingress_leaf)
            if spine_back_port:
                self._install_simple_flow(dp_spine, dst, src, spine_back_port, idle_timeout=FLOW_IDLE_TIMEOUT)
        if dp_dst and dp_spine:
            uplinks = LEAF_SPINE_PORTS.get(dst_leaf, [])
            idx_of_spine = SPINES.index(spine_dpid) if spine_dpid in SPINES else None
            dst_uplink_port = None
            if idx_of_spine is not None and idx_of_spine < len(uplinks):
                dst_uplink_port = uplinks[idx_of_spine]
            if dst_uplink_port:
                self._install_simple_flow(dp_dst, dst, src, dst_uplink_port, idle_timeout=FLOW_IDLE_TIMEOUT)
        src_host_port = self._get_host_port(ingress_leaf, src)
        if dp_ing and src_host_port:
            self._install_simple_flow(dp_ing, dst, src, src_host_port, idle_timeout=FLOW_IDLE_TIMEOUT)

        meta = {
            'flow_key': (src, dst),
            'dpid': ingress_leaf,
            'state': state_np,
            'action_probs': probs.copy() if isinstance(probs, np.ndarray) else np.array(probs, dtype=np.float32),
            'chosen_idx': chosen_idx,
            'time': time.time(),
            'type': 'mice',
            'candidate_ports': candidate_ports
        }
        with self.lock:
            self.flow_memory[(src, dst)] = meta

        out = dp_ing.ofproto_parser.OFPPacketOut(datapath=dp_ing, buffer_id=dp_ing.ofproto.OFP_NO_BUFFER, in_port=dp_ing.ofproto.OFPP_CONTROLLER, actions=actions_out, data=msg.data)
        dp_ing.send_msg(out)

    # --- Install Flow Helpers ---
    def _install_simple_flow(self, datapath, src, dst, out_port, priority=100, idle_timeout=0):
        if datapath is None or out_port is None: return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        actions = [parser.OFPActionOutput(out_port)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst, idle_timeout=idle_timeout)
        datapath.send_msg(mod)

    def _install_select_group(self, datapath, group_id, out_ports, weights):
        if datapath is None: return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        buckets = []
        for p, w in zip(out_ports, weights):
            actions = [parser.OFPActionOutput(p)]
            buckets.append(parser.OFPBucket(weight=int(w), watch_port=ofp.OFPP_ANY, watch_group=ofp.OFPG_ANY, actions=actions))
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
        if datapath is None: return
        parser = datapath.ofproto_parser
        ofp = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src, ipv4_dst=dst)
        actions = [parser.OFPActionGroup(group_id)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, command=ofp.OFPFC_ADD, priority=priority, match=match, instructions=inst, idle_timeout=idle_timeout, hard_timeout=0)
        datapath.send_msg(mod)

    # --- Compute Reward ---
    def _compute_reward(self, flow_key, meta):
        dpid = meta['dpid']
        candidate_ports = meta['candidate_ports']
        src_ip, dst_ip = meta.get('flow_key', (None, None))
        if src_ip is None: return None

        flow_throughput_norm = 0.0

        if flow_key == PRIMARY_FLOW_KEY:
            # Use external iperf3 data
            with self.lock:
                external_data = self.iperf3_stats
            
            if external_data['ts'] > meta['time'] and external_data['throughput_mbps'] > 0:
                throughput_mbps = external_data['throughput_mbps']
                flow_throughput_norm = min(throughput_mbps / CAPACITY_Mbps, 1.0)
            else:
                return None 
        else:
            # Use OpenFlow Stats
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
            
            if prev_bytes is None or byte_count is None:
                if byte_count is not None:
                    self.flow_prev_bytes[key_prev] = byte_count
                return None
            
            interval = POLL_INTERVAL
            bytes_delta = max(0, byte_count - prev_bytes)
            throughput_mbps = (bytes_delta * 8.0) / (interval * 1e6)
            flow_throughput_norm = min(throughput_mbps / CAPACITY_Mbps, 1.0)
            self.flow_prev_bytes[key_prev] = byte_count

        # Penalties
        with self.lock:
            utils = [min(self.port_stats.get(dpid, {}).get(p, {}).get('util', 0.0) / CAPACITY_Mbps, 1.0) for p in candidate_ports]
        if not utils: return None
        util_skew = float(np.std(utils))

        with self.lock:
            drop_sum = 0.0
            for p in candidate_ports:
                entry = self.port_stats.get(dpid, {}).get(p, {})
                drop_sum += float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0))
        packet_loss_norm = min(drop_sum / 1000.0, 1.0)

        reward = (ALPHA * flow_throughput_norm) - (BETA * util_skew) - (GAMMA * packet_loss_norm) - (DELTA * 0.0)
        reward = max(-1.0, min(1.0, reward))
        with open(f"{CHECKPOINT_DIR}/rl_reward_log.csv", "a", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow([time.time(), reward])
        return reward

    # --- Trainer ---
    def _trainer(self):
        while True:
            with self.lock:
                keys = list(self.flow_memory.keys())
            for k in keys:
                with self.lock:
                    meta = self.flow_memory.get(k)
                if not meta: continue
                now = time.time()
                if now - meta['time'] >= max(1.5, POLL_INTERVAL):
                    r = self._compute_reward(k, meta)
                    if r is not None:
                        s = meta['state']
                        a_probs = meta['action_probs']
                        if a_probs is None:
                            chosen_idx = int(meta.get('chosen_idx', 0))
                            a_probs = np.zeros(len(meta['candidate_ports']), dtype=np.float32)
                            a_probs[chosen_idx] = 1.0
                        else:
                            a_probs = np.asarray(a_probs, dtype=np.float32)
                        
                        # Next state
                        dpid = meta['dpid']
                        candidate_ports = meta['candidate_ports']
                        with self.lock:
                            next_state = []
                            for p in candidate_ports:
                                entry = self.port_stats.get(dpid, {}).get(p, {})
                                util = entry.get('util', 0.0)
                                drop = float(entry.get('tx_dropped_delta', 0) + entry.get('rx_dropped_delta', 0)) 
                                next_state.append(min(util / CAPACITY_Mbps, 1.0))
                                next_state.append(min(drop / 1000.0, 1.0))
                        next_state_np = np.array(next_state, dtype=np.float32)
                        
                        self.transitions.append((s, a_probs, r, next_state_np, False, None))

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
                        self.rl_update_step += 1
                        if self.rl_update_step % MODEL_SAVE_STEPS == 0:
                            self.model.save_checkpoint(f"{CHECKPOINT_DIR}/rl_latest.pt")
                    except Exception as e:
                        self.logger.error("Failed to save RL model: %s", e)
                    
            hub.sleep(POLL_INTERVAL)

    # --- Group Cleaner ---
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
        with self.lock:
            meta = self.promoted_meta.get(flow_id_key)
            if not meta:
                if flow_id_key in self.promoted_flows:
                    self.promoted_flows.discard(flow_id_key)
                return

            gid = int(meta.get('gid'))
            dpid = meta.get('dpid')
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

        dp = self.datapaths.get(dpid)
        if dp is None: return

        parser = dp.ofproto_parser
        ofp = dp.ofproto

        try:
            grp_del = parser.OFPGroupMod(datapath=dp, command=ofp.OFPGC_DELETE,
                                        type_=ofp.OFPGT_ALL, group_id=gid, buckets=[])
            dp.send_msg(grp_del)
        except Exception:
            pass

        try:
            src_dst = flow_id_key[0]
            src_ip, dst_ip = src_dst
            match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip, ipv4_dst=dst_ip)
            fm = parser.OFPFlowMod(datapath=dp, command=ofp.OFPFC_DELETE, out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY, match=match, table_id=0)
            dp.send_msg(fm)
        except Exception:
            pass