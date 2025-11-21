import time
import random
import threading
import os
import json
import csv
import numpy as np
import yaml
from mininet.log import info, error

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EXPERIMENT_DURATION = 120  # Seconds (2 minutes)
POLL_INTERVAL = 1.0        # How often to read stats
RANDOM_SEED = 42           # For reproducibility
OUTPUT_DIR = "results_metrics"
HANDSHAKE_FILE = "active_flow_config.json"
TRAFFIC_LOG_DIR = "./traffic_logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAFFIC_LOG_DIR, exist_ok=True)

# Global Flags
STOP_FLAG = threading.Event()
FLOW_METRICS = {}  
LINK_STATS = {}    

# ==============================================================================
# HELPER: Flow Parameters (Your Specific Logic)
# ==============================================================================
def get_random_flow_params(flow_type, remaining_time):
    params = {}
    
    # 1. Define constraints
    if flow_type == 'elephant':
        raw_t = random.randint(20, 140)
        bw_mbps = random.randint(5, 5000) 
        params['b'] = f"{bw_mbps}M"
        params['P'] = random.randint(4, 10)
    elif flow_type == 'video':
        raw_t = random.randint(10, 30)
        bw_mbps = random.randint(10, 100)
        params['b'] = f"{bw_mbps}M"
        params['P'] = random.randint(2, 4)
    elif flow_type == 'interactive':
        raw_t = random.randint(5, 10)
        params['b'] = "100K"
        params['P'] = 1
    else: # mice
        raw_t = random.uniform(0.1, 1.0)
        params['b'] = "200K"
        params['P'] = 1

    # 2. Safety Clamp (Don't run past experiment end)
    allowed_t = remaining_time - 2.0
    if allowed_t < 0.2: return None
        
    params['t'] = min(raw_t, allowed_t)
    
    # Format for iperf
    if isinstance(params['t'], float):
        params['t_str'] = f"{params['t']:.2f}"
    else:
        params['t_str'] = str(int(params['t']))
        
    return params

# ==============================================================================
# METRICS
# ==============================================================================
def calculate_jains_fairness(values):
    if not values: return 0.0
    active = [v for v in values if v > 0]
    if not active: return 0.0
    n = len(active)
    sum_x = sum(active)
    sum_sq = sum(v**2 for v in active)
    if sum_sq == 0: return 0.0
    return (sum_x ** 2) / (n * sum_sq)

def calculate_link_skew(link_bytes_map):
    if not link_bytes_map: return 0.0
    values = list(link_bytes_map.values())
    return np.std(values)

# ==============================================================================
# TRAFFIC GENERATION THREAD
# ==============================================================================
def run_iperf_flow(src, dst_ip, duration_str, bandwidth, streams, flow_type, flow_id):
    FLOW_METRICS[flow_id] = {
        'start': time.time(), 'status': 'running', 'bytes': 0, 'type': flow_type
    }
    
    # JSON output for parsing
    cmd = f"iperf3 -c {dst_ip} -t {duration_str} -b {bandwidth} -P {streams} -J"
    
    try:
        result_json = src.cmd(cmd)
        FLOW_METRICS[flow_id]['end'] = time.time()
        
        try:
            data = json.loads(result_json)
            if 'error' in data:
                FLOW_METRICS[flow_id]['status'] = 'error'
                return

            sent_bytes = data['end']['sum_sent']['bytes']
            sender_thru = data['end']['sum_sent']['bits_per_second'] / 1e6
            
            FLOW_METRICS[flow_id]['status'] = 'completed'
            FLOW_METRICS[flow_id]['bytes'] = sent_bytes
            FLOW_METRICS[flow_id]['throughput'] = sender_thru
            
        except json.JSONDecodeError:
            FLOW_METRICS[flow_id]['status'] = 'failed_parse'
            
    except Exception:
        FLOW_METRICS[flow_id]['status'] = 'exec_error'

def traffic_generator_loop(net, seed, start_time):
    random.seed(seed)
    hosts = net.hosts
    counter = 0
    
    types = ['mice', 'interactive', 'video', 'elephant']
    weights = [0.4, 0.3, 0.2, 0.1] 
    
    info(f"*** Starting Dynamic Traffic Loop...\n")
    
    while not STOP_FLAG.is_set():
        elapsed = time.time() - start_time
        remaining = EXPERIMENT_DURATION - elapsed
        
        if remaining < 5: break
            
        # Random Pair
        src = random.choice(hosts)
        dst = random.choice(hosts)
        if src == dst: continue
        
        # Random Type
        f_type = random.choices(types, weights=weights, k=1)[0]
        params = get_random_flow_params(f_type, remaining)
        
        if not params: continue
            
        fid = f"{f_type}_{counter}"
        t = threading.Thread(
            target=run_iperf_flow, 
            args=(src, dst.IP(), params['t_str'], params['b'], params['P'], f_type, fid)
        )
        t.daemon = True
        t.start()
        
        counter += 1
        
        # Rate limiting to avoid crashing Mininet
        if f_type == 'elephant':
            time.sleep(random.uniform(2.0, 4.0))
        else:
            time.sleep(random.uniform(0.2, 0.8))

# ==============================================================================
# HANDSHAKE & SETUP
# ==============================================================================
def setup_rl_handshake(net):
    """Starts Primary Flow and writes config for Controller."""
    config_path = os.environ.get("NETWORK_CONFIG_FILE", "network_config.yaml")
    if not os.path.exists(config_path): 
        info("Config file not found for handshake, skipping.\n")
        return

    try:
        with open(config_path, 'r') as f: raw = yaml.safe_load(f)
        leaf_map = {}
        for h in raw['hosts']:
            l = h['connected_to']
            if l not in leaf_map: leaf_map[l] = []
            leaf_map[l].append(h['name'])
        
        leaves = list(leaf_map.keys())
        if len(leaves) < 2: return
        
        # Pick hosts on different leaves
        src_h = net.get(leaf_map[leaves[0]][0])
        dst_h = net.get(leaf_map[leaves[1]][0])
        
        # Start Primary Flow (Long Duration)
        log_path = os.path.abspath(os.path.join(TRAFFIC_LOG_DIR, "rl_target_flow.json"))
        dur = EXPERIMENT_DURATION + 10
        # Note: -P 4 for parallel
        cmd = f"iperf3 -c {dst_h.IP()} -t {dur} -b 50M -P 4 -J > {log_path} &"
        src_h.cmd(cmd)
        
        # Write Config
        cfg = {"src_ip": src_h.IP(), "dst_ip": dst_h.IP(), "log_path": log_path}
        with open(os.path.join(TRAFFIC_LOG_DIR, HANDSHAKE_FILE), 'w') as f:
            json.dump(cfg, f)
            
        info(f"*** RL Target: {src_h.name} -> {dst_h.name}\n")
        
    except Exception as e:
        error(f"Handshake failed: {e}\n")

# ==============================================================================
# STATS MONITOR (THE FIX FOR 0 THROUGHPUT)
# ==============================================================================
def monitor_links(net):
    # We assume spines start with 's1' (like s11, s12, s13)
    spines = [n for n in net.switches if n.name.startswith('s1')]
    csv_path = os.path.join(OUTPUT_DIR, "time_series_stats.csv")
    
    with open(csv_path, 'w') as f:
        csv.writer(f).writerow(["Time", "NetworkThroughput_Mbps", "LinkSkew", "JainFairness"])
    
    start_t = time.time()
    prev_bytes = 0
    prev_time = start_t
    
    while not STOP_FLAG.is_set():
        current_bytes = 0
        usage_list = []
        
        # 1. Read Stats from Linux Kernel (Fast)
        for sw in spines:
            for intf in sw.intfList():
                if intf.name == "lo": continue
                try:
                    path = f"/sys/class/net/{intf.name}/statistics/tx_bytes"
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            b = int(f.read().strip())
                            current_bytes += b
                            usage_list.append(b)
                except: pass
        
        # 2. Calculate Throughput (Delta)
        now = time.time()
        delta_t = now - prev_time
        throughput = 0.0
        
        if delta_t > 0 and prev_bytes > 0:
            delta_bytes = current_bytes - prev_bytes
            throughput = (delta_bytes * 8) / (delta_t * 1e6) # Mbps
            
        prev_bytes = current_bytes
        prev_time = now
        
        # 3. Calculate Load Balancing Metrics
        skew = calculate_link_skew({i: v for i, v in enumerate(usage_list)})
        fairness = calculate_jains_fairness(usage_list)
        
        # 4. Log
        with open(csv_path, 'a') as f:
            csv.writer(f).writerow([
                round(now - start_t, 2), 
                round(throughput, 2), 
                round(skew, 4), 
                round(fairness, 4)
            ])
            
        time.sleep(POLL_INTERVAL)

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def run_experiment(net, seed=RANDOM_SEED):
    os.system(f"rm -f {TRAFFIC_LOG_DIR}/*.json")
    
    # 1. Start Servers on ALL hosts (Crucial for avoiding Connection Refused)
    info("*** Starting iperf3 servers...\n")
    for h in net.hosts:
        h.cmd("iperf3 -s -p 5001 &")
    
    # 2. Start Primary Flow (RL Target)
    setup_rl_handshake(net)
    
    # 3. Start Monitoring
    monitor_t = threading.Thread(target=monitor_links, args=(net,))
    monitor_t.daemon = True
    monitor_t.start()
    
    # 4. Start Random Traffic
    start_t = time.time()
    gen_t = threading.Thread(target=traffic_generator_loop, args=(net, seed, start_t))
    gen_t.daemon = True
    gen_t.start()
    
    # 5. Wait Loop
    info(f"*** Running Experiment for {EXPERIMENT_DURATION}s...\n")
    for i in range(EXPERIMENT_DURATION):
        time.sleep(1)
        if i % 10 == 0 and i > 0:
            info(f"... {i}s\n")
            
    # 6. Cleanup
    STOP_FLAG.set()
    info("*** Stopping...\n")
    time.sleep(5) # Grace period
    for h in net.hosts: h.cmd("killall -9 iperf3")
        
    compute_final_report()

def compute_final_report():
    info("\n*** Final Summary ***\n")
    type_stats = {}
    
    # Group by Flow Type
    for fid, d in FLOW_METRICS.items():
        if d.get('status') == 'completed':
            t = d['type']
            if t not in type_stats: type_stats[t] = {'fct': [], 'thru': []}
            
            dur = d['end'] - d['start']
            type_stats[t]['fct'].append(dur)
            type_stats[t]['thru'].append(d['throughput'])
            
    csv_file = "final_comparison_results.csv"
    file_exists = os.path.isfile(csv_file)
    algo = os.environ.get("ROUTING_ALGO", "Unknown")
    
    rows = []
    for t, s in type_stats.items():
        rows.append({
            "Algorithm": algo,
            "FlowType": t,
            "Count": len(s['fct']),
            "Avg_FCT": sum(s['fct'])/len(s['fct']),
            "Avg_Throughput": sum(s['thru'])/len(s['thru'])
        })
        
    print(json.dumps(rows, indent=2))
    
    with open(csv_file, 'a') as f:
        headers = ["Algorithm", "FlowType", "Count", "Avg_FCT", "Avg_Throughput"]
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists: writer.writeheader()
        for r in rows: writer.writerow(r)
        
    info(f"*** Saved to {csv_file} ***\n")

def start_analysis(net):
    run_experiment(net)