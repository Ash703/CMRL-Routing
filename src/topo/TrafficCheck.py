import time
import random
import threading
import os
import json
import csv
import numpy as np
from mininet.log import info, error

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EXPERIMENT_DURATION = 120  # Seconds (2 minutes)
POLL_INTERVAL = 1.0        # How often to read stats
RANDOM_SEED = 42           # For reproducibility
OUTPUT_DIR = "/home/Ash/Documents/ACN/CMRL-Routing/results_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global Flags
STOP_FLAG = threading.Event()
FLOW_METRICS = {}  # Stores start/end times for FCT
LINK_STATS = {}    # Stores bytes for Skew/Fairness

# ==============================================================================
# FLOW GENERATION LOGIC
# ==============================================================================

def get_random_flow_params(flow_type):
    """
    Returns randomized parameters (duration, bandwidth, streams) based on type constraints.
    """
    params = {}
    
    if flow_type == 'elephant':
        # Duration: 20s to 140s
        params['t'] = random.randint(20, 140)
        # Bandwidth: 5Mbps to 5Gbps (Randomly pick a value in Mbps)
        # Note: 5Gbps is huge for Mininet/OVS, usually capped by link speed (1G), 
        # but we set the param as requested.
        bw_mbps = random.randint(5, 1000)
        params['b'] = f"{bw_mbps}M"
        # Parallel Streams: 4 to 10
        params['P'] = random.randint(4, 10)
        
    elif flow_type == 'video':
        # Duration: 10s to 30s
        params['t'] = random.randint(10, 30)
        # Bandwidth: 10Mbps to 100Mbps
        bw_mbps = random.randint(10, 100)
        params['b'] = f"{bw_mbps}M"
        # Parallel Streams: 2 to 4
        params['P'] = random.randint(2, 4)
        
    elif flow_type == 'interactive':
        # Duration: 5s to 10s
        params['t'] = random.randint(5, 10)
        # Bandwidth: 100Kbps (Fixed or small range)
        params['b'] = "100K"
        params['P'] = 1
        
    else: # 'mice' (Default)
        # Duration: 0.1s to 1s
        params['t'] = random.uniform(0.1, 1.0)
        # Bandwidth: 200Kbps
        params['b'] = "200K"
        params['P'] = 1
        
    return params

# ==============================================================================
# METRIC CALCULATIONS
# ==============================================================================

def calculate_jains_fairness(values):
    if not values or len(values) == 0: return 0.0
    active_values = [v for v in values if v > 0]
    if not active_values: return 0.0
    n = len(active_values)
    sum_x = sum(active_values)
    sum_sq = sum(v**2 for v in active_values)
    if sum_sq == 0: return 0.0
    return (sum_x ** 2) / (n * sum_sq)

def calculate_link_skew(link_bytes_map):
    if not link_bytes_map: return 0.0
    values = list(link_bytes_map.values())
    return np.std(values)

# ==============================================================================
# TRAFFIC GENERATION
# ==============================================================================

def run_iperf_flow(src, dst_ip, duration, bandwidth, streams, flow_type, flow_id):
    """
    Runs a single flow and records Flow Completion Time (FCT).
    """
    FLOW_METRICS[flow_id] = {
        'start': time.time(), 
        'status': 'running', 
        'bytes': 0, 
        'type': flow_type
    }
    
    # -P: Parallel streams, -t: duration, -b: bandwidth, -J: JSON output
    # We assume duration is int, if float (mice), format it
    dur_str = f"{duration:.2f}" if isinstance(duration, float) else str(duration)
    
    cmd = f"iperf3 -c {dst_ip} -t {dur_str} -b {bandwidth} -P {streams} -J"
    
    try:
        result_json = src.cmd(cmd)
        FLOW_METRICS[flow_id]['end'] = time.time()
        
        try:
            data = json.loads(result_json)
            # Sum sent bytes across all streams
            sent_bytes = data['end']['sum_sent']['bytes']
            sender_throughput = data['end']['sum_sent']['bits_per_second'] / 1e6 # Mbps
            
            FLOW_METRICS[flow_id]['status'] = 'completed'
            FLOW_METRICS[flow_id]['bytes'] = sent_bytes
            FLOW_METRICS[flow_id]['throughput'] = sender_throughput
        except:
            FLOW_METRICS[flow_id]['status'] = 'failed'
            
    except Exception as e:
        FLOW_METRICS[flow_id]['status'] = 'error'

def traffic_generator_loop(net, seed):
    """
    Generates randomized traffic between random nodes using specified Flow Types.
    """
    random.seed(seed)
    hosts = net.hosts
    flow_counter = 0
    
    # Flow Type Probabilities (Adjust as needed)
    # Mice are frequent, Elephants are rare
    types = ['mice', 'interactive', 'video', 'elephant']
    weights = [0.4, 0.3, 0.2, 0.1] 
    
    info(f"*** Starting Traffic Gen (Seed: {seed}, Duration: {EXPERIMENT_DURATION}s)...\n")
    
    start_time = time.time()
    
    while not STOP_FLAG.is_set():
        if time.time() - start_time > (EXPERIMENT_DURATION - 5):
            break
            
        # 1. Pick Random Pair
        src = random.choice(hosts)
        dst = random.choice(hosts)
        if src == dst: continue
        
        # 2. Pick Flow Type & Parameters
        f_type = random.choices(types, weights=weights, k=1)[0]
        params = get_random_flow_params(f_type)
        
        # 3. Launch Flow
        flow_id = f"{f_type}_{flow_counter}"
        t = threading.Thread(
            target=run_iperf_flow, 
            args=(src, dst.IP(), params['t'], params['b'], params['P'], f_type, flow_id)
        )
        t.daemon = True
        t.start()
        
        flow_counter += 1
        
        # Inter-arrival time: Poisson-like
        # Heavy flows shouldn't start too frequently to avoid instant saturation
        if f_type == 'elephant':
            time.sleep(random.uniform(2.0, 5.0))
        else:
            time.sleep(random.uniform(0.1, 1.0))

# ==============================================================================
# STATISTICS MONITOR
# ==============================================================================

def monitor_links(net):
    switches = [n for n in net.switches if n.name.startswith('s')]
    snapshot_file = os.path.join(OUTPUT_DIR, "time_series_stats.csv")
    
    with open(snapshot_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "AvgThroughput_Mbps", "LinkSkew", "JainFairness"])
    
    start_t = time.time()
    
    while not STOP_FLAG.is_set():
        current_link_usage = []
        for sw in switches:
            for intf in sw.intfList():
                if intf.name == "lo": continue
                try:
                    with open(f"/sys/class/net/{intf.name}/statistics/tx_bytes", "r") as f:
                        current_link_usage.append(int(f.read().strip()))
                except: pass
        
        skew = calculate_link_skew({i: val for i, val in enumerate(current_link_usage)})
        fairness = calculate_jains_fairness(current_link_usage)
        
        completed = [f['throughput'] for f in FLOW_METRICS.values() if f.get('throughput')]
        avg_thru = sum(completed)/len(completed) if completed else 0
        
        with open(snapshot_file, 'a') as f:
            csv.writer(f).writerow([round(time.time()-start_t, 2), round(avg_thru, 2), round(skew, 4), round(fairness, 4)])
            
        time.sleep(POLL_INTERVAL)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_experiment(net, seed=RANDOM_SEED):
    info("*** Starting iperf3 servers...\n")
    for h in net.hosts: h.cmd("iperf3 -s -p 5001 &")
        
    monitor_t = threading.Thread(target=monitor_links, args=(net,))
    monitor_t.daemon = True
    monitor_t.start()
    
    gen_t = threading.Thread(target=traffic_generator_loop, args=(net, seed))
    gen_t.daemon = True
    gen_t.start()
    
    info(f"*** Experiment Running for {EXPERIMENT_DURATION} seconds...\n")
    for i in range(EXPERIMENT_DURATION):
        time.sleep(1)
        if i % 10 == 0: info(f"Time: {i}/{EXPERIMENT_DURATION}s\n")
            
    STOP_FLAG.set()
    info("*** Stopping...\n")
    time.sleep(5) 
    for h in net.hosts: h.cmd("killall -9 iperf3")
        
    compute_final_report()

def compute_final_report():
    info("\n*** Computing Final Metrics ***\n")
    
    # Group metrics by Flow Type
    type_stats = {}
    
    for fid, data in FLOW_METRICS.items():
        if data['status'] == 'completed':
            ftype = data['type']
            if ftype not in type_stats:
                type_stats[ftype] = {'fct': [], 'thru': []}
            
            duration = data['end'] - data['start']
            type_stats[ftype]['fct'].append(duration)
            type_stats[ftype]['thru'].append(data['throughput'])
    
    summary_file = "final_comparison_results.csv"
    file_exists = os.path.isfile(summary_file)
    # algo = os.environ.get("ROUTING_ALGO", "Unknown")
    algo="RL"
    
    rows = []
    for ftype, stats in type_stats.items():
        rows.append({
            "Algorithm": algo,
            "FlowType": ftype,
            "Count": len(stats['fct']),
            "Avg_FCT": sum(stats['fct'])/len(stats['fct']),
            "Avg_Throughput": sum(stats['thru'])/len(stats['thru'])
        })
        
    print(json.dumps(rows, indent=4))
    
    with open(summary_file, 'a') as f:
        fieldnames = ["Algorithm", "FlowType", "Count", "Avg_FCT", "Avg_Throughput"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        for r in rows: writer.writerow(r)
        
    info(f"*** Results saved to {summary_file} ***\n")

def start_analysis(net):
    run_experiment(net)