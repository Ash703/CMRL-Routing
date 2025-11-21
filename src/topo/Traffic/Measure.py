import time
import threading
import os
import json
import csv
import numpy as np
from mininet.log import info, error

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCHEDULE_FILE = "traffic_schedule.json"
OUTPUT_DIR = "results_metrics"
HANDSHAKE_FILE = "active_flow_config.json"
POLL_INTERVAL = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global State
STOP_FLAG = threading.Event()
FLOW_METRICS = {} 
LINK_STATS = {}

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
# EXECUTION
# ==============================================================================
def run_flow_task(src_node, dst_ip, flow_data):
    """Executes a single flow from the schedule."""
    fid = flow_data['id']
    FLOW_METRICS[fid] = {
        'start_ts': time.time(),
        'status': 'running',
        'type': flow_data['type'],
        'bytes': 0,
        'throughput': 0
    }
    
    # Construct command from schedule data
    # -J for JSON output
    cmd = (f"iperf3 -c {dst_ip} -t {flow_data['duration']} "
           f"-b {flow_data['bandwidth']} -P {flow_data['streams']} -J")
    
    try:
        res = src_node.cmd(cmd)
        FLOW_METRICS[fid]['end_ts'] = time.time()
        
        try:
            data = json.loads(res)
            if 'error' in data:
                info(f"X Flow {fid} Error: {data['error']}\n")
                FLOW_METRICS[fid]['status'] = 'error'
                return

            sent = data['end']['sum_sent']['bytes']
            bps = data['end']['sum_sent']['bits_per_second']
            
            FLOW_METRICS[fid]['status'] = 'completed'
            FLOW_METRICS[fid]['bytes'] = sent
            FLOW_METRICS[fid]['throughput'] = bps / 1e6 # Mbps
            
        except json.JSONDecodeError:
            # If iperf fails (e.g. timeout, connection refused), output isn't JSON
            # info(f"X Flow {fid} Raw Output: {res.strip()[:50]}...\n")
            FLOW_METRICS[fid]['status'] = 'failed_parse'
            
    except Exception as e:
        info(f"X Flow {fid} Exception: {e}\n")
        FLOW_METRICS[fid]['status'] = 'exec_error'

def setup_handshake(net, schedule):
    """Writes the config so the RL controller knows the primary target."""
    primary = schedule['metadata'].get('primary_flow')
    if not primary: return
    
    src = net.get(primary['src'])
    dst = net.get(primary['dst'])
    
    log_path = os.path.abspath(os.path.join(TRAFFIC_LOG_DIR, "rl_target_flow.json"))
    
    cfg = {
        "src_ip": src.IP(), "dst_ip": dst.IP(),
        "log_path": log_path 
    }
    
    with open(os.path.join(TRAFFIC_LOG_DIR, HANDSHAKE_FILE), 'w') as f:
        json.dump(cfg, f)
        
    info(f"*** Handshake Configured: {primary['src']} -> {primary['dst']}\n")

# ==============================================================================
# MONITORING & REPORTING
# ==============================================================================
def monitor_stats(net):
    """
    Monitors real-time network statistics.
    FIXED: Now calculates actual throughput delta instead of logging 0.
    """
    switches = [n for n in net.switches if n.name.startswith('s')]
    csv_file = os.path.join(OUTPUT_DIR, "time_series_stats.csv")
    
    with open(csv_file, 'w') as f:
        csv.writer(f).writerow(["Time", "NetworkThroughput_Mbps", "LinkSkew", "JainFairness"])
        
    start_experiment = time.time()
    
    # Store previous byte counts to calculate delta
    prev_stats = {}
    
    while not STOP_FLAG.is_set():
        iter_start = time.time()
        
        current_usage = []
        total_tx_bytes = 0
        
        # 1. Gather Link Stats
        for sw in switches:
            for intf in sw.intfList():
                if intf.name == "lo": continue
                try:
                    with open(f"/sys/class/net/{intf.name}/statistics/tx_bytes", "r") as f:
                        b = int(f.read().strip())
                        current_usage.append(b)
                        total_tx_bytes += b
                except: pass
        
        # 2. Calculate Metrics
        # Fairness & Skew based on current cumulative counters (distribution matters, not delta)
        skew = calculate_link_skew({i:v for i,v in enumerate(current_usage)})
        fairness = calculate_jains_fairness(current_usage)
        
        # 3. Calculate Throughput Delta
        # Throughput = (Current_Total_Bytes - Prev_Total_Bytes) * 8 / Time_Delta
        network_throughput = 0.0
        if 'total_bytes' in prev_stats:
            delta_bytes = total_tx_bytes - prev_stats['total_bytes']
            delta_time = iter_start - prev_stats['time']
            if delta_time > 0 and delta_bytes > 0:
                network_throughput = (delta_bytes * 8) / (delta_time * 1e6) # Mbps
        
        # Update previous stats
        prev_stats['total_bytes'] = total_tx_bytes
        prev_stats['time'] = iter_start
        
        # 4. Log
        with open(csv_file, 'a') as f:
            csv.writer(f).writerow([
                round(iter_start - start_experiment, 2), 
                round(network_throughput, 2), 
                round(skew, 4), 
                round(fairness, 4)
            ])
        
        time.sleep(POLL_INTERVAL)

# ==============================================================================
# MAIN WRAPPER
# ==============================================================================
def run_replay(net):
    if not os.path.exists(SCHEDULE_FILE):
        error(f"Schedule file {SCHEDULE_FILE} not found! Run traffic_scheduler.py first.\n")
        return

    with open(SCHEDULE_FILE, 'r') as f:
        schedule = json.load(f)

    os.system(f"rm -f {TRAFFIC_LOG_DIR}/*.json")
    info("*** Starting servers...\n")
    for h in net.hosts: h.cmd("iperf3 -s -p 5001 &")

    setup_handshake(net, schedule)

    # Inject logfile path for primary flow so controller can read it
    for f in schedule['flows']:
        if f['type'] == 'primary':
            f['logfile'] = os.path.abspath(os.path.join(TRAFFIC_LOG_DIR, "rl_target_flow.json"))

    # Start Monitor
    monitor_t = threading.Thread(target=monitor_stats, args=(net,))
    monitor_t.daemon = True
    monitor_t.start()

    # Start Flows
    flows = schedule['flows']
    start_time = time.time()
    idx = 0
    
    # We define a specific wrapper for the Primary flow to capture its output to file
    def primary_flow_wrapper(src, dst, data):
        cmd = (f"iperf3 -c {dst} -t {data['duration']} "
               f"-b {data['bandwidth']} -P {data['streams']} -J > {data['logfile']}")
        try:
            src.cmd(cmd)
            # Read back result for metrics
            with open(data['logfile'], 'r') as lf:
                js = json.load(lf)
                sent = js['end']['sum_sent']['bytes']
                bps = js['end']['sum_sent']['bits_per_second']
                FLOW_METRICS[data['id']] = {
                    'status': 'completed', 'bytes': sent, 'throughput': bps/1e6, 'type': 'primary'
                }
        except: 
            FLOW_METRICS[data['id']] = {'status': 'failed', 'type': 'primary'}

    while idx < len(flows) and not STOP_FLAG.is_set():
        now = time.time() - start_time
        nf = flows[idx]
        
        if now >= nf['start_time']:
            src = net.get(nf['src'])
            dst = net.get(nf['dst'])
            
            if nf['type'] == 'primary':
                t = threading.Thread(target=primary_flow_wrapper, args=(src, dst.IP(), nf))
            else:
                t = threading.Thread(target=run_flow_task, args=(src, dst.IP(), nf))
                
            t.daemon = True
            t.start()
            idx += 1
        else:
            time.sleep(0.1)
            
    remaining = schedule['metadata']['duration'] - (time.time() - start_time)
    if remaining > 0: 
        info(f"*** Waiting {int(remaining)}s for completion...\n")
        time.sleep(remaining)

    STOP_FLAG.set()
    time.sleep(2)
    for h in net.hosts: h.cmd("killall -9 iperf3")
    
    compute_final_report()

def compute_final_report():
    info("\n*** Final Metrics ***\n")
    type_stats = {}
    
    for fid, d in FLOW_METRICS.items():
        if d.get('status') == 'completed':
            t = d['type']
            if t not in type_stats: type_stats[t] = {'thru':[]}
            type_stats[t]['thru'].append(d['throughput'])

    csv_file = "final_comparison_results.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, "a") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["Algorithm", "FlowType", "Count", "Avg_Throughput_Mbps"])
            
        algo = os.environ.get("ROUTING_ALGO", "Unknown")
        for t, s in type_stats.items():
            avg_thru = sum(s['thru'])/len(s['thru'])
            w.writerow([algo, t, len(s['thru']), avg_thru])
            print(f"Type: {t:<12} | Avg Thru: {avg_thru:.2f} Mbps")
            
    info(f"Saved to {csv_file}\n")

def start_analysis(net):
    run_replay(net)