import time
import random
import threading
import os
import json
import yaml
from mininet.log import info, error

# Configuration
TRAFFIC_LOG_DIR = "./traffic_logs"
CONFIG_FILE = "active_flow_config.json" # Handshake file
STOP_FLAG = threading.Event()
FLOW_THREADS = []

def ensure_log_directory_exists():
    os.makedirs(TRAFFIC_LOG_DIR, exist_ok=True)

# --- iperf3 Utility Functions ---
def start_iperf_server(host):
    ensure_log_directory_exists()
    log_file = os.path.join(TRAFFIC_LOG_DIR, f"iperf3-server-log-{host.name}.txt")
    host.cmd(f"iperf3 -s -p 5001 > {log_file} 2>&1 &")

def stop_all_iperf_servers(net):
    info("*** Stopping all iperf3 processes...\n")
    for host in net.hosts:
        host.cmd("killall -9 iperf3")

# --- Dynamic RL Target Selection ---
def select_rl_target_pair(net, network_config_path):
    """
    Intelligently picks a Source and Destination host that are on DIFFERENT leaves.
    This ensures the traffic must cross the Spines (crucial for RL routing).
    """
    with open(network_config_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    
    # Group hosts by their connected Leaf switch
    leaf_to_hosts = {}
    host_lookup = {}
    
    for h_conf in raw_cfg['hosts']:
        leaf = h_conf['connected_to']
        if leaf not in leaf_to_hosts:
            leaf_to_hosts[leaf] = []
        leaf_to_hosts[leaf].append(h_conf['name'])
        host_lookup[h_conf['name']] = h_conf
        
    leaves = list(leaf_to_hosts.keys())
    if len(leaves) < 2:
        info("*** WARNING: Only 1 Leaf found. Cannot generate cross-spine traffic.\n")
        return net.hosts[0], net.hosts[-1]

    # Pick two different leaves
    src_leaf = leaves[0]
    dst_leaf = leaves[1] # Simple default: first two leaves
    
    # Pick first host from each
    src_name = leaf_to_hosts[src_leaf][0]
    dst_name = leaf_to_hosts[dst_leaf][0]
    
    info(f"*** Auto-selected RL Target: {src_name} (on {src_leaf}) -> {dst_name} (on {dst_leaf})\n")
    
    return net.get(src_name), net.get(dst_name)

def start_target_flow(net, src_host, dst_host):
    """Start the long-running flow and write config for Controller."""
    try:
        ensure_log_directory_exists()
        
        # 1. Define the Flow
        log_filename = "rl_target_flow.json"
        log_path = os.path.join(TRAFFIC_LOG_DIR, log_filename)
        
        duration_sec = 86400 # 24 hours (effectively infinite)
        bandwidth = "50M"
        streams = 4
        
        # 2. Start iperf3
        cmd = (
            f"iperf3 -c {dst_host.IP()} -p 5001 -t {duration_sec} -b {bandwidth} -P {streams} "
            f"--json > {log_path} &"
        )
        src_host.cmd(cmd)
        
        # 3. Write Handshake File for Controller
        config_data = {
            "src_ip": src_host.IP(),
            "dst_ip": dst_host.IP(),
            "src_name": src_host.name,
            "dst_name": dst_host.name,
            "log_path": log_path
        }
        
        with open(os.path.join(TRAFFIC_LOG_DIR, CONFIG_FILE), 'w') as f:
            json.dump(config_data, f)
            
        info(f"*** RL Flow Started. Config written to {CONFIG_FILE}\n")

    except Exception as e:
        error(f"Failed to start Target flow: {e}\n")

# --- Random Background Traffic ---
def run_traffic_flow(src_host, dst_ip, duration_sec, bandwidth):
    cmd = f"iperf3 -c {dst_ip} -p 5001 -t {duration_sec} -b {bandwidth} --json > /dev/null 2>&1"
    info(f"[BG] {src_host.name} -> {dst_ip} ({bandwidth})\n")
    src_host.cmd(cmd)

def traffic_generation_loop(net, target_src, target_dst, interval=1.5):
    hosts = net.hosts
    bw_opts = [f"{i}M" for i in range(5, 25, 5)]
    
    while not STOP_FLAG.is_set():
        if len(hosts) < 2: continue
        
        # Pick random pair
        src = random.choice(hosts)
        dst = random.choice(hosts)
        
        # Avoid self-traffic and avoid overwhelming the RL target hosts
        if src == dst: continue
        # Optional: Don't generate background traffic on the exact RL target pair to keep signal clean
        if src == target_src and dst == target_dst: continue 
        
        dur = random.randint(5, 10)
        bw = random.choice(bw_opts)
        
        t = threading.Thread(target=run_traffic_flow, args=(src, dst.IP(), dur, bw))
        t.daemon = True
        t.start()
        FLOW_THREADS.append(t)
        
        time.sleep(interval)

# --- Main Interface ---
def generate_traffic(net, flow_interval_sec=2.0):
    global TRAFFIC_LOOP_THREAD
    
    # We need the config path to find leaves. 
    # Assuming standard location or passed via env. Defaulting to 'network_config.yaml'
    config_path = os.environ.get("NETWORK_CONFIG_FILE", "network_config.yaml")
    
    ensure_log_directory_exists()
    stop_all_iperf_servers(net)
    
    for h in net.hosts: start_iperf_server(h)
    
    # 1. Pick Target
    src_h, dst_h = select_rl_target_pair(net, config_path)
    
    # 2. Start Target
    start_target_flow(net, src_h, dst_h)
    
    # 3. Start Loop
    TRAFFIC_LOOP_THREAD = threading.Thread(
        target=traffic_generation_loop, 
        args=(net, src_h, dst_h, flow_interval_sec)
    )
    TRAFFIC_LOOP_THREAD.daemon = True
    TRAFFIC_LOOP_THREAD.start()

def stop_traffic(net):
    STOP_FLAG.set()
    stop_all_iperf_servers(net)