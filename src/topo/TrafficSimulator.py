import time
import random
import threading
import os
from mininet.log import info, error

# Configuration
TRAFFIC_LOG_DIR = "./traffic_logs" # The directory to store all iperf3 log files

# Global control flag to stop all traffic threads gracefully
STOP_FLAG = threading.Event()
# List to hold references to all running flow threads
FLOW_THREADS = []

def ensure_log_directory_exists():
    """Creates the dedicated log directory if it does not already exist."""
    os.makedirs(TRAFFIC_LOG_DIR, exist_ok=True)
    info(f"*** Ensured log directory exists: {TRAFFIC_LOG_DIR}\n")

# --- iperf3 Utility Functions ---

def start_iperf_server(host):
    """Starts an iperf3 server on a given Mininet host in the background."""
    # Ensure the log directory is ready
    ensure_log_directory_exists()

    log_file = os.path.join(TRAFFIC_LOG_DIR, f"iperf3-server-log-{host.name}.txt")
    info(f"*** Starting iperf3 server on {host.name}, logging to {log_file}\n")
    
    # Run iperf3 server in the background, redirecting output to the log file
    host.cmd(f"iperf3 -s -p 5001 > {log_file} 2>&1 &")

def stop_all_iperf_servers(net):
    """Stops all running iperf3 processes on all hosts."""
    info("*** Stopping all iperf3 processes...\n")
    # Kill any existing iperf3 processes (both client and server)
    for host in net.hosts:
        host.cmd("killall -9 iperf3")

# --- Specific H1 -> H4 Parallel Flow (RL Target) ---

def start_h1_h4_parallel_flow(net):
    """
    Starts a persistent, high-bandwidth, 4-stream parallel flow from h1 to h4.
    The output is saved as JSON for easy parsing by the RL agent.
    """
    try:
        ensure_log_directory_exists()
        
        # Find host objects
        h1 = net.get('h1')
        h4 = net.get('h4')
        h4_ip = h4.IP()
        
        # Log file for the primary flow (JSON format)
        log_file = os.path.join(TRAFFIC_LOG_DIR, "h1_h4_parallel_flow.json")
        
        # Parameters for the persistent flow
        duration_sec = 3600 # Run for 1 hour
        bandwidth = "50M" 
        parallel_streams = 4 
        
        # Command uses --json output and directs it to the log file
        cmd = (
            f"iperf3 -c {h4_ip} -p 5001 -t {duration_sec} -b {bandwidth} -P {parallel_streams} "
            f"--json > {log_file} &" 
        )
        
        info(f"*** Starting PRIMARY RL Flow (h1 -> h4): {bandwidth} via {parallel_streams} streams. Results to {log_file}.\n")
        
        # Run the iperf3 command on h1 in the background
        h1.cmd(cmd)

    except Exception as e:
        error(f"Failed to start H1 -> H4 parallel flow. Ensure h1 and h4 exist. Error: {e}\n")

# --- Random Background Traffic Generation ---

def run_traffic_flow(src_host, dst_ip, duration_sec, bandwidth):
    """
    Runs a single iperf3 client flow between a source host and destination IP.
    The output is discarded as these are short-lived background flows.
    """
    # iperf3 client command. Output is discarded to prevent file spam.
    cmd = (
        f"iperf3 -c {dst_ip} -p 5001 -t {duration_sec} -b {bandwidth} "
        f"--json > /dev/null 2>&1" 
    )
    
    info(f"[BG FLOW] {src_host.name} -> {dst_ip}: {bandwidth} for {duration_sec}s\n")
    
    # Run the iperf command on the source host
    src_host.cmd(cmd)

def traffic_generation_loop(net, flow_interval_sec=1.5, min_duration=5, max_duration=15):
    """
    The main thread that periodically launches new random background traffic flows.
    """
    hosts = net.hosts
    # Define a range of typical bandwidths 
    bandwidth_options = [f"{i}M" for i in range(5, 31, 5)]

    info(f"*** Starting periodic background traffic generation loop (interval: {flow_interval_sec}s)...\n")
    
    while not STOP_FLAG.is_set():
        if len(hosts) < 2:
            time.sleep(flow_interval_sec)
            continue
            
        # 1. Randomly select source and destination hosts
        src_host = random.choice(hosts)
        dst_ip = random.choice([h.IP() for h in hosts if h.IP() != src_host.IP()])
        
        # 2. Randomly select flow parameters
        duration = random.randint(min_duration, max_duration)
        bandwidth = random.choice(bandwidth_options)
        
        # 3. Start the flow in a new thread
        flow_thread = threading.Thread(
            target=run_traffic_flow, 
            args=(src_host, dst_ip, duration, bandwidth)
        )
        flow_thread.daemon = True 
        flow_thread.start()
        FLOW_THREADS.append(flow_thread)
        
        # Wait for the next interval, respecting the stop flag
        time.sleep(flow_interval_sec)

# --- Main Interface ---

def generate_traffic(net, flow_interval_sec=1.5, min_duration=5, max_duration=15):
    """
    Initializes iperf3 servers and starts both the persistent and random traffic flows.
    """
    global TRAFFIC_LOOP_THREAD
    
    info("*** Traffic Generator: Initializing...\n")
    
    # 1. Clean up old processes
    stop_all_iperf_servers(net)
    
    # 2. Start iperf3 server on all hosts
    for host in net.hosts:
        start_iperf_server(host)
        
    # 3. Start the PRIMARY RL Flow (h1 -> h4)
    start_h1_h4_parallel_flow(net)
        
    # 4. Start the background traffic generation loop in a separate thread
    TRAFFIC_LOOP_THREAD = threading.Thread(
        target=traffic_generation_loop, 
        args=(net, flow_interval_sec, min_duration, max_duration)
    )
    TRAFFIC_LOOP_THREAD.daemon = True
    TRAFFIC_LOOP_THREAD.start()
    info("\n*** Traffic generator running. Check ./traffic_logs/ for h1_h4_parallel_flow.json\n")

def stop_traffic(net):
    """
    Stops the traffic generation loop and cleans up iperf3 servers/clients.
    """
    global TRAFFIC_LOOP_THREAD
    info("\n*** Signalling traffic generator to stop...\n")
    
    STOP_FLAG.set()
    
    if 'TRAFFIC_LOOP_THREAD' in globals() and TRAFFIC_LOOP_THREAD.is_alive():
        TRAFFIC_LOOP_THREAD.join(timeout=2) 
    
    # Clean up all iperf3 processes
    stop_all_iperf_servers(net)
    
    info("*** Traffic generation stopped and cleaned up.\n")

if __name__ == "__main__":
    info("This module is designed to be imported into the Mininet topology script.\n")