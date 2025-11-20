import time
import random
import threading
from mininet.log import info

# Global control flag to stop all traffic threads gracefully
STOP_FLAG = threading.Event()
# List to hold references to all running flow threads
FLOW_THREADS = []

def start_iperf_server(host):
    """Starts an iperf server on a given Mininet host in the background."""
    info(f"*** Starting iperf server on {host.name} ({host.IP()})\n")
    # Use host.cmd() to run iperf server in the background
    host.cmd("iperf3 -s -p 5001 > /tmp/iperf-server-log-{}.txt 2>&1 &".format(host.name))

def stop_all_iperf_servers(net):
    """Stops all running iperf processes on all hosts."""
    info("*** Stopping all iperf processes...\n")
    for host in net.hosts:
        # Kill any existing iperf processes
        host.cmd("killall -9 iperf3")

def run_traffic_flow(src_host, dst_ip, duration_sec, bandwidth):
    """
    Runs a single iperf client flow between a source host and destination IP.
    
    This function is executed in a separate thread.
    """
    cmd = (
        f"iperf3 -c {dst_ip} -p 5001 -t {duration_sec} -b {bandwidth} "
        f"-y C > /dev/null 2>&1" # -y C for CSV output, redirected to null
    )
    
    info(f"[FLOW] {src_host.name} -> {dst_ip}: {bandwidth} for {duration_sec}s\n")
    
    # Run the iperf command on the source host
    src_host.cmd(cmd)

def traffic_generation_loop(net, flow_interval_sec=1.5, min_duration=5, max_duration=15):
    """
    The main thread that periodically launches new random traffic flows.
    """
    hosts = net.hosts
    host_ips = [h.IP() for h in hosts]
    
    # Define a range of typical bandwidths (e.g., 10M, 20M, ..., 100M)
    bandwidth_options = [f"{i}M" for i in range(10, 101, 10)]

    info(f"*** Starting periodic traffic generation loop (interval: {flow_interval_sec}s)...\n")
    
    while not STOP_FLAG.is_set():
        if len(hosts) < 2:
            info("Not enough hosts to generate traffic.\n")
            time.sleep(flow_interval_sec)
            continue
        
        # 1. Randomly select source and destination hosts
        src_host = random.choice(hosts)
        
        # Ensure destination is different from source
        dst_ip = random.choice([ip for ip in host_ips if ip != src_host.IP()])
        
        # 2. Randomly select flow parameters
        duration = random.randint(min_duration, max_duration)
        bandwidth = random.choice(bandwidth_options)
        
        # 3. Start the flow in a new thread
        flow_thread = threading.Thread(
            target=run_traffic_flow, 
            args=(src_host, dst_ip, duration, bandwidth)
        )
        flow_thread.daemon = True # Allows the thread to exit with the main program
        flow_thread.start()
        FLOW_THREADS.append(flow_thread)
        
        # Wait for the next interval, respecting the stop flag
        time.sleep(flow_interval_sec)

def generate_traffic(net, flow_interval_sec=1.5, min_duration=5, max_duration=15):
    """
    Initializes iperf servers and starts the periodic traffic generation loop.
    
    :param net: The Mininet network object.
    :param flow_interval_sec: The delay between starting new flows.
    :param min_duration: Minimum duration of a flow in seconds.
    :param max_duration: Maximum duration of a flow in seconds.
    """
    global TRAFFIC_LOOP_THREAD
    
    # 1. Start iperf server on all hosts
    for host in net.hosts:
        start_iperf_server(host)
        
    # 2. Start the main traffic generation loop in a separate thread
    TRAFFIC_LOOP_THREAD = threading.Thread(
        target=traffic_generation_loop, 
        args=(net, flow_interval_sec, min_duration, max_duration)
    )
    TRAFFIC_LOOP_THREAD.daemon = True
    TRAFFIC_LOOP_THREAD.start()
    info("\n*** Traffic generator initialized. Flows are being launched dynamically.\n")

def stop_traffic(net):
    """
    Stops the traffic generation loop and cleans up iperf servers.
    """
    global TRAFFIC_LOOP_THREAD
    info("\n*** Signalling traffic generator to stop...\n")
    
    # 1. Set the stop flag
    STOP_FLAG.set()
    
    # 2. Wait for the main generation thread to finish (if it was running)
    if 'TRAFFIC_LOOP_THREAD' in globals() and TRAFFIC_LOOP_THREAD.is_alive():
        # Give it a brief time to exit the loop
        TRAFFIC_LOOP_THREAD.join(timeout=2) 
    
    # 3. Clean up all iperf processes
    stop_all_iperf_servers(net)
    
    info("*** Traffic generation stopped and cleaned up.\n")

if __name__ == "__main__":
    # This block is for testing the functions independently if needed, 
    # but typically this file is imported and used by the Mininet script.
    info("This module is designed to be imported into the Mininet topology script.\n")