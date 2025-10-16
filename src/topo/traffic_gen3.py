"""
Traffic generator for leaf-spine Mininet topologies.
Automatically opens xterms for servers and clients to show live traffic.
Supports contextual flows: mice, elephant, video, interactive.
"""

import os
import random
import time
from mininet.log import info

# Experiment duration in seconds
EXPERIMENT_TIME = 60

# Flow types with parameters: t=duration (s), b=bandwidth (Kbps), P=parallel streams
FLOW_TYPES = {
    'mice': {'t': 0.2, 'b': 200, 'P': 1},
    'elephant': {'t': 30, 'b': 5000, 'P': 4},
    'video': {'t': 15, 'b': 2000, 'P': 2},
    'interactive': {'t': 5, 'b': 100, 'P': 1}
}

def start_servers(net):
    """
    Start iperf3 servers on all hosts in xterms for live monitoring.
    """
    info("*** Starting iperf3 servers on all hosts\n")
    for host in net.hosts:
        if os.environ.get("DISPLAY"):
            host.cmd(f"xterm -T '{host.name} server' -hold -e 'iperf3 -s' &")
        else:
            host.cmd("iperf3 -s &")

def generate_traffic(net, num_flows=10):
    """
    Generate flows between hosts, clients run in background, servers show traffic in xterms.
    """
    hosts = net.hosts
    if len(hosts) < 2:
        raise ValueError("Need at least 2 hosts to generate traffic!")

    info("*** Generating traffic flows\n")

    for i in range(num_flows):
        src, dst = random.sample(hosts, 2)
        flow_type = random.choice(list(FLOW_TYPES.keys()))
        params = FLOW_TYPES[flow_type]

        cmd = f"iperf3 -c {dst.IP()} -t {params['t']} -b {params['b']}K -P {params['P']} -i 1"

        info(f"*** Flow {i+1}: {src.name} -> {dst.name} | Type: {flow_type} | "
             f"Bandwidth: {params['b']}K | Duration: {params['t']}s\n")

        # Run client in background (no xterm)
        src.cmd(cmd + " &")

        # Slight stagger for realism
        time.sleep(random.uniform(0.5, 2))

def stop_traffic(net):
    """
    Kill all iperf3 processes on hosts
    """
    info("*** Stopping all iperf3 processes\n")
    for h in net.hosts:
        h.cmd("pkill iperf3")

if __name__ == "__main__":
    print("⚠️ This script is intended to be imported and called from your topology script.")
