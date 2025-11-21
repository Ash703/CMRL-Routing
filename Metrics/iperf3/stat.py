import json
import matplotlib.pyplot as plt
import os
import numpy as np
FILES_TO_PLOT = {
    "RL Algorithm": "Metrics\iperf3\iperf_rl.json",
    # "ECMP": "Metrics\iperf3\iperf_ecmp.json",      
    # "Greedy": "Metrics\iperf3\iperf_greedy.json",  
    # "Static": "Metrics\iperf3\iperf_static.json",
    # "Round Robin": "Metrics\iperf3\iperf_roundrobin.json",
    # "Random": "Metrics\iperf3\iperf_random.json"
}

COLORS = {
    "RL Algorithm": "blue",
    "ECMP": "orange",
    "Greedy": "green",
    "Static": "red",
    "Round Robin": "purple",
    "Random": "cyan"
}

def parse_iperf_json(filename):
    """Parses iperf3 JSON to extract time-series data."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None

    with open(filename, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding {filename}")
            return None

    times = []
    throughputs = [] # Mbps
    retransmits = []
    
    # Extract interval data
    if 'intervals' not in data:
        print(f"No intervals found in {filename}")
        return None

    start_time = data['start']['timestamp']['timesecs']

    for i, interval in enumerate(data['intervals']):
        # Time offset (0, 1, 2...)
        times.append(interval['sum']['end'])
        
        # Throughput in Mbps
        mbps = interval['sum']['bits_per_second'] / 1e6
        throughputs.append(mbps)
        
        # Retransmits (TCP Analysis)
        # Note: retransmits are usually in the 'sum' object for TCP
        r = interval['sum'].get('retransmits', 0)
        retransmits.append(r)

    # Extract Summary Data
    avg_throughput = data['end']['sum_received']['bits_per_second'] / 1e6
    total_retransmits = data['end']['sum_sent']['retransmits']

    return {
        "time": times,
        "throughput": throughputs,
        "retransmits": retransmits,
        "avg_throughput": avg_throughput,
        "total_retransmits": total_retransmits
    }

def plot_graphs(all_data):
    # --- PLOT 1: Throughput Over Time ---
    plt.figure(figsize=(12, 6))
    
    for label, data in all_data.items():
        color = COLORS.get(label, "black")
        plt.plot(data['time'], data['throughput'], label=label, 
                 color=color, linewidth=2, marker='o', markersize=3)

    plt.title("Throughput Stability Comparison (Elephant Flow)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Throughput (Mbps)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("Metrics/iperf3/graph_throughput_timeseries_RL.png")
    print("Saved: graph_throughput_timeseries.png")
    plt.close()

    # --- PLOT 2: Retransmissions (Congestion Indicator) ---
    plt.figure(figsize=(12, 6))
    
    for label, data in all_data.items():
        color = COLORS.get(label, "black")
        # Use step plot for discrete events like packet loss
        plt.step(data['time'], data['retransmits'], label=label, 
                 color=color, linewidth=2, where='mid')

    plt.title("TCP Retransmissions over Time (Congestion Events)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Retransmits (Count)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("Metrics/iperf3/graph_retransmits_RL.png")
    print("Saved: graph_retransmits.png")
    plt.close()

    # --- PLOT 3: Summary Bar Chart ---
    plt.figure(figsize=(10, 6))
    labels = list(all_data.keys())
    avgs = [d['avg_throughput'] for d in all_data.values()]
    
    bars = plt.bar(labels, avgs, color=[COLORS.get(x, 'gray') for x in labels])
    
    plt.title("Average Throughput Comparison")
    plt.ylabel("Avg Mbps")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', va='bottom', ha='center')

    plt.savefig("Metrics/iperf3/graph_throughput_bar_RL.png")
    print("Saved: graph_throughput_bar.png")
    plt.close()

if __name__ == "__main__":
    dataset = {}
    
    # 1. Load Data
    for algo_name, fname in FILES_TO_PLOT.items():
        print(f"Processing {algo_name}...")
        res = parse_iperf_json(fname)
        if res:
            dataset[algo_name] = res
            
    # 2. Plot
    if dataset:
        plot_graphs(dataset)
    else:
        print("No valid data files found.")