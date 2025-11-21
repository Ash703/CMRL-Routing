import matplotlib.pyplot as plt
import re
import numpy as np
import os

FILES_TO_PLOT = {
    # "RL Algorithm": "Metrics/ping_rl.txt",
    "ECMP": "Metrics/ping_ecmp.txt",       
    "Greedy": "Metrics/ping_greedy.txt",    
    "Random": "Metrics/ping_rand.txt",
    "Round Robin": "Metrics/ping_rr.txt",
    "Static": "Metrics/ping_static.txt",
    
}

def parse_ping_file(filename):
    """Parses raw ping output to extract sequence number and time."""
    sequences = []
    times = []
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Skipping.")
        return [], []

    with open(filename, 'r') as f:
        for line in f:
            # Regex to find "icmp_seq=X" and "time=Y ms"
            match = re.search(r'icmp_seq=(\d+).*time=([\d.]+)', line)
            if match:
                seq = int(match.group(1))
                rtt = float(match.group(2))
                sequences.append(seq)
                times.append(rtt)
    return sequences, times

def plot_time_series(all_data):
    plt.figure(figsize=(10, 5))
    
    for label, (seq, rtt) in all_data.items():
        if not seq: continue
        plt.plot(seq, rtt, marker='o', markersize=2, linestyle='-', label=label, linewidth=1)

    plt.title("Round-Trip Time (RTT) Time Series")
    plt.xlabel("ICMP Sequence Number")
    plt.ylabel("Latency (ms)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Highlight the danger zone (Bufferbloat)
    plt.axhline(y=100, color='r', linestyle=':', alpha=0.5)
    plt.text(0, 120, 'High Congestion Threshold', color='red', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("Metrics/graph_rtt_timeseries_rest.png")
    print("Generated: graph_rtt_timeseries.png")
    plt.close()

def plot_cdf(all_data):
    plt.figure(figsize=(10, 5))
    
    for label, (seq, rtt) in all_data.items():
        if not rtt: continue
        
        # Sort data for CDF
        sorted_rtt = np.sort(rtt)
        # Calculate cumulative probability (0 to 1)
        yvals = np.arange(len(sorted_rtt)) / float(len(sorted_rtt) - 1)
        
        plt.plot(sorted_rtt, yvals, label=label, linewidth=2)

    plt.title("Latency CDF (Cumulative Distribution Function)")
    plt.xlabel("Latency (ms) - Log Scale")
    plt.ylabel("Probability (CDF)")
    plt.xscale('log') # Log scale is CRITICAL because your spikes are 1000x larger than normal
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("Metrics/graph_rtt_cdf_rest.png")
    print("Generated: graph_rtt_cdf.png")
    plt.close()

if __name__ == "__main__":
    # 1. Load Data
    all_data = {}
    for label, fname in FILES_TO_PLOT.items():
        seq, rtt = parse_ping_file(fname)
        if seq:
            all_data[label] = (seq, rtt)
            print(f"Loaded {label}: {len(seq)} packets. Max RTT: {max(rtt)}ms")

    # 2. Generate Plots
    if all_data:
        plot_time_series(all_data)
        plot_cdf(all_data)
    else:
        print("No data found. Please create 'ping_rl.txt' with your data.")