import json
import matplotlib.pyplot as plt
import os
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
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

def parse_tcp_internals(filename):
    if not os.path.exists(filename): return None
    
    with open(filename, 'r') as f:
        try:
            data = json.load(f)
        except: return None

    if 'intervals' not in data: return None

    times = []
    avg_cwnd = []
    avg_rttvar = []

    for interval in data['intervals']:
        times.append(interval['sum']['end'])
        
        # Extract data from all parallel streams in this interval
        cwnds = []
        rttvars = []
        for stream in interval['streams']:
            cwnds.append(stream.get('snd_cwnd', 0) / 1024) # Convert Bytes to KB
            rttvars.append(stream.get('rttvar', 0) / 1000) # Convert microsec to ms
            
        # Average across the parallel streams for this second
        if cwnds:
            avg_cwnd.append(sum(cwnds) / len(cwnds))
            avg_rttvar.append(sum(rttvars) / len(rttvars))
        else:
            avg_cwnd.append(0)
            avg_rttvar.append(0)

    return {"time": times, "cwnd": avg_cwnd, "rttvar": avg_rttvar}

def plot_advanced_graphs(all_data):
    # --- PLOT 1: Congestion Window (CWND) ---
    plt.figure(figsize=(12, 6))
    for label, d in all_data.items():
        c = COLORS.get(label, "black")
        plt.plot(d['time'], d['cwnd'], label=label, color=c, linewidth=1.5)
        
    plt.title("TCP Congestion Window (CWND) Evolution")
    plt.ylabel("Congestion Window Size (KB)")
    plt.xlabel("Time (seconds)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.savefig("Metrics\iperf3\graph_tcp_cwnd_RL.png")
    print("Saved: graph_tcp_cwnd.png")
    plt.close()

    # --- PLOT 2: RTT Variance (Jitter) ---
    plt.figure(figsize=(12, 6))
    for label, d in all_data.items():
        c = COLORS.get(label, "black")
        plt.plot(d['time'], d['rttvar'], label=label, color=c, linewidth=1.5)
        
    plt.title("Jitter (RTT Variance) Stability")
    plt.ylabel("Variance (ms)")
    plt.xlabel("Time (seconds)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.savefig("Metrics\iperf3\graph_tcp_jitter_RL.png")
    print("Saved: graph_tcp_jitter.png")
    plt.close()

if __name__ == "__main__":
    dataset = {}
    for name, fname in FILES_TO_PLOT.items():
        res = parse_tcp_internals(fname)
        if res: dataset[name] = res
    
    if dataset: plot_advanced_graphs(dataset)
    else: print("No data found.")