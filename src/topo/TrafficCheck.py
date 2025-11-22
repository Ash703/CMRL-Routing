import time
import random
import threading
import os
import math
from mininet.log import info, error

# Configuration
TRAFFIC_LOG_DIR = "/home/Ash/Documents/ACN/CMRL-Routing"  # The directory to store iperf3 log files

# Global control flag to stop all traffic threads gracefully
STOP_FLAG = threading.Event()
FLOW_THREADS = []
MASTER_RUN_SECONDS = 120  # 2 minutes

def ensure_log_directory_exists():
    os.makedirs(TRAFFIC_LOG_DIR, exist_ok=True)
    info(f"*** Ensured log directory exists: {TRAFFIC_LOG_DIR}\n")

# -------------------- iperf3 Utility --------------------

def start_iperf_server(host, port=5001):
    ensure_log_directory_exists()
    log_file = os.path.join(TRAFFIC_LOG_DIR, f"iperf3-server-log-{host.name}.txt")
    info(f"*** Starting iperf3 server on {host.name}, logging to {log_file}\n")
    host.cmd(f"iperf3 -s -p {port} > {log_file} 2>&1 &")

def stop_all_iperf_servers(net):
    info("*** Stopping all iperf3 processes...\n")
    for host in net.hosts:
        host.cmd("killall -9 iperf3 > /dev/null 2>&1")

# -------------------- Flow Type Ranges --------------------

FLOW_TYPES = {
    'mice': {'t': 0.2, 'b': 200, 'P': 1},
    'elephant': {'t': 30, 'b': 5000, 'P': 4},
    'video': {'t': 15, 'b': 2000, 'P': 2},
    'interactive': {'t': 5, 'b': 100, 'P': 1}
}

# UPDATED: mice & interactive now in Kbps
FLOW_RANGES = {
    'mice': {
        't': (0.1, 1.0),
        'b': (0.05, 1.0),        # 50 Kbps – 1 Mbps
        'P': (1, 2)
    },
    'elephant': {
        't': (20, 40),
        'b': (100, 1000),        # Mbps
        'P': (2, 8)
    },
    'video': {
        't': (10, 30),
        'b': (2, 80),            # Mbps
        'P': (1, 4)
    },
    'interactive': {
        't': (5, 10),
        'b': (0.05, 0.2),        # 50–200 Kbps
        'P': (1, 1)
    }
}

FLOW_TYPE_WEIGHTS = {
    'mice': 0.6,
    'interactive': 0.15,
    'video': 0.15,
    'elephant': 0.1
}

def choose_flow_type():
    types = list(FLOW_TYPE_WEIGHTS.keys())
    w = [FLOW_TYPE_WEIGHTS[t] for t in types]
    return random.choices(types, weights=w, k=1)[0]

def format_bandwidth_mbps(mbps_float):
    if mbps_float >= 1:
        if mbps_float >= 1000:
            g = mbps_float / 1000.0
            return f"{g:.3f}G" if not math.isclose(g, round(g)) else f"{int(round(g))}G"
        return f"{mbps_float:.1f}M"
    else:
        # <1 Mbps → show as Kbps
        kbps = mbps_float * 1000
        return f"{int(kbps)}K"

# -------------------- Run One Flow --------------------

def run_traffic_flow(src_host, dst_ip, duration_sec, bandwidth_mbps,
                     parallel_streams=1, port=5001, json_logfile=None):

    duration_sec = min(duration_sec, MASTER_RUN_SECONDS)
    bw_arg = format_bandwidth_mbps(float(bandwidth_mbps))
    pstreams = max(1, int(parallel_streams))

    if json_logfile:
        cmd = (
            f"iperf3 -c {dst_ip} -p {port} -t {int(duration_sec)} "
            f"-b {bw_arg} -P {pstreams} --json > {json_logfile} 2>&1"
        )
    else:
        cmd = (
            f"iperf3 -c {dst_ip} -p {port} -t {int(duration_sec)} "
            f"-b {bw_arg} -P {pstreams} > /dev/null 2>&1"
        )

    info(f"[FLOW START] {src_host.name}->{dst_ip} {duration_sec}s bw={bw_arg} P={pstreams}\n")
    src_host.cmd(cmd)
    info(f"[FLOW END] {src_host.name}->{dst_ip}\n")

# -------------------- Random Param Sampler --------------------

def sample_flow_params(ftype):
    r = FLOW_RANGES[ftype]
    duration = random.uniform(*r['t'])
    bandwidth = random.uniform(*r['b'])
    pstreams = random.randint(int(r['P'][0]), int(r['P'][1]))
    return duration, bandwidth, pstreams

# -------------------- Primary RL Flow --------------------

def start_h1_h4_parallel_flow(net, port=5001):
    try:
        ensure_log_directory_exists()
        h1 = net.get('h1')
        h4 = net.get('h4')
    except Exception:
        error("*** h1/h4 not found\n")
        return

    duration, bandwidth, pstreams = sample_flow_params('elephant')
    duration = min(duration, MASTER_RUN_SECONDS)
    log_file = os.path.join(TRAFFIC_LOG_DIR, "h1_h4_parallel_flow.json")

    def runner():
        run_traffic_flow(h1, h4.IP(), duration, bandwidth, pstreams,
                         port=port, json_logfile=log_file)

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    FLOW_THREADS.append(t)

    info(f"*** PRIMARY FLOW h1->h4 duration={duration:.1f} bw={format_bandwidth_mbps(bandwidth)} P={pstreams}\n")

# -------------------- Background Traffic Loop --------------------

def traffic_generation_loop(net, flow_interval_sec=1.5):
    start_time = time.time()
    hosts = net.hosts

    info("*** Starting background traffic...\n")

    while not STOP_FLAG.is_set():
        if time.time() - start_time >= MASTER_RUN_SECONDS:
            info("*** Time limit reached\n")
            STOP_FLAG.set()
            break

        if len(hosts) < 2:
            time.sleep(flow_interval_sec)
            continue

        src_host = random.choice(hosts)
        dst = random.choice([h for h in hosts if h.IP() != src_host.IP()])
        dst_ip = dst.IP()

        ftype = choose_flow_type()
        duration, bandwidth, pstreams = sample_flow_params(ftype)

        remaining = MASTER_RUN_SECONDS - (time.time() - start_time)
        duration = min(duration, remaining)

        json_log = None
        if ftype == 'elephant':
            json_log = os.path.join(
                TRAFFIC_LOG_DIR,
                f"bg_elephant_{src_host.name}_to_{dst.name}_{int(time.time())}.json"
            )

        t = threading.Thread(
            target=run_traffic_flow,
            args=(src_host, dst_ip, duration, bandwidth, pstreams),
            kwargs={'json_logfile': json_log},
            daemon=True
        )
        t.start()
        FLOW_THREADS.append(t)

        time.sleep(flow_interval_sec)

    for t in FLOW_THREADS:
        try: t.join(timeout=1)
        except: pass

# -------------------- API --------------------

def generate_traffic(net, flow_interval_sec=1.5, master_run_seconds=120):
    global MASTER_RUN_SECONDS
    MASTER_RUN_SECONDS = master_run_seconds

    STOP_FLAG.clear()

    stop_all_iperf_servers(net)

    for h in net.hosts:
        start_iperf_server(h)

    start_h1_h4_parallel_flow(net)

    gen_thread = threading.Thread(
        target=traffic_generation_loop,
        args=(net, flow_interval_sec),
        daemon=True
    )
    gen_thread.start()

    def watchdog():
        time.sleep(MASTER_RUN_SECONDS)
        STOP_FLAG.set()
        info("*** Watchdog: stopping\n")

    threading.Thread(target=watchdog, daemon=True).start()

    info(f"*** Traffic generator running for {MASTER_RUN_SECONDS}s\n")
    return gen_thread

def stop_traffic(net):
    info("*** Stopping traffic...\n")
    STOP_FLAG.set()
    for t in FLOW_THREADS:
        try: t.join(timeout=1)
        except: pass
    stop_all_iperf_servers(net)
    info("*** Stopped.\n")

if __name__ == "__main__":
    info("This module is designed to be imported.\n")
