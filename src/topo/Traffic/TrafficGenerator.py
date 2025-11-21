import random
import json
import os
import yaml

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EXPERIMENT_DURATION = 120  # Seconds
RANDOM_SEED = 42           # Change this to vary the "Scenario"
OUTPUT_FILE = "traffic_schedule.json"
NETWORK_CONFIG = os.environ.get("NETWORK_CONFIG_FILE", "network_config.yaml")

# Flow Probabilities
FLOW_TYPES_PROBS = {
    'mice': 0.4,
    'interactive': 0.3,
    'video': 0.2,
    'elephant': 0.1
}

def load_hosts():
    if not os.path.exists(NETWORK_CONFIG):
        print(f"Error: {NETWORK_CONFIG} not found.")
        return None
    with open(NETWORK_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    return [h['name'] for h in config['hosts']]

def get_flow_params(flow_type, start_time):
    """Returns params strictly adhering to your constraints."""
    remaining = EXPERIMENT_DURATION - start_time - 2 # 2s buffer
    if remaining < 1: return None

    params = {'type': flow_type}
    
    if flow_type == 'elephant':
        # 20s to 140s
        dur = random.randint(20, 140)
        bw = random.randint(5, 5000) # 5Mbps - 5Gbps
        params['b'] = f"{bw}M"
        params['P'] = random.randint(4, 10)
        
    elif flow_type == 'video':
        # 10s to 30s
        dur = random.randint(10, 30)
        bw = random.randint(10, 100)
        params['b'] = f"{bw}M"
        params['P'] = random.randint(2, 4)
        
    elif flow_type == 'interactive':
        # 5s to 10s
        dur = random.randint(5, 10)
        params['b'] = "100K"
        params['P'] = 1
        
    else: # mice
        # 0.1s to 1s
        dur = random.uniform(0.1, 1.0)
        params['b'] = "200K"
        params['P'] = 1

    # CLAMP DURATION to strictly fit in experiment time
    # If the natural duration is longer than remaining time, cut it.
    # This prevents the "Empty Result" error by ensuring flows finish naturally.
    final_dur = min(dur, remaining)
    
    # If clamping makes it too short for the type (e.g. elephant < 5s), skip it
    if flow_type == 'elephant' and final_dur < 5: return None
    
    params['t'] = final_dur
    params['t_str'] = f"{final_dur:.2f}"
    return params

def generate():
    random.seed(RANDOM_SEED)
    hosts = load_hosts()
    if not hosts: return

    schedule = {
        "metadata": {
            "duration": EXPERIMENT_DURATION,
            "seed": RANDOM_SEED,
            "primary_flow": None
        },
        "flows": []
    }

    # 1. Select Primary RL Target (Different Leaves)
    # We just pick first and last host for simplicity as "Primary" candidates
    # ideally logic should check leaf connectivity like before, but for schedule gen
    # we'll assume h1 and h_last are good candidates.
    p_src, p_dst = hosts[0], hosts[-1]
    
    # Primary Flow: Runs full duration
    schedule["flows"].append({
        "id": "primary_target",
        "src": p_src,
        "dst": p_dst,
        "start_time": 1.0,
        "duration": EXPERIMENT_DURATION + 5, # Let it run past end
        "bandwidth": "50M",
        "streams": 4,
        "type": "primary"
    })
    schedule["metadata"]["primary_flow"] = {"src": p_src, "dst": p_dst}

    # 2. Generate Background Traffic
    # We simulate a timeline
    current_time = 1.0
    flow_id = 0
    
    while current_time < EXPERIMENT_DURATION - 5:
        # Pick random pair
        src = random.choice(hosts)
        dst = random.choice(hosts)
        
        if src == dst: continue
        if src == p_src and dst == p_dst: continue # Don't overlap primary exactly
        
        # Pick Type
        ftype = random.choices(
            list(FLOW_TYPES_PROBS.keys()), 
            weights=list(FLOW_TYPES_PROBS.values())
        )[0]
        
        params = get_flow_params(ftype, current_time)
        if not params: 
            current_time += 0.5
            continue

        schedule["flows"].append({
            "id": f"{ftype}_{flow_id}",
            "src": src,
            "dst": dst,
            "start_time": round(current_time, 2),
            "duration": params['t_str'],
            "bandwidth": params['b'],
            "streams": params['P'],
            "type": ftype
        })
        
        flow_id += 1
        
        # Arrival Rate
        if ftype == 'elephant':
            current_time += random.uniform(1.5, 4.0)
        else:
            current_time += random.uniform(0.1, 0.8)

    # Sort by start time for the replayer
    schedule["flows"].sort(key=lambda x: x["start_time"])

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(schedule, f, indent=4)
    
    print(f"*** Schedule generated: {len(schedule['flows'])} flows.")
    print(f"*** Primary Target: {p_src} -> {p_dst}")
    print(f"*** Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate()