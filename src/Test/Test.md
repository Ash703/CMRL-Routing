# Navigate to your project folder
cd sdn_project

# Clean up old logs before a new run
rm -rf traffic_logs/* checkpoints2/*.csv

# Run the Controller
# NOTE: ryu-manager might need to be run as root if accessing specific ports, 
# but usually user-level is fine if Mininet connects to localhost:6633
ryu-manager rldc_controller_integrated.py

#### Terminal 2: The Network (Body)
This runs the topology and the traffic generator.

```bash
# Navigate to your project folder
cd sdn_project

# Clean up Mininet (important to prevent errors from previous runs)
sudo mn -c

# Run the Topology script
# sudo is required for Mininet
sudo python3 mn_spineleaf_topo.py network_config.yaml

### 4. How to "Play" the Comparison

To generate the data for your report:

1.  **Run RL:**
    * Edit `rldc_controller_integrated.py`. Set `ROUTING_ALGO = "RL"`.
    * Run Terminal 1 & 2. Let it run for 5 minutes.
    * Type `exit` in Mininet CLI to stop.
    * Copy `checkpoints2/rl_reward_log.csv` and `traffic_logs/h1_h4_parallel_flow.json` to a folder named `results_rl/`.

2.  **Run ECMP:**
    * Edit `rldc_controller_integrated.py`. Set `ROUTING_ALGO = "ECMP"`.
    * Run Terminal 1 & 2. Let it run for 5 minutes.
    * Stop. Copy logs to `results_ecmp/`.

3.  **Run Greedy:**
    * Edit `rldc_controller_integrated.py`. Set `ROUTING_ALGO = "GREEDY"`.
    * Run, Wait, Stop. Copy logs to `results_greedy/`.
