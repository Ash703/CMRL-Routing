from mininet.net import Mininet
from mininet.log import setLogLevel, info
import random, time

setLogLevel("info")

# Duration of total experiment
EXPERIMENT_TIME = 60

# Flow types with parameters
FLOW_TYPES = {
    'mice':     {'t': 0.2,  'b': 200,   'P': 1},
    'elephant': {'t': 30,   'b': 5000,  'P': 4},
    'video':    {'t': 15,   'b': 2000,  'P': 2},
    'interactive': {'t': 5, 'b': 100,   'P': 1}
}

def generate_traffic(net):
    hosts = net.hosts
    info("*** Starting contextual traffic flows\n")

    for _ in range(10):  # number of flows
        src, dst = random.sample(hosts, 2)
        flow_type = random.choice(list(FLOW_TYPES.keys()))
        params = FLOW_TYPES[flow_type]

        cmd = f"iperf3 -c {dst.IP()} -t {params['t']} -b {params['b']}K -P {params['P']} -i 5 > /dev/null 2>&1 &"
        info(f"{src.name} -> {dst.name} ({flow_type} flow)\n")
        src.cmd(cmd)
        time.sleep(random.uniform(0.5, 2))  # random start times

if __name__ == '__main__':
    net = Mininet()  # Ideally, import your leaf-spine topology here
    net.start()
    generate_traffic(net)
    info("*** Waiting for experiment to finish...\n")
    time.sleep(EXPERIMENT_TIME)
    info("*** Stopping iperf3\n")
    for h in net.hosts:
        h.cmd("pkill iperf3")
    net.stop()