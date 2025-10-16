from mininet.net import Mininet
from mininet.log import setLogLevel, info
import time

setLogLevel("info")

# Set parameters
duration = 60
streams = 4
bandwidth = 100  # Kbps

# Get hosts
host_names = ["h1", "h2", "h3", "h4", "h5", "h6"]

hosts = []
net = Mininet()
for h in host_names:
    hosts.append(net.get(h))

server = hosts[3]

# Start iperf3 server on h4
info("*** Starting iperf3 server\n")
server.cmd("iperf3 -s &")

# Start iperf3 clients on h1, h2, and h3 to send traffic to h4
info("*** Starting iperf3 clients\n")
for h in hosts:
    if h is not server:
        h.cmd(
            f"iperf3 -c {server.IP()} -t {duration} -i 10 -P {streams} -b {bandwidth}K -M 1400 > /dev/null 2>&1 &"
        )

# Wait for the tests to complete
info("*** Waiting for iperf3 tests to complete\n")

# Sleep for the duration of the test
time.sleep(duration + 2)

# Print output from h4
info(server.cmd("pkill iperf3"))
