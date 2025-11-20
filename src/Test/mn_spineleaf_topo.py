import os
import sys
import yaml
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.node import OVSSwitch
from mininet.node import Host
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI

# IMPORT THE TRAFFIC GENERATOR
from traffic_generator import generate_traffic, stop_traffic

class MyOVSSwitch(OVSSwitch):
    def __init__(self, *args, **kwargs):
        kwargs["protocols"] = "OpenFlow13"
        super().__init__(*args, **kwargs)


def create_mininet_network(config_file, ctrl="127.0.0.1"):
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    net = Mininet(controller=RemoteController, link=TCLink, switch=MyOVSSwitch)

    info("*** Adding controller\n")
    c0 = net.addController("ctrl", controller=RemoteController, ip=ctrl, port=6633)

    switches = {}
    hosts = {}

    info("*** Adding switches\n")
    for switch_config in config["switches"]:
        # Using 'id' as name if name is not provided, or mapping name->id if needed
        # The controller expects dpid to match. Mininet usually assigns dpid based on switch name order
        # unless dpid is explicitly set.
        sw_name = switch_config["name"]
        dpid = str(switch_config["id"]) # Set explicit dpid from YAML
        switches[sw_name] = net.addSwitch(sw_name, dpid=dpid)

    info("*** Adding inter-switch links\n")
    for link_config in config["links"]:
        source_switch = switches[link_config["source"]]
        target_switch = switches[link_config["target"]]
        net.addLink(
            source_switch,
            target_switch,
            port1=link_config["source_port"],
            port2=link_config["target_port"],
            bw=1000 # Assuming 1Gbps links for Spine-Leaf
        )

    info("*** Adding hosts\n")
    for host_config in config["hosts"]:
        host = net.addHost(
            host_config["name"],
            ip=host_config["ip"],
            mac=host_config["mac"],
            defaultRoute=host_config["default_route"],
        )
        hosts[host_config["name"]] = host
        switch = switches[host_config["connected_to"]]
        port = host_config["port"]
        net.addLink(host, switch, port1=port)

    info("*** Starting network\n")
    net.build()
    info("*** Starting controllers\n")
    c0.start()
    info("*** Starting switches\n")
    for switch in switches.values():
        switch.start([c0])

    # --- START TRAFFIC GENERATION ---
    info("*** Starting traffic flows (Background + RL Target)\n")
    # Use defaults: interval=1.5s, duration 5-15s
    generate_traffic(net, flow_interval_sec=1.5) 
    # --------------------------------

    info("*** Dropping to CLI\n")
    CLI(net)

    # --- STOP TRAFFIC GENERATION ---
    stop_traffic(net)
    # -------------------------------

    info("*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    ctrl_address = os.getenv("SDN_CONTROLLER", "127.0.0.1")

    if len(sys.argv) < 2:
        print("Usage: sudo python3 mn_spineleaf_topo.py <netconfig_file> [controller_address]")
        sys.exit(1)

    config_file = sys.argv[1]

    if len(sys.argv) > 2:
        ctrl_address = sys.argv[2]

    setLogLevel("info")
    create_mininet_network(config_file, ctrl_address)