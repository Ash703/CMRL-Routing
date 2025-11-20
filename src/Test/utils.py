import yaml
import os

class Network:
    """
    Utility class to parse network configuration and provide structured access 
    to topology elements based on the 'type' field in the YAML.
    """
    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Network configuration file not found: {config_file}")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        self.switches_by_name = {sw["name"]: sw for sw in config.get("switches", [])}
        self.switches_by_id = {sw["id"]: sw for sw in config.get("switches", []) if "id" in sw}
        
        self.leaves = []
        self.spines = []

        # Explicitly categorize based on 'type' field in YAML
        for sw in config.get("switches", []):
            dpid = sw.get("id")
            sw_type = sw.get("type", "").lower()
            
            if sw_type == "leaf":
                self.leaves.append(dpid)
            elif sw_type == "spine":
                self.spines.append(dpid)
            else:
                # Fallback if type is missing: assume high IDs are leaves (switch dependent)
                pass

        # Sort for consistent indexing
        self.leaves.sort()
        self.spines.sort()

        # Recreate links structure: self.links[(src, dst)] = {'port': p}
        self.links = {}
        for link in config.get("links", []):
            src_name = link["source"]
            dst_name = link["target"]
            
            src_sw = self.switches_by_name.get(src_name)
            dst_sw = self.switches_by_name.get(dst_name)
            
            if src_sw and dst_sw:
                src_dpid = src_sw["id"]
                dst_dpid = dst_sw["id"]
                
                # Forward link
                self.links[(src_dpid, dst_dpid)] = {
                    "port": link["source_port"]
                }
                # Reverse link (assuming symmetric topology)
                self.links[(dst_dpid, src_dpid)] = {
                    "port": link["target_port"]
                }
                
        print(f"Network initialized from {config_file}")
        print(f"Leaves: {self.leaves}")
        print(f"Spines: {self.spines}")