import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_vrp_xml(data_path):
    tree = ET.parse(data_path)
    root = tree.getroot()

    nodes = []
    for node in root.find('network').find('nodes').findall('node'):
        node_id = int(node.get('id'))
        x = float(node.find('cx').text)
        y = float(node.find('cy').text)
        node_type = int(node.get('type'))
        nodes.append({"id": node_id, "cx": x, "cy": y, "type": node_type})

    nodes_df = pd.DataFrame(nodes).sort_values("id").reset_index(drop=True)

    demand_map = {node_id: 0.0 for node_id in nodes_df["id"]}
    for req in root.find("requests"):
        node_id = int(req.get("node"))
        demand_map[node_id] = float(req.find("quantity").text)

    demands_df = pd.DataFrame({
        "id": list(demand_map.keys()),
        "demand": list(demand_map.values()),
    }).sort_values("id").reset_index(drop=True)

    decimals = int(root.find("network").find("decimals").text)
    coords = nodes_df[["cx", "cy"]].to_numpy()
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    dist = np.round(dist, decimals)
    dist_df = pd.DataFrame(dist, index=nodes_df["id"], columns=nodes_df["id"])

    capacity = float(root.find("fleet").find("vehicle_profile").find("capacity").text)

    return nodes_df, demands_df, dist_df, capacity


def data_inputs_to_cvrp(demands_df, dist_df):
    
    cost_matrix = dist_df.values.tolist()
    demand = demands_df.sort_values("id")["demand"].tolist()
    return cost_matrix, demand


def load_all_instances(data_dir):
    
    instances = {}
    for data_path in sorted(Path(data_dir).glob("*.xml")):
        name = data_path.stem
        nodes_df, demands_df, dist_df, capacity = parse_vrp_xml(data_path)
        instances[name] = {
            "nodes": nodes_df,
            "demands": demands_df,
            "dist": dist_df,
            "capacity": capacity,
        }
    return instances


if __name__ == "__main__":
    instances = load_all_instances("data/christofides-et-al-1979-set-m")
    for name, inst in instances.items():
        n_customers = (inst["nodes"]["type"] == 1).sum()
        print(f"{name}: {n_customers} customers, capacity={inst['capacity']}")