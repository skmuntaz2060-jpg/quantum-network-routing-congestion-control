import argparse
import networkx as nx
import pandas as pd
import numpy as np
import json
import os
from quantum_routing.routing.classical import ClassicalRouter

def run_classical_routing(network_file: str, traffic_file: str, output_dir: str):
    print(f"Loading network from {network_file}...")
    
    # Load network (we expect a node-link json format for graph with attributes)
    try:
        with open(network_file, 'r') as f:
            data = json.load(f)
            G = nx.node_link_graph(data)
    except FileNotFoundError:
        print(f"Network file {network_file} not found. Please run the simulation step first.")
        return
        
    print(f"Loading traffic requests from {traffic_file}...")
    try:
        df_traffic = pd.read_csv(traffic_file)
    except FileNotFoundError:
        print(f"Traffic file {traffic_file} not found. Please run the simulation step first.")
        return
        
    # Take a sample of traffic requests to route (e.g., from the first time step)
    # The traffic file has 'time_step', 'source', 'destination', 'demand'
    step_0_traffic = df_traffic[df_traffic['time_step'] == 0]
    
    requests = []
    for _, row in step_0_traffic.iterrows():
        requests.append({
            'source': int(row['source']),
            'target': int(row['destination']),
            'demand': float(row['demand'])
        })
        
    print(f"Routing {len(requests)} requests using Classical Congestion-Aware Routing...")
    
    # We will use congestion aware routing
    weights = {
        'latency': 1.0,
        'utilization': 50.0,
        'packet_loss': 100.0,
        'congestion_penalty': 500.0
    }
    
    router = ClassicalRouter(G, weights=weights)
    df_results = router.route_batch(requests)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'classical_routing.csv')
    df_results.to_csv(out_path, index=False)
    
    print(f"Routing complete. Results saved to {out_path}")
    print("\nSample Results:")
    print(df_results[['source', 'target', 'demand', 'selected_path', 'route_cost']].head())
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classical routing benchmark")
    parser.add_argument("--network", type=str, default="data/network.json", help="Path to network JSON")
    parser.add_argument("--traffic", type=str, default="data/traffic.csv", help="Path to traffic CSV")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    run_classical_routing(args.network, args.traffic, args.output_dir)
