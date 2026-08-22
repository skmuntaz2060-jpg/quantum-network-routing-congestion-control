import argparse
import networkx as nx
import pandas as pd
import json
import os
from quantum_routing.routing.quantum import QuantumRouter

def run_quantum_routing(network_file: str, traffic_file: str, output_dir: str):
    print(f"Loading network from {network_file}...")
    
    try:
        with open(network_file, 'r') as f:
            data = json.load(f)
            G = nx.node_link_graph(data)
    except FileNotFoundError:
        print(f"Network file {network_file} not found.")
        return
        
    print(f"Loading traffic requests from {traffic_file}...")
    try:
        df_traffic = pd.read_csv(traffic_file)
    except FileNotFoundError:
        print(f"Traffic file {traffic_file} not found.")
        return
        
    # Take a sample of traffic requests to route (e.g., from the first time step)
    step_0_traffic = df_traffic[df_traffic['time_step'] == 0]
    
    requests = []
    for _, row in step_0_traffic.iterrows():
        requests.append({
            'source': int(row['source']),
            'target': int(row['destination']),
            'demand': float(row['demand'])
        })
        
    print(f"Routing {len(requests)} requests using QAOA Quantum Routing...")
    
    weights = {
        'latency': 1.0,
        'utilization': 50.0,
        'packet_loss': 100.0,
        'congestion_penalty': 500.0
    }
    
    # We use penalty_lambda large enough to strictly enforce feasibility
    router = QuantumRouter(G, weights=weights, penalty_lambda=1000.0)
    
    # We solve using QAOA
    df_results = router.route_batch_quantum(requests, use_qaoa=True, qaoa_reps=1)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'quantum_routing.csv')
    df_results.to_csv(out_path, index=False)
    
    print(f"Routing complete. Results saved to {out_path}")
    print("\nSample Results:")
    print(df_results[['source', 'target', 'demand', 'selected_path', 'route_cost', 'solver', 'qubo_fval', 'feasibility_failed']].head())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QAOA quantum routing benchmark")
    parser.add_argument("--network", type=str, default="data/topology.json", help="Path to network JSON")
    parser.add_argument("--traffic", type=str, default="data/traffic.csv", help="Path to traffic CSV")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    run_quantum_routing(args.network, args.traffic, args.output_dir)
