import networkx as nx
import numpy as np
import pandas as pd
import json
import os
import argparse

class NetworkSimulator:
    """
    Handles the simulation of network topology and traffic generation.
    """

    def __init__(self, num_nodes: int = 15, seed: int = 42):
        """
        Initialize the network simulator.
        
        Args:
            num_nodes (int): Number of nodes in the network.
            seed (int): Random seed for reproducibility.
        """
        self.num_nodes = num_nodes
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.graph = None

    def generate_topology(self) -> nx.Graph:
        """
        Generates a realistic simulated network topology using Barabasi-Albert model.
        
        Returns:
            nx.Graph: The generated network graph with link properties.
        """
        # Barabasi-Albert for a realistic hub-and-spoke-like topology
        # m is the number of edges to attach from a new node to existing nodes
        m = max(2, self.num_nodes // 5)
        self.graph = nx.barabasi_albert_graph(self.num_nodes, m, seed=self.seed)
        
        # Add realistic properties to each link
        for u, v in self.graph.edges():
            self.graph[u][v]['capacity'] = float(self.rng.uniform(100, 1000)) # Mbps or Gbps
            self.graph[u][v]['latency'] = float(self.rng.uniform(1, 50)) # ms
            self.graph[u][v]['packet_loss'] = float(self.rng.uniform(0.0001, 0.01)) # probability
            self.graph[u][v]['utilization'] = 0.0 # initially 0
            
        return self.graph

    def generate_traffic(self, time_steps: int = 100, num_pairs: int = 5) -> pd.DataFrame:
        """
        Generates traffic demand matrices over time with normal, peak, and congestion events.
        
        Args:
            time_steps (int): The number of time steps to simulate.
            num_pairs (int): Number of active source-destination pairs.
            
        Returns:
            pd.DataFrame: Simulated traffic data over time.
        """
        if self.graph is None:
            self.generate_topology()
            
        nodes = list(self.graph.nodes())
        
        # Select random src-dst pairs
        pairs = []
        while len(pairs) < num_pairs:
            src, dst = self.rng.choice(nodes, 2, replace=False)
            if (src, dst) not in pairs and (dst, src) not in pairs:
                pairs.append((src, dst))
                
        records = []
        
        for step in range(time_steps):
            # Baseline diurnal pattern (sine wave)
            # Normal to peak traffic variation
            time_factor = np.sin(step / time_steps * 2 * np.pi) 
            # scale time factor to [0.5, 1.5] for normal/peak
            diurnal_multiplier = 1.0 + (time_factor * 0.5) 
            
            for src, dst in pairs:
                # Base demand for this pair
                base_demand = self.rng.uniform(10, 50)
                demand = base_demand * diurnal_multiplier
                
                # Add noise
                noise = self.rng.normal(0, 2)
                demand = max(0, demand + noise)
                
                # Random congestion event (e.g., 2% chance per step per pair)
                is_congestion = self.rng.random() < 0.02
                if is_congestion:
                    # 3x to 5x spike in traffic
                    spike_multiplier = self.rng.uniform(3, 5)
                    demand *= spike_multiplier
                    
                records.append({
                    'time_step': step,
                    'source': int(src),
                    'destination': int(dst),
                    'demand': float(demand),
                    'is_congestion_event': bool(is_congestion)
                })
                
        return pd.DataFrame(records)

    def get_link_capacities(self) -> dict:
        """
        Returns the capacity of each link in the network.
        
        Returns:
            dict: Mapping of edge tuples to capacity values.
        """
        if self.graph is None:
            self.generate_topology()
        return {(u, v): d['capacity'] for u, v, d in self.graph.edges(data=True)}

def save_dataset(graph: nx.Graph, traffic_df: pd.DataFrame, output_dir: str):
    """
    Saves the generated topology and traffic data to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save topology
    topology_data = nx.node_link_data(graph)
    topology_file = os.path.join(output_dir, 'topology.json')
    with open(topology_file, 'w') as f:
        json.dump(topology_data, f, indent=2)
        
    # Save traffic
    traffic_file = os.path.join(output_dir, 'traffic.csv')
    traffic_df.to_csv(traffic_file, index=False)
    
    print(f"Dataset successfully saved to {output_dir}/")
    print(f"  - Topology: {topology_file}")
    print(f"  - Traffic: {traffic_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Network Simulation Dataset")
    parser.add_argument("--nodes", type=int, default=15, help="Number of routers (nodes)")
    parser.add_argument("--timesteps", type=int, default=200, help="Number of time steps for traffic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for dataset")
    args = parser.parse_args()

    print(f"Generating network topology with {args.nodes} nodes...")
    sim = NetworkSimulator(num_nodes=args.nodes, seed=args.seed)
    topology = sim.generate_topology()
    
    print(f"Generating time-varying traffic for {args.timesteps} steps...")
    traffic_df = sim.generate_traffic(time_steps=args.timesteps, num_pairs=args.nodes//2)
    
    save_dataset(topology, traffic_df, args.output_dir)

if __name__ == '__main__':
    main()
