import pytest
import os
import networkx as nx
import pandas as pd
from quantum_routing.simulation.network import NetworkSimulator, save_dataset

def test_network_simulator_init():
    sim = NetworkSimulator(num_nodes=12, seed=42)
    assert sim.num_nodes == 12
    assert sim.seed == 42
    assert sim.graph is None

def test_generate_topology():
    sim = NetworkSimulator(num_nodes=10, seed=123)
    graph = sim.generate_topology()
    
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 10
    assert graph.number_of_edges() > 0
    
    # Check if edge attributes exist and are in valid ranges
    for u, v, data in graph.edges(data=True):
        assert 'capacity' in data
        assert 100 <= data['capacity'] <= 1000
        assert 'latency' in data
        assert 1 <= data['latency'] <= 50
        assert 'packet_loss' in data
        assert 0.0001 <= data['packet_loss'] <= 0.01
        assert 'utilization' in data
        assert data['utilization'] == 0.0

def test_generate_traffic():
    sim = NetworkSimulator(num_nodes=15, seed=42)
    time_steps = 50
    num_pairs = 5
    
    df = sim.generate_traffic(time_steps=time_steps, num_pairs=num_pairs)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == time_steps * num_pairs
    assert 'time_step' in df.columns
    assert 'source' in df.columns
    assert 'destination' in df.columns
    assert 'demand' in df.columns
    assert 'is_congestion_event' in df.columns
    
    # Ensure there are some non-negative demands
    assert (df['demand'] >= 0).all()

def test_reproducibility():
    sim1 = NetworkSimulator(num_nodes=10, seed=99)
    graph1 = sim1.generate_topology()
    df1 = sim1.generate_traffic(time_steps=10, num_pairs=3)
    
    sim2 = NetworkSimulator(num_nodes=10, seed=99)
    graph2 = sim2.generate_topology()
    df2 = sim2.generate_traffic(time_steps=10, num_pairs=3)
    
    # Compare graphs
    assert nx.is_isomorphic(graph1, graph2)
    
    # Compare edge capacities to ensure exact same attributes
    edges1 = sorted(graph1.edges(data=True))
    edges2 = sorted(graph2.edges(data=True))
    for e1, e2 in zip(edges1, edges2):
        assert e1[0] == e2[0] and e1[1] == e2[1]
        assert e1[2]['capacity'] == e2[2]['capacity']
        
    # Compare traffic df
    pd.testing.assert_frame_equal(df1, df2)

def test_save_dataset(tmp_path):
    sim = NetworkSimulator(num_nodes=10, seed=42)
    graph = sim.generate_topology()
    df = sim.generate_traffic(time_steps=5, num_pairs=2)
    
    save_dir = tmp_path / "data"
    save_dataset(graph, df, str(save_dir))
    
    assert (save_dir / "topology.json").exists()
    assert (save_dir / "traffic.csv").exists()
