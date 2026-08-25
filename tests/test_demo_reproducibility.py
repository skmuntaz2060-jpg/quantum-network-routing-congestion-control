import pytest
import networkx as nx
import pandas as pd
from quantum_routing.simulation.network import NetworkSimulator

def test_demo_reproducibility():
    """
    Test that two NetworkSimulator runs with the same seed produce
    identical topologies and traffic.
    """
    seed = 42
    nodes = 8
    timesteps = 2
    pairs = 4
    
    # Run 1
    sim1 = NetworkSimulator(num_nodes=nodes, seed=seed)
    G1 = sim1.generate_topology()
    traffic1 = sim1.generate_traffic(time_steps=timesteps, num_pairs=pairs)
    
    # Run 2
    sim2 = NetworkSimulator(num_nodes=nodes, seed=seed)
    G2 = sim2.generate_topology()
    traffic2 = sim2.generate_traffic(time_steps=timesteps, num_pairs=pairs)
    
    # Assert topologies are identical
    assert list(G1.nodes()) == list(G2.nodes())
    assert list(G1.edges()) == list(G2.edges())
    
    for u, v, data1 in G1.edges(data=True):
        data2 = G2[u][v]
        assert data1['capacity'] == data2['capacity']
        assert data1['latency'] == data2['latency']
        assert data1['packet_loss'] == data2['packet_loss']
        
    # Assert traffic is identical
    pd.testing.assert_frame_equal(traffic1, traffic2)
