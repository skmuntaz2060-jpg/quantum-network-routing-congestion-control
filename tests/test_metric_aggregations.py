import pytest
import pandas as pd
import numpy as np
import networkx as nx
from quantum_routing.main import calculate_network_metrics

def get_dummy_data():
    # Successful routes
    df_results = pd.DataFrame([
        {'selected_path': '[0, 1]', 'demand': 100.0, 'packet_loss_rate': 0.1, 'total_latency': 10.0, 'max_utilization': 0.5},
        {'selected_path': '[1, 2]', 'demand': 200.0, 'packet_loss_rate': 0.5, 'total_latency': 20.0, 'max_utilization': 0.9},
        {'selected_path': '[0, 2]', 'demand': 50.0, 'packet_loss_rate': 0.0, 'total_latency': 5.0, 'max_utilization': 0.2}
    ])
    
    # Mock final graph (utilization decayed)
    G = nx.Graph()
    G.add_edge(0, 1, utilization=0.1)
    G.add_edge(1, 2, utilization=0.45)
    
    return df_results, G

def test_weighted_packet_loss():
    df, G = get_dummy_data()
    metrics = calculate_network_metrics(df, G, [0.1, 0.1, 0.1], [])
    
    # Expected: (100 * 0.1 + 200 * 0.5 + 50 * 0.0) / 350
    # = (10 + 100 + 0) / 350 = 110 / 350 = 0.3142857...
    expected_loss = 110.0 / 350.0
    
    assert metrics['packet_loss_rate'] == pytest.approx(expected_loss)
    assert 0 <= metrics['packet_loss_rate'] <= 1

def test_packet_loss_zero_demand():
    df = pd.DataFrame([
        {'selected_path': '[0, 1]', 'demand': 0.0, 'packet_loss_rate': 0.5, 'total_latency': 10.0, 'max_utilization': 0.5}
    ])
    G = nx.Graph()
    metrics = calculate_network_metrics(df, G, [], [])
    assert metrics['packet_loss_rate'] == 0.0
    assert metrics['throughput'] == 0.0

def test_congestion_rate_from_history():
    df, G = get_dummy_data()
    metrics = calculate_network_metrics(df, G, [0.1, 0.1, 0.1], [])
    
    # 1 out of 3 routes has max_utilization > 0.8 (the second route with 0.9)
    assert metrics['congestion_rate'] == pytest.approx(1.0 / 3.0)

def test_average_and_peak_utilization():
    df, G = get_dummy_data()
    metrics = calculate_network_metrics(df, G, [0.1, 0.1, 0.1], [])
    
    # average: (0.5 + 0.9 + 0.2) / 3 = 1.6 / 3 = 0.5333...
    assert metrics['average_utilization'] == pytest.approx(1.6 / 3.0)
    # peak: 0.9
    assert metrics['peak_utilization'] == 0.9

def test_load_imbalance():
    df, G = get_dummy_data()
    metrics = calculate_network_metrics(df, G, [0.1, 0.1, 0.1], [])
    
    # Expected std deviation of [0.5, 0.9, 0.2]
    expected_std = np.std([0.5, 0.9, 0.2], ddof=1)
    assert metrics['load_imbalance'] == pytest.approx(expected_std)

def test_final_graph_decay_does_not_affect_history():
    df, G = get_dummy_data()
    
    # Change final graph utilization completely
    G.edges[0, 1]['utilization'] = 0.0
    G.edges[1, 2]['utilization'] = 0.0
    
    metrics = calculate_network_metrics(df, G, [0.1, 0.1, 0.1], [])
    
    # Metrics should still use route history
    assert metrics['congestion_rate'] == pytest.approx(1.0 / 3.0)
    assert metrics['average_utilization'] == pytest.approx(1.6 / 3.0)
