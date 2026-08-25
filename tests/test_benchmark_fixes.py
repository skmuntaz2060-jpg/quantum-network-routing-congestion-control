import pytest
import networkx as nx
from quantum_routing.routing.classical import ClassicalRouter

def get_test_graph():
    G = nx.Graph()
    G.add_edge(0, 1, capacity=100.0, latency=10.0, packet_loss=0.01, utilization=0.0)
    G.add_edge(1, 2, capacity=100.0, latency=10.0, packet_loss=0.01, utilization=0.0)
    G.add_edge(0, 2, capacity=100.0, latency=30.0, packet_loss=0.01, utilization=0.0)
    return G

def test_utilization_exceeds_one():
    G = get_test_graph()
    router = ClassicalRouter(G)
    
    # 1. Below capacity -> no overload
    router._apply_demand_to_path([0, 1], 50.0)
    assert router.graph[0][1]['utilization'] == 0.5
    
    # 2. Exactly at capacity -> no overload
    router._apply_demand_to_path([0, 1], 50.0)
    assert router.graph[0][1]['utilization'] == 1.0
    
    # 3. Above capacity -> utilization exceeds 1.0
    router._apply_demand_to_path([0, 1], 50.0)
    assert router.graph[0][1]['utilization'] == 1.5

def test_dynamic_overload_packet_loss():
    G = get_test_graph()
    G[0][1]['utilization'] = 1.0
    
    router = ClassicalRouter(G)
    
    # At utilization 1.0, packet loss is just the base
    metrics = router.calculate_path_metrics([0, 1])
    assert metrics['packet_loss_rate'] == pytest.approx(0.01)
    
    # At utilization 1.5, packet loss increases by (1.5 - 1.0) * 0.5 = 0.25
    router.graph[0][1]['utilization'] = 1.5
    metrics = router.calculate_path_metrics([0, 1])
    assert metrics['packet_loss_rate'] == pytest.approx(0.01 + 0.25)

def test_packet_loss_cap():
    G = get_test_graph()
    G[0][1]['utilization'] = 5.0 # Extremely high load
    
    router = ClassicalRouter(G)
    metrics = router.calculate_path_metrics([0, 1])
    # Ensure it's capped at 1.0
    assert metrics['packet_loss_rate'] == 1.0
    
def test_ai_penalty_changes_edge_cost():
    G = get_test_graph()
    router = ClassicalRouter(G)
    
    base_cost = router._calculate_edge_cost(0, 1, router.graph[0][1])
    
    router.graph[0][1]['routing_cost_ai_penalty'] = 200.0
    penalized_cost = router._calculate_edge_cost(0, 1, router.graph[0][1])
    
    assert penalized_cost == base_cost + 200.0

def test_ai_penalty_changes_route_preference():
    G = get_test_graph()
    router = ClassicalRouter(G)
    
    # Without penalty, shortest path is [0, 1, 2] (cost 20 latency vs 30 latency)
    path, cost, _ = router.route_request(0, 2)
    assert path == [0, 1, 2]
    
    # Add AI penalty to edge (0, 1) making it very expensive
    router.graph[0][1]['routing_cost_ai_penalty'] = 200.0
    
    # With penalty, shortest path should switch to [0, 2]
    path, cost, _ = router.route_request(0, 2)
    assert path == [0, 2]

def test_bounded_k_shortest_paths():
    # Create a dense graph where number of simple paths is large
    G = nx.complete_graph(10)
    for u, v in G.edges():
        G[u][v]['latency'] = 1.0
        G[u][v]['utilization'] = 0.0
        G[u][v]['packet_loss'] = 0.0
        G[u][v]['capacity'] = 100.0
        
    router = ClassicalRouter(G)
    # The bounded generator should return exactly 3 paths rapidly without computing all possible simple paths
    path, cost, _ = router.route_request(0, 1, k_candidates=3)
    assert path is not None
    assert len(path) >= 2
