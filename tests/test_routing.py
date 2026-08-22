import pytest
import networkx as nx
import pandas as pd
from quantum_routing.routing.classical import ClassicalRouter
from quantum_routing.routing.quantum import QuantumRouter

@pytest.fixture
def dummy_graph():
    G = nx.Graph()
    # Simple diamond topology
    # 0 --- 1 --- 3
    # |           |
    # +---- 2 ----+
    
    # Path 0-1-3: Low latency, but 1-3 is highly utilized (congested)
    G.add_edge(0, 1, capacity=100.0, latency=10.0, utilization=0.1, packet_loss=0.01)
    G.add_edge(1, 3, capacity=100.0, latency=10.0, utilization=0.9, packet_loss=0.05)
    
    # Path 0-2-3: Higher latency, but uncongested
    G.add_edge(0, 2, capacity=100.0, latency=20.0, utilization=0.2, packet_loss=0.01)
    G.add_edge(2, 3, capacity=100.0, latency=20.0, utilization=0.2, packet_loss=0.01)
    
    return G

# --- Classical Routing Tests ---
def test_router_initialization(dummy_graph):
    router = ClassicalRouter(dummy_graph)
    assert router.weights['latency'] == 1.0
    assert router.weights['utilization'] == 0.0
    
    custom_weights = {'latency': 0.5, 'utilization': 2.0}
    router2 = ClassicalRouter(dummy_graph, weights=custom_weights)
    assert router2.weights['latency'] == 0.5
    assert router2.weights['utilization'] == 2.0
    assert router2.weights['packet_loss'] == 0.0

def test_edge_cost_calculation(dummy_graph):
    weights = {
        'latency': 1.0,
        'utilization': 100.0,
        'packet_loss': 1000.0,
        'congestion_penalty': 500.0
    }
    router = ClassicalRouter(dummy_graph, weights=weights)
    
    # Check edge 1-3 (congested: util 0.9, loss 0.05)
    data_1_3 = dummy_graph[1][3]
    cost_1_3 = router._calculate_edge_cost(1, 3, data_1_3)
    assert cost_1_3 == 650.0

    # Check edge 2-3 (uncongested: util 0.2, loss 0.01)
    data_2_3 = dummy_graph[2][3]
    cost_2_3 = router._calculate_edge_cost(2, 3, data_2_3)
    assert cost_2_3 == 50.0

def test_dijkstra_shortest_path(dummy_graph):
    router = ClassicalRouter(dummy_graph, weights={'latency': 1.0, 'utilization': 0.0, 'congestion_penalty': 0.0})
    path, cost, metrics = router.route_request(source=0, target=3)
    assert path == [0, 1, 3]
    assert cost == 20.0

def test_congestion_aware_routing(dummy_graph):
    weights = {
        'latency': 1.0,
        'utilization': 0.0,
        'congestion_penalty': 100.0
    }
    router = ClassicalRouter(dummy_graph, weights=weights)
    path, cost, metrics = router.route_request(source=0, target=3)
    assert path == [0, 2, 3]
    assert cost == 40.0

def test_route_batch_and_update_utilization(dummy_graph):
    router = ClassicalRouter(dummy_graph, weights={'latency': 1.0, 'utilization': 100.0})
    requests = [
        {'source': 0, 'target': 2, 'demand': 10.0},
        {'source': 0, 'target': 2, 'demand': 20.0}
    ]
    df_results = router.route_batch(requests)
    assert len(df_results) == 2
    assert router.graph[0][2]['utilization'] == 0.5


# --- Quantum Routing Tests ---
def test_qubo_formulation(dummy_graph):
    router = QuantumRouter(dummy_graph, penalty_lambda=100.0)
    router._update_edge_costs() # Latency=1.0, so 0-1-3=20, 0-2-3=40
    
    candidate_paths = [[0, 1, 3], [0, 2, 3]]
    qp = router._formulate_qubo(candidate_paths)
    
    # Variables should be x_0, x_1
    assert qp.get_num_vars() == 2
    
    # Linear coefficients: C_i - lambda
    # C_0 = 20, C_1 = 40. max_C = 40
    # lambda = max(100.0, 40*2 + 1000) = 1080.0
    # lin_0 = 20 - 1080 = -1060
    # lin_1 = 40 - 1080 = -1040
    assert qp.objective.linear.to_array()[0] == -1060.0
    assert qp.objective.linear.to_array()[1] == -1040.0

    # Quadratic coefficient: 2 * lambda = 2160
    assert qp.objective.quadratic.to_array()[0, 1] == 2160.0

def test_exact_eigensolver(dummy_graph):
    router = QuantumRouter(dummy_graph, penalty_lambda=1000.0)
    
    path, cost, metrics, meta = router.route_request_quantum(source=0, target=3, k_candidates=2, use_qaoa=False)
    
    assert path == [0, 1, 3] # Shortest path by latency
    assert cost == 20.0
    assert meta['solver'] == 'Exact'
    assert meta['feasibility_failed'] == False

def test_qaoa_solver(dummy_graph):
    router = QuantumRouter(dummy_graph, penalty_lambda=1000.0)
    
    # QAOA can be probabilistic, but for a simple 2-variable problem with lambda=1000, 
    # the minimum energy state is overwhelmingly likely to be found.
    path, cost, metrics, meta = router.route_request_quantum(source=0, target=3, k_candidates=2, use_qaoa=True, qaoa_reps=1)
    
    assert path == [0, 1, 3]
    assert cost == 20.0
    assert meta['solver'] == 'QAOA'
    assert meta['feasibility_failed'] == False
