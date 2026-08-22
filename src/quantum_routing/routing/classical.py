import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import os

class ClassicalRouter:
    """
    Classical network routing baseline algorithms.
    Supports Dijkstra shortest-path and custom congestion-aware routing.
    """

    def __init__(self, network_graph: nx.Graph, weights: Dict[str, float] = None):
        """
        Initialize the classical router.
        
        Args:
            network_graph (nx.Graph): The network topology graph.
            weights (Dict[str, float], optional): Weights for the cost function.
                Expected keys: 'latency', 'utilization', 'packet_loss', 'congestion_penalty'.
                Defaults to emphasizing latency (standard Dijkstra).
        """
        self.graph = network_graph.copy()
        
        # Default weights (1.0 for latency gives standard shortest-path based on latency)
        self.weights = {
            'latency': 1.0,
            'utilization': 0.0,
            'packet_loss': 0.0,
            'congestion_penalty': 0.0
        }
        
        if weights:
            self.weights.update(weights)

    def _calculate_edge_cost(self, u: int, v: int, edge_data: dict) -> float:
        """
        Calculates the composite cost for a single edge based on the configured weights.
        """
        latency = edge_data.get('latency', 1.0)
        utilization = edge_data.get('utilization', 0.0)
        packet_loss = edge_data.get('packet_loss', 0.0)
        
        # If utilization is very high, apply congestion penalty
        is_congested = 1.0 if utilization > 0.8 else 0.0
        
        cost = (
            self.weights['latency'] * latency +
            self.weights['utilization'] * utilization +
            self.weights['packet_loss'] * packet_loss +
            self.weights['congestion_penalty'] * is_congested
        )
        return max(cost, 1e-6) # Ensure non-zero positive cost

    def _update_edge_costs(self):
        """
        Updates the 'routing_cost' attribute on all edges based on current state.
        """
        for u, v, data in self.graph.edges(data=True):
            data['routing_cost'] = self._calculate_edge_cost(u, v, data)

    def calculate_path_metrics(self, path: List[int]) -> Dict[str, float]:
        """
        Calculate the total latency, max utilization, and total packet loss for a given path.
        """
        total_latency = 0.0
        max_util = 0.0
        total_loss = 0.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            data = self.graph[u][v]
            
            total_latency += data.get('latency', 0.0)
            max_util = max(max_util, data.get('utilization', 0.0))
            
            # Loss accumulates probabilistically: 1 - prod(1 - loss_i)
            # For simplicity in metrics, we can just sum them or do probabilistic sum.
            # We'll use sum for simplicity as a metric, or proper probability:
            loss = data.get('packet_loss', 0.0)
            total_loss = 1.0 - ((1.0 - total_loss) * (1.0 - loss))
            
        return {
            'total_latency': total_latency,
            'max_utilization': max_util,
            'total_packet_loss': total_loss
        }

    def route_request(self, source: int, target: int, demand: float = 0.0, k_candidates: int = 3) -> Tuple[List[int], float, Dict[str, float]]:
        """
        Routes a single request by finding the minimum cost path.
        Optionally evaluates top K shortest paths and picks the best (useful if metrics change dynamically).
        
        Args:
            source (int): Source node.
            target (int): Target node.
            demand (float): Traffic demand to add to the selected path.
            k_candidates (int): Number of candidate paths to evaluate.
            
        Returns:
            Tuple containing:
            - Selected path (List[int])
            - Total routing cost of the path (float)
            - Dictionary of path metrics (latency, utilization, loss)
        """
        self._update_edge_costs()
        
        try:
            # Use k-shortest paths based on the routing_cost
            candidate_paths = list(nx.shortest_simple_paths(self.graph, source, target, weight='routing_cost'))
            candidate_paths = candidate_paths[:k_candidates]
        except nx.NetworkXNoPath:
            return [], float('inf'), {}

        best_path = None
        best_cost = float('inf')
        
        for path in candidate_paths:
            # Calculate total cost of this path
            path_cost = sum(self.graph[path[i]][path[i+1]]['routing_cost'] for i in range(len(path) - 1))
            if path_cost < best_cost:
                best_cost = path_cost
                best_path = path

        if best_path and demand > 0.0:
            self._apply_demand_to_path(best_path, demand)
            
        metrics = self.calculate_path_metrics(best_path) if best_path else {}
        return best_path, best_cost, metrics

    def _apply_demand_to_path(self, path: List[int], demand: float):
        """
        Updates the utilization of edges along the selected path based on the demand.
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            capacity = self.graph[u][v].get('capacity', 100.0)
            current_util = self.graph[u][v].get('utilization', 0.0)
            
            # utilization is a fraction (0.0 to 1.0)
            added_util = demand / capacity
            self.graph[u][v]['utilization'] = min(1.0, current_util + added_util)

    def route_batch(self, requests: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Route a batch of requests sequentially, updating network state after each.
        
        Args:
            requests: List of dicts, each with 'source', 'target', and 'demand'.
            
        Returns:
            pd.DataFrame of routing results.
        """
        results = []
        for req in requests:
            src = req['source']
            dst = req['target']
            demand = req.get('demand', 0.0)
            
            path, cost, metrics = self.route_request(src, dst, demand)
            
            res = {
                'source': src,
                'target': dst,
                'demand': demand,
                'selected_path': str(path),
                'route_cost': cost,
                'total_latency': metrics.get('total_latency', float('inf')),
                'max_utilization': metrics.get('max_utilization', float('inf')),
                'total_packet_loss': metrics.get('total_packet_loss', float('inf'))
            }
            results.append(res)
            
        return pd.DataFrame(results)
        
    def save_results(self, df: pd.DataFrame, output_dir: str = 'results', filename: str = 'classical_routing.csv'):
        """
        Save the routing results to a CSV file.
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath
