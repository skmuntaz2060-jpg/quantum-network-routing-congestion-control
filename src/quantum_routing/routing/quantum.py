import time
import networkx as nx
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Tuple

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

from .classical import ClassicalRouter

class QuantumRouter(ClassicalRouter):
    """
    Quantum network routing using QAOA.
    Inherits from ClassicalRouter to reuse edge cost and path generation logic.
    """

    def __init__(self, network_graph: nx.Graph, weights: Dict[str, float] = None, penalty_lambda: float = 1000.0):
        """
        Initialize the quantum router.
        
        Args:
            network_graph: The network topology graph.
            weights: Weights for the classical edge cost function.
            penalty_lambda: Penalty weight for the exactly-one-path constraint.
        """
        super().__init__(network_graph, weights)
        self.penalty_lambda = penalty_lambda

    def _formulate_qubo(self, candidate_paths: List[List[int]]) -> QuadraticProgram:
        """
        Formulate route selection as a binary optimization problem (QUBO).
        
        Args:
            candidate_paths: A list of K candidate paths.
            
        Returns:
            qiskit_optimization.QuadraticProgram
        """
        qp = QuadraticProgram()
        K = len(candidate_paths)
        
        # Add binary variables x_i
        for i in range(K):
            qp.binary_var(name=f"x_{i}")
            
        # Compute classical cost C_i for each path
        C = []
        for path in candidate_paths:
            path_cost = sum(self.graph[path[i]][path[i+1]]['routing_cost'] for i in range(len(path) - 1))
            C.append(path_cost)
            
        # Objective: sum_i C_i x_i + lambda * (sum_i x_i - 1)^2
        # = sum_i (C_i - lambda) x_i + 2 * lambda * sum_{i < j} x_i x_j + lambda
        
        # Dynamically scale lambda to ensure constraint is strictly enforced
        # It must be greater than any individual path cost.
        max_c = max(C) if C else 0
        actual_lambda = max(self.penalty_lambda, max_c * 2.0 + 1000.0)
        
        linear = {}
        for i in range(K):
            linear[f"x_{i}"] = C[i] - actual_lambda
            
        quadratic = {}
        for i in range(K):
            for j in range(i + 1, K):
                quadratic[(f"x_{i}", f"x_{j}")] = 2.0 * actual_lambda
                
        qp.minimize(linear=linear, quadratic=quadratic, constant=actual_lambda)
        return qp

    def solve_qaoa(self, qp: QuadraticProgram, reps: int = 1) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        Solve the QUBO using QAOA.
        """
        sampler = StatevectorSampler()
        optimizer = COBYLA(maxiter=15)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
        meo = MinimumEigenOptimizer(qaoa)
        
        start_time = time.time()
        result = meo.solve(qp)
        runtime = time.time() - start_time
        
        metadata = {
            'solver': 'QAOA',
            'reps': reps,
            'runtime_sec': runtime,
            'fval': result.fval,
            'status': str(result.status)
        }
        
        return list(result.x), result.fval, metadata

    def solve_exact(self, qp: QuadraticProgram) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        Solve the QUBO exactly (for comparison/validation).
        """
        exact = NumPyMinimumEigensolver()
        meo = MinimumEigenOptimizer(exact)
        
        start_time = time.time()
        result = meo.solve(qp)
        runtime = time.time() - start_time
        
        metadata = {
            'solver': 'Exact',
            'runtime_sec': runtime,
            'fval': result.fval,
            'status': str(result.status)
        }
        
        return list(result.x), result.fval, metadata

    def route_request_quantum(self, source: int, target: int, demand: float = 0.0, k_candidates: int = 3, use_qaoa: bool = True, qaoa_reps: int = 1) -> Tuple[List[int], float, Dict[str, float], Dict[str, Any]]:
        """
        Route a request using quantum optimization over K candidate paths.
        """
        self._update_edge_costs()
        
        try:
            # Generate K shortest simple paths based on edge costs
            candidate_paths = list(nx.shortest_simple_paths(self.graph, source, target, weight='routing_cost'))
            candidate_paths = candidate_paths[:k_candidates]
        except nx.NetworkXNoPath:
            return [], float('inf'), {}, {}

        # If only 1 path exists, no need for optimization
        if len(candidate_paths) == 1:
            best_path = candidate_paths[0]
            cost = sum(self.graph[best_path[i]][best_path[i+1]]['routing_cost'] for i in range(len(best_path) - 1))
            if demand > 0.0:
                self._apply_demand_to_path(best_path, demand)
            metrics = self.calculate_path_metrics(best_path)
            return best_path, cost, metrics, {'solver': 'trivial', 'num_candidates': 1}

        # Formulate QUBO
        qp = self._formulate_qubo(candidate_paths)
        
        # Solve
        if use_qaoa:
            x_sol, fval, metadata = self.solve_qaoa(qp, reps=qaoa_reps)
        else:
            x_sol, fval, metadata = self.solve_exact(qp)
            
        metadata['num_candidates'] = len(candidate_paths)
        
        # Decode bitstring
        # Enforce exact one route selected.
        selected_idx = -1
        x_sol_int = [int(round(val)) for val in x_sol]
        metadata['feasibility_failed'] = False
        if sum(x_sol_int) == 1:
            selected_idx = x_sol_int.index(1)
        else:
            # Feasibility failed. The penalty wasn't high enough or optimizer converged to local min.
            metadata['feasibility_failed'] = True
            # Find the path with minimal raw cost as fallback
            costs = [sum(self.graph[p[i]][p[i+1]]['routing_cost'] for i in range(len(p) - 1)) for p in candidate_paths]
            selected_idx = np.argmin(costs)
            
        best_path = candidate_paths[selected_idx]
        best_cost = sum(self.graph[best_path[i]][best_path[i+1]]['routing_cost'] for i in range(len(best_path) - 1))
        
        if demand > 0.0:
            self._apply_demand_to_path(best_path, demand)
            
        metrics = self.calculate_path_metrics(best_path)
        return best_path, best_cost, metrics, metadata

    def route_batch_quantum(self, requests: List[Dict[str, Any]], use_qaoa: bool = True, qaoa_reps: int = 1) -> pd.DataFrame:
        """
        Route a batch of requests sequentially using quantum optimization.
        """
        results = []
        for req in requests:
            src = req['source']
            dst = req['target']
            demand = req.get('demand', 0.0)
            
            path, cost, metrics, meta = self.route_request_quantum(src, dst, demand, use_qaoa=use_qaoa, qaoa_reps=qaoa_reps)
            
            res = {
                'source': src,
                'target': dst,
                'demand': demand,
                'selected_path': str(path),
                'route_cost': cost,
                'total_latency': metrics.get('total_latency', float('inf')),
                'max_utilization': metrics.get('max_utilization', float('inf')),
                'total_packet_loss': metrics.get('total_packet_loss', float('inf')),
                'solver': meta.get('solver', ''),
                'runtime_sec': meta.get('runtime_sec', 0.0),
                'qubo_fval': meta.get('fval', 0.0),
                'feasibility_failed': meta.get('feasibility_failed', False)
            }
            results.append(res)
            
        return pd.DataFrame(results)
