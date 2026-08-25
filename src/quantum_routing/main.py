import argparse
import os
import copy
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any

from quantum_routing.config import PROJECT_ROOT, DATA_DIR, RESULTS_DIR
from quantum_routing.simulation.network import NetworkSimulator, save_dataset
from quantum_routing.ml.predictor import TrafficPredictor
from quantum_routing.routing.classical import ClassicalRouter
from quantum_routing.routing.quantum import QuantumRouter

def calculate_network_metrics(df_results: pd.DataFrame, G: nx.Graph, route_times: list, qaoa_times: list) -> Dict[str, Any]:
    """Calculate aggregated performance metrics for a routing strategy."""
    
    # Identify failed routes
    failed = df_results[df_results['selected_path'] == '[]']
    success = df_results[df_results['selected_path'] != '[]']
    
    throughput = success['demand'].sum() if not success.empty else 0.0
    
    if throughput > 0:
        packet_loss_rate = (success['packet_loss_rate'] * success['demand']).sum() / throughput
    else:
        packet_loss_rate = 0.0
    
    avg_latency = success['total_latency'].mean() if not success.empty else float('inf')
    p95_latency = success['total_latency'].quantile(0.95) if not success.empty else float('inf')
    
    num_success = len(success)
    if num_success > 0:
        congestion_rate = len(success[success['max_utilization'] > 0.8]) / num_success
        average_utilization = success['max_utilization'].mean()
        peak_utilization = success['max_utilization'].max()
        load_imbalance = success['max_utilization'].std() if num_success > 1 else 0.0
    else:
        congestion_rate = 0.0
        average_utilization = 0.0
        peak_utilization = 0.0
        load_imbalance = 0.0
    
    avg_route_time = np.mean(route_times) if route_times else 0.0
    avg_qaoa_time = np.mean(qaoa_times) if qaoa_times else 0.0
    
    avg_obj = success['qubo_fval'].mean() if 'qubo_fval' in success.columns and not success.empty else 0.0
    
    return {
        'total_requests': len(df_results),
        'failed_requests': len(failed),
        'throughput': throughput,
        'packet_loss_rate': packet_loss_rate,
        'avg_latency': avg_latency,
        'p95_latency': p95_latency,
        'average_utilization': average_utilization,
        'peak_utilization': peak_utilization,
        'congestion_rate': congestion_rate,
        'load_imbalance': load_imbalance,
        'avg_route_time_sec': avg_route_time,
        'qaoa_runtime_sec': avg_qaoa_time,
        'avg_objective_val': avg_obj
    }

def run_experiment(num_routers=10, steps_to_simulate=3, qaoa_reps=1):
    print("--- 1. Initialization & Simulation ---")
    print("Generating topology and traffic...")
    simulator = NetworkSimulator(num_nodes=num_routers, seed=42)
    G_base = simulator.generate_topology()
    
    # We must train the ML model on a sufficient history, and THEN simulate routing on future (out-of-sample) steps.
    train_steps = 100
    total_steps = train_steps + steps_to_simulate
    traffic_df = simulator.generate_traffic(time_steps=total_steps, num_pairs=num_routers)
    
    # Save the dataset
    save_dataset(G_base, traffic_df, str(DATA_DIR))
    
    # Split the dataset for strict evaluation
    train_traffic_df = traffic_df[traffic_df['time_step'] < train_steps].copy()
    
    print("\n--- 2. Machine Learning ---")
    predictor = TrafficPredictor()
    print(f"Training TrafficPredictor on first {train_steps} steps...")
    try:
        predictor.train_and_evaluate(train_traffic_df, output_dir=str(RESULTS_DIR))
        print(f"ML Model Trained. RF RMSE: {predictor.metrics['regression']['RandomForest']['RMSE']:.4f}")
    except Exception as e:
        print(f"Failed to train ML model: {e}. Skipping AI integration.")
        predictor.is_trained = False
        
    print(f"\n--- 3. Running Scenarios (Simulating {steps_to_simulate} steps out-of-sample) ---")
    
    scenarios = {
        'A': 'Shortest-path baseline',
        'B': 'Congestion-aware classical',
        'C': 'AI + QAOA'
    }
    
    results = {k: [] for k in scenarios.keys()}
    metrics_summary = {}
    
    for scenario_key, scenario_name in scenarios.items():
        print(f"\nScenario {scenario_key}: {scenario_name}")
        
        # Deep copy network to ensure exactly the same starting state for each scenario
        G_scenario = copy.deepcopy(G_base)
        
        if scenario_key == 'A':
            weights = {'latency': 1.0, 'utilization': 0.0, 'packet_loss': 0.0, 'congestion_penalty': 0.0}
            router = ClassicalRouter(G_scenario, weights)
        elif scenario_key == 'B':
            weights = {'latency': 1.0, 'utilization': 50.0, 'packet_loss': 100.0, 'congestion_penalty': 500.0}
            router = ClassicalRouter(G_scenario, weights)
        elif scenario_key == 'C':
            weights = {'latency': 1.0, 'utilization': 50.0, 'packet_loss': 100.0, 'congestion_penalty': 500.0}
            # AI predictions will dynamically update costs. QAOA will route.
            router = QuantumRouter(G_scenario, weights, penalty_lambda=1000.0)
            
        route_times = []
        qaoa_times = []
        scenario_results = []
        
        for step in range(train_steps, total_steps):
            step_traffic = traffic_df[traffic_df['time_step'] == step]
            
            # Predict congestion for AI scenario
            if scenario_key == 'C' and predictor.is_trained:
                # Naive AI integration: penalize edges predicted to have high demand
                # We need features for the current step. We use the full traffic_df to generate features
                # and then select the current step.
                all_features = predictor.create_features(traffic_df)
                step_features = all_features[all_features['time_step'] == step]
                
                if not step_features.empty:
                    feature_cols = [
                        'time_sin', 'time_cos', 'demand_lag_1', 'demand_lag_2',
                        'demand_roll_mean_3', 'demand_roll_std_3',
                        'demand_roll_mean_5', 'demand_roll_std_5'
                    ]
                    
                    # Choose model
                    model = getattr(predictor, 'xgb_regressor', predictor.rf_regressor)
                    if model is None:
                        model = predictor.rf_regressor
                        
                    preds = model.predict(step_features[feature_cols])
                    
                    # For every edge predicted to have high demand (> 20), increase congestion penalty weight temporarily
                    high_demand_indices = np.where(preds > 20)[0]
                    if len(high_demand_indices) > 0:
                        high_demand_edges = step_features.iloc[high_demand_indices]
                        for _, row in high_demand_edges.iterrows():
                            u, v = int(row['source']), int(row['destination'])
                            if G_scenario.has_edge(u, v):
                                 G_scenario[u][v]['routing_cost_ai_penalty'] = 200.0 # Arbitrary AI penalty
                
            for _, row in step_traffic.iterrows():
                src = int(row['source'])
                dst = int(row['destination'])
                demand = float(row['demand'])
                
                start_time = time.time()
                if scenario_key in ['A', 'B']:
                    path, cost, metric = router.route_request(src, dst, demand)
                    qaoa_time = 0.0
                    fval = 0.0
                else:
                    path, cost, metric, meta = router.route_request_quantum(src, dst, demand, use_qaoa=True, qaoa_reps=qaoa_reps)
                    qaoa_time = meta.get('runtime_sec', 0.0)
                    fval = meta.get('fval', 0.0)
                    
                end_time = time.time()
                
                route_times.append(end_time - start_time)
                if qaoa_time > 0:
                    qaoa_times.append(qaoa_time)
                    
                res = {
                    'step': step,
                    'source': src,
                    'target': dst,
                    'demand': demand,
                    'selected_path': str(path),
                    'route_cost': cost,
                    'total_latency': metric.get('total_latency', float('inf')),
                    'packet_loss_rate': metric.get('packet_loss_rate', float('inf')),
                    'max_utilization': metric.get('max_utilization', 0.0),
                    'qubo_fval': fval
                }
                scenario_results.append(res)
                
            # Decay utilization at end of step
            for u, v, data in G_scenario.edges(data=True):
                data['utilization'] *= 0.5
                
        df_results = pd.DataFrame(scenario_results)
        results[scenario_key] = df_results
        
        # Calculate aggregated metrics
        agg_metrics = calculate_network_metrics(df_results, G_scenario, route_times, qaoa_times)
        metrics_summary[scenario_name] = agg_metrics
        
        print(f"  Avg Latency: {agg_metrics['avg_latency']:.2f}")
        print(f"  Congestion Rate: {agg_metrics['congestion_rate']:.2%}")
        
    print("\n--- 4. Saving Results ---")
    
    # Save raw
    for k, df in results.items():
        df.to_csv(RESULTS_DIR / f'scenario_{k}_raw.csv', index=False)
        
    # Save summary
    with open(RESULTS_DIR / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    # Print Markdown Table
    print("\n### Performance Comparison")
    print("| Metric | Shortest-Path (A) | Congestion-Aware (B) | AI + QAOA (C) |")
    print("|---|---|---|---|")
    
    metrics_to_print = ['avg_latency', 'p95_latency', 'throughput', 'packet_loss_rate', 'congestion_rate', 'average_utilization', 'peak_utilization', 'load_imbalance', 'avg_route_time_sec', 'qaoa_runtime_sec']
    
    for m in metrics_to_print:
        val_A = metrics_summary['Shortest-path baseline'][m]
        val_B = metrics_summary['Congestion-aware classical'][m]
        val_C = metrics_summary['AI + QAOA'][m]
        
        if 'rate' in m:
            print(f"| {m} | {val_A:.2%} | {val_B:.2%} | {val_C:.2%} |")
        else:
            print(f"| {m} | {val_A:.4f} | {val_B:.4f} | {val_C:.4f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run End-to-End Routing Benchmark")
    parser.add_argument("--steps", type=int, default=2, help="Number of time steps to simulate")
    parser.add_argument("--routers", type=int, default=10, help="Number of routers in topology")
    
    args = parser.parse_args()
    run_experiment(num_routers=args.routers, steps_to_simulate=args.steps)
