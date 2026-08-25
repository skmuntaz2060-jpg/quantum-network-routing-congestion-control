# Research Benchmark — 20 Routers × 50 Steps

## Experiment Configuration
- **Network size**: 20 routers
- **Simulation steps**: 50
- **Traffic pairs**: 20
- **Total routing requests**: 1,000
- **Candidate routes K**: 3
- **QAOA configuration**: Reps=1, Penalty Lambda=1000.0, Qiskit StatevectorSampler, COBYLA optimizer (maxiter=15)
- **Random seed**: 123 (implicit network seed for Barabasi-Albert topology)

## Results

| Metric | Shortest Path | Congestion-Aware | AI + QAOA |
|---|---|---|---|
| avg_latency | 22.36 | 45.10 | 45.38 |
| p95_latency | 36.41 | 90.75 | 90.75 |
| throughput | 20,225 | 20,225 | 20,225 |
| packet_loss_rate | 95.06% | 74.55% | 73.97% |
| congestion_rate | 97.70% | 87.30% | 87.20% |
| average_utilization | 17.93 | 3.88 | 3.82 |
| peak_utilization | 54.68 | 13.64 | 13.16 |
| load_imbalance | 12.29 | 2.84 | 2.79 |
| avg_route_time_sec | 0.0027s | 0.0015s | 0.9826s |
| qaoa_runtime_sec | 0.0s | 0.0s | 0.9811s |
| avg_objective_val | 0.0 | 0.0 | 1065.82 |

## ML Routing Influence
- **Total routing requests evaluated**: 1,000
- **Requests with non-zero ML penalty**: 885 requests encountered an ML-penalized edge
- **Route changes caused by ML**: 341
- **Route-change rate**: 34.10%

**Representative Example (Request 57)**:
- **Source**: 5, **Destination**: 9, **Demand**: 22.79
- **Congestion-aware Route**: `[5, 0, 9]` (Cost: 83.95)
- **AI-penalized Route**: `[5, 6, 9]` (Cost: 92.74)

## QAOA Validation
- **Candidate Paths (K)**: 3 generated via `nx.shortest_simple_paths` using the augmented routing cost.
- **AI Penalty**: Confirmed included in `routing_cost` attribute of edges.
- **QUBO Objective**: Uses the augmented objective; properly penalized paths correctly shift probability to alternative routes.

**Representative Example (Request 2004)**:
- **Source**: 8, **Destination**: 14, **Demand**: 11.11
- **Selected Route**: `[8, 3, 14]`
- **Classical Cost**: 22.4877
- **QUBO Objective (fval)**: 22.4877
- *The objective precisely matched the classical candidate cost.*

## Interpretation
- **AI + QAOA reduced packet loss slightly** relative to congestion-aware classical routing in this experiment (73.97% vs 74.55%).
- **AI + QAOA reduced congestion slightly** (87.20% vs 87.30%).
- **QAOA runtime was substantially higher** than classical routing (0.98 seconds per route compared to 0.0015 seconds).
- **This experiment does NOT demonstrate quantum computational speedup or quantum advantage**. The system perfectly recreates the optimal classical routing decision based on the candidate paths and objective function provided.

## Limitations
- **Synthetic Network/Traffic**: Uses simulated Barabasi-Albert graphs and parameterized traffic generation rather than real ISP/datacenter topology traces.
- **Simulator-based QAOA**: Executed on Qiskit's `StatevectorSampler` simulator, which scales exponentially with candidate size and does not reflect real QPU noise or connectivity constraints.
- **Finite Benchmark Scale**: Limited to a 20-router footprint.
- **Heavy Simulated Congestion**: Peak utilization factors exceeding 50x baseline link capacity represent extreme flood scenarios, preventing meaningful loss minimization regardless of routing strategy.
- **Candidate-Path Approach**: QAOA is restricted to selecting from $K=3$ classically pre-calculated paths rather than exploring the full adjacency matrix space natively.
- **Absence of Quantum Advantage**: The solver identifies the global minimum of the QUBO, which maps identically to sorting $K$ classical path costs.

## Reproducibility
To reproduce these findings exactly, execute the following command:
```bash
python src/quantum_routing/main.py --routers 20 --steps 50
```
This forces the dataset generation to construct a 20-node topology and simulate traffic over 50 test intervals, outputting results into `results/scenario_{A,B,C}_raw.csv`.
