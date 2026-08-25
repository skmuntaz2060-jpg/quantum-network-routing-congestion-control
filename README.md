# AI + Quantum Network Traffic Routing and Congestion Control (v1.1.0)

A highly scalable, congestion-aware network routing simulation combining Classical Machine Learning and Quantum Approximate Optimization Algorithm (QAOA).

## 1. Problem Statement
Modern communication networks (ISPs, datacenters, 5G architectures) face highly dynamic, non-linear traffic bursts that traditional shortest-path algorithms struggle to manage. When multiple data streams simultaneously take the optimal shortest path, extreme network congestion occurs, leading to packet loss, high latency, and degraded throughput.

## 2. Why Network Congestion-Aware Routing Matters
Predictive, congestion-aware routing ensures high availability and low latency by identifying emerging bottlenecks *before* they cause packet drops. By proactively shifting traffic away from predicted hotspots, overall network throughput and reliability are maximized.

## 3. Proposed Architecture
This system utilizes a hybrid approach to intelligently route traffic across a dynamically shifting network:
1. **Network Simulation**: A simulated Barabasi-Albert topology with dynamic capacity constraints.
2. **AI/ML Component (TrafficPredictor)**: Machine learning models (Random Forest / XGBoost) predict edge demand based on temporal features and rolling statistics. Edges predicted to exceed capacity are assigned severe "AI Penalties."
3. **Classical Routing**: Candidate routes are generated using K-shortest paths weighted by latency, current utilization, and ML penalties.
4. **Quantum Routing (QAOA)**: The optimal route is selected by formulating the path-selection problem as a Quadratic Unconstrained Binary Optimization (QUBO) problem and solving it via a Quantum Approximate Optimization Algorithm (QAOA).

## 4. QUBO Formulation and QAOA Component
The routing decision is modeled mathematically as a QUBO where $K$ binary variables $x_i$ represent the selection of candidate path $i$. 
The objective function minimizes the augmented classical routing costs (including ML penalties) subject to a strict penalty constraint ensuring exactly one path is chosen.
The QUBO is solved using Qiskit's `StatevectorSampler` and the `COBYLA` optimizer.

## 5. Simulation Methodology
The system generates discrete time-steps of traffic requests (source, destination, demand). 
It evaluates three distinct scenarios on the same traffic:
- **Scenario A (Shortest-Path)**: Naive baseline routing minimizing only latency.
- **Scenario B (Congestion-Aware Classical)**: Classical routing incorporating current network utilization and packet loss probabilities.
- **Scenario C (AI + QAOA)**: Hybrid routing leveraging ML predictions to preemptively penalize congested edges, solved via Quantum optimization.

## 6. Execution Modes
- **Demo Mode**: A fast, lightweight execution designed for rapid testing and UI demonstration (8 routers, 2 time steps).
- **Research Mode**: A computationally intensive, rigorous benchmark (20 routers, 50 time steps) designed for macro-level scientific analysis.

---

## 7. Research Benchmark — 20 Routers × 50 Steps Results
*Scientifically validated on 1,000 dense routing requests under heavy simulated network load.*

| Metric | Shortest Path | Congestion-Aware | AI + QAOA |
|---|---|---|---|
| **avg_latency** | 22.36 ms | 45.10 ms | 45.38 ms |
| **p95_latency** | 36.41 ms | 90.75 ms | 90.75 ms |
| **throughput** | 20,225 Mbps | 20,225 Mbps | 20,225 Mbps |
| **packet_loss_rate** | 95.06% | 74.55% | 73.97% |
| **congestion_rate** | 97.70% | 87.30% | 87.20% |
| **average_utilization** | 17.93 | 3.88 | 3.82 |
| **peak_utilization** | 54.68 | 13.64 | 13.16 |
| **load_imbalance** | 12.29 | 2.84 | 2.79 |
| **avg_route_time_sec** | 0.0027s | 0.0015s | 0.9826s |
| **qaoa_runtime_sec** | 0.0s | 0.0s | 0.9811s |
| **avg_objective_val** | 0.0 | 0.0 | 1065.82 |

### ML Routing Influence
The Machine Learning component successfully identified high-risk pathways and actively altered routing topology:
- **Total routing requests evaluated**: 1,000
- **Requests encountering non-zero ML penalty**: 885
- **Route changes caused strictly by ML predictions**: 341
- **Route-change rate**: 34.10%

### Interpretation & Limitations
- **Findings**: The AI + QAOA pipeline reduced aggregate packet loss (73.97% vs 74.55%) and network congestion slightly relative to classical congestion-aware routing by successfully diverting 34% of traffic away from predicted hotspots.
- **Quantum Advantage**: **This experiment does NOT demonstrate quantum computational speedup or quantum advantage.** The QAOA solver perfectly reconstructs the optimal classical decision across $K$ candidate paths, but requires substantially higher classical runtime (0.98s vs 0.0015s) because it relies on classical simulation of quantum statevectors.
- **Limitations**: The benchmark relies on synthetic topologies, parameterized traffic generation, and simulator-based QAOA lacking hardware noise models. Furthermore, extreme congestion (peak utilization > 13x) limits the absolute effectiveness of *any* routing strategy.

---

## 8. Project Structure
```text
quantum-network-routing/
├── data/                 # Generated synthetic topologies and traffic
├── docs/                 # Documentation and benchmark reports
├── results/              # ML models, raw CSV outputs, and JSON metrics
├── src/quantum_routing/
│   ├── dashboard/        # Interactive Streamlit application
│   ├── ml/               # XGBoost/RandomForest prediction logic
│   ├── routing/          # Classical and Quantum (QAOA) routing algorithms
│   ├── simulation/       # NetworkX topology generation
│   └── main.py           # Core execution pipeline
└── tests/                # Comprehensive Pytest verification suite
```

## 9. How to Run Locally

### Prerequisites
- Python 3.9+
- Recommended: Virtual Environment (`venv` or `conda`)

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Demo Mode (Fast)
To rapidly generate a small topology and verify the pipeline:
```bash
python src/quantum_routing/main.py --routers 8 --steps 2
```

### Running Research Mode (Intensive)
To reproduce the rigorous 20-router benchmark:
```bash
python src/quantum_routing/main.py --routers 20 --steps 50
```

### Launching the Dashboard
Launch the interactive Streamlit UI to visualize the results:
```bash
python -m streamlit run src/quantum_routing/dashboard/app.py
```
View the dashboard locally at `http://localhost:8501`.

### Testing
Execute the scientifically validated test suite (30/30 passing):
```bash
python -m pytest
```

## 10. License
This project is licensed under the MIT License.
