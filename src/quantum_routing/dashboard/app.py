import streamlit as st
from quantum_routing.dashboard.utils import render_sidebar_info

st.set_page_config(
    page_title="Quantum Network Routing",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar_info()

st.title("🌌 AI + Quantum Network Routing Dashboard")

st.markdown("""
Welcome to the Quantum Network Routing experiment dashboard. 
This dashboard visualizes an end-to-end pipeline comparing classical shortest-path routing, congestion-aware classical routing, and a hybrid AI + Quantum QAOA approach.

### Architecture Overview

1. **Simulation**: Generates a synthetic Barabasi-Albert topology and applies diurnal and spiky traffic patterns.
2. **Machine Learning (AI)**: Predicts upcoming traffic demands based on historical lags and rolling averages.
3. **Classical Routing**: Solves pathfinding exactly using NetworkX algorithms based on simple edge weights (Latency + Congestion).
4. **Quantum QAOA Routing**: Uses the AI predictions to modulate network costs, and formulates the route selection as a QUBO problem, which is solved via Qiskit's Quantum Approximate Optimization Algorithm (QAOA).

Please select a module from the **Sidebar** to explore the data.
""")

st.info("👈 Use the navigation on the left to view the interactive visualizations.")
