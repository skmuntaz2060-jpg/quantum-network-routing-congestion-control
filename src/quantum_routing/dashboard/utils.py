import os
import json
import pandas as pd
import networkx as nx
import streamlit as st

from quantum_routing.config import DATA_DIR, RESULTS_DIR

@st.cache_resource
def ensure_demo_artifacts():
    """
    On application startup, ensure that a lightweight Demo Mode artifact set exists.
    If they do not exist, generate them automatically.
    """
    topology_file = DATA_DIR / 'topology.json'
    
    if topology_file.exists():
        return
        
    print("Demo artifacts not found. Generating lightweight demo artifacts...")
    # Import locally to avoid circular imports and load times if not needed
    from quantum_routing.main import run_experiment
    
    try:
        # Run the fast, safe demo configuration
        run_experiment(num_routers=8, steps_to_simulate=2, qaoa_reps=1, seed=42)
        print("Demo initialization complete.")
        # Clear data cache so subsequent data loads pick up new files
        st.cache_data.clear()
    except Exception as e:
        print(f"Failed to generate demo artifacts: {e}")
        raise e

@st.cache_data
def load_topology():
    ensure_demo_artifacts()
    topology_file = DATA_DIR / 'topology.json'
    if not topology_file.exists():
        return None
    with open(topology_file, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)

@st.cache_data
def load_traffic():
    ensure_demo_artifacts()
    traffic_file = DATA_DIR / 'traffic.csv'
    if not traffic_file.exists():
        return None
    return pd.read_csv(traffic_file)

@st.cache_data
def load_routing_scenario(scenario_key):
    ensure_demo_artifacts()
    file_path = RESULTS_DIR / f'scenario_{scenario_key}_raw.csv'
    if not file_path.exists():
        return None
    
    # Safely load the CSV and parse lists if any
    df = pd.read_csv(file_path)
    
    # 'selected_path' is saved as a string like "[0, 1, 3]". We convert it back to a list
    if 'selected_path' in df.columns:
        import ast
        def safe_eval(x):
            try:
                return ast.literal_eval(x)
            except:
                return []
        df['selected_path'] = df['selected_path'].apply(safe_eval)
        
    return df

@st.cache_data
def load_metrics_summary():
    ensure_demo_artifacts()
    file_path = RESULTS_DIR / 'metrics_summary.json'
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_ml_metrics():
    ensure_demo_artifacts()
    file_path = RESULTS_DIR / 'ml_metrics.json'
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

def render_sidebar_info():
    from quantum_routing import __version__
    
    st.sidebar.markdown("### Quantum Routing Simulator")
    st.sidebar.markdown(f"**Version:** {__version__}")
    st.sidebar.markdown("An end-to-end simulation pipeline combining Classical algorithms, Machine Learning predictions, and Quantum QAOA optimization.")
    st.sidebar.info("Navigate through the pages above to explore the different modules.")
