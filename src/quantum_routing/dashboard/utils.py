import os
import json
import pandas as pd
import networkx as nx
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results')

@st.cache_data
def load_topology():
    topology_file = os.path.join(DATA_DIR, 'topology.json')
    if not os.path.exists(topology_file):
        return None
    with open(topology_file, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)

@st.cache_data
def load_traffic():
    traffic_file = os.path.join(DATA_DIR, 'traffic.csv')
    if not os.path.exists(traffic_file):
        return None
    return pd.read_csv(traffic_file)

@st.cache_data
def load_routing_scenario(scenario_key):
    file_path = os.path.join(RESULTS_DIR, f'scenario_{scenario_key}_raw.csv')
    if not os.path.exists(file_path):
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
    file_path = os.path.join(RESULTS_DIR, 'metrics_summary.json')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_ml_metrics():
    file_path = os.path.join(RESULTS_DIR, 'ml_metrics.json')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

def render_sidebar_info():
    st.sidebar.markdown("### Quantum Routing Simulator")
    st.sidebar.markdown("An end-to-end simulation pipeline combining Classical algorithms, Machine Learning predictions, and Quantum QAOA optimization.")
    st.sidebar.info("Navigate through the pages above to explore the different modules.")
