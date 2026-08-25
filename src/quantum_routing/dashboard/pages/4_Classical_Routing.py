import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from quantum_routing.dashboard.utils import load_topology, load_routing_scenario, render_sidebar_info

st.set_page_config(page_title="Classical Routing", page_icon="🛣️", layout="wide")
render_sidebar_info()

st.title("🛣️ Classical Routing Visualization")
st.markdown("Compare exact routes chosen by Shortest-Path vs Congestion-Aware algorithms.")

G = load_topology()
df_A = load_routing_scenario('A')
df_B = load_routing_scenario('B')

if G is None or df_A is None or df_B is None:
    st.warning("Missing data.")
    st.stop()

# Select Request
st.sidebar.subheader("Route Selection")
step = st.sidebar.selectbox("Select Time Step", sorted(df_A['step'].unique()))
df_A_step = df_A[df_A['step'] == step]

req_options = []
for idx, row in df_A_step.iterrows():
    req_options.append(f"Request {idx}: Node {row['source']} -> {row['target']}")
    
selected_req = st.sidebar.selectbox("Select Request", req_options)
req_idx = int(selected_req.split(":")[0].replace("Request ", ""))

row_A = df_A.loc[req_idx]
row_B = df_B.loc[req_idx]

col1, col2 = st.columns(2)

def plot_route(G, path, title):
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    base_edges = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    
    path_edges_x = []
    path_edges_y = []
    if isinstance(path, list) and len(path) > 1:
        for i in range(len(path)-1):
            x0, y0 = pos[path[i]]
            x1, y1 = pos[path[i+1]]
            path_edges_x.extend([x0, x1, None])
            path_edges_y.extend([y0, y1, None])
            
    active_edges = go.Scatter(x=path_edges_x, y=path_edges_y, line=dict(width=4, color='red'), hoverinfo='none', mode='lines')
    
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    nodes = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[str(n) for n in G.nodes()], 
                       marker=dict(size=20, color='lightblue'), textposition="bottom center")
                       
    fig = go.Figure(data=[base_edges, active_edges, nodes], layout=go.Layout(title=title, showlegend=False, 
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

with col1:
    st.subheader("Scenario A (Shortest-Path)")
    st.write(f"**Path**: {row_A['selected_path']}")
    st.write(f"**Latency**: {row_A['total_latency']:.2f} ms")
    st.write(f"**Max Utilization**: {row_A['max_utilization']:.2%}")
    figA = plot_route(G, row_A['selected_path'], "Shortest Path Route")
    st.plotly_chart(figA, width="stretch")

with col2:
    st.subheader("Scenario B (Congestion-Aware)")
    st.write(f"**Path**: {row_B['selected_path']}")
    st.write(f"**Latency**: {row_B['total_latency']:.2f} ms")
    st.write(f"**Max Utilization**: {row_B['max_utilization']:.2%}")
    figB = plot_route(G, row_B['selected_path'], "Congestion-Aware Route")
    st.plotly_chart(figB, width="stretch")
