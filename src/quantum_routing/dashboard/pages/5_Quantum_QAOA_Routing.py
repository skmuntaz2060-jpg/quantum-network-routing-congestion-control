import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from quantum_routing.dashboard.utils import load_topology, load_routing_scenario, render_sidebar_info

st.set_page_config(page_title="Quantum QAOA Routing", page_icon="⚛️", layout="wide")
render_sidebar_info()

st.title("⚛️ Quantum QAOA Routing")
st.markdown("Dive into the QAOA decisions formulated from QUBO problems with dynamic AI-adjusted weights.")

G = load_topology()
df_C = load_routing_scenario('C')

if G is None or df_C is None:
    st.warning("Missing data.")
    st.stop()

# Select Request
st.sidebar.subheader("Route Selection")
step = st.sidebar.selectbox("Select Time Step", sorted(df_C['step'].unique()))
df_step = df_C[df_C['step'] == step]

req_options = []
for idx, row in df_step.iterrows():
    req_options.append(f"Request {idx}: Node {row['source']} -> {row['target']}")
    
selected_req = st.sidebar.selectbox("Select Request", req_options)
req_idx = int(selected_req.split(":")[0].replace("Request ", ""))

row = df_C.loc[req_idx]

st.subheader(f"Quantum Route (Req {req_idx})")

col1, col2 = st.columns([1, 1])

with col1:
    st.write(f"**Selected Path**: {row['selected_path']}")
    st.write(f"**Solver**: {row.get('solver', 'N/A')}")
    st.write(f"**QAOA Runtime**: {row.get('runtime_sec', 0.0):.4f} sec")
    st.write(f"**QUBO Objective Value**: {row.get('qubo_fval', 0.0)}")
    
    st.write("### Path Metrics")
    st.write(f"- Latency: {row['total_latency']:.2f} ms")
    st.write(f"- Packet Loss: {row['total_packet_loss']:.4%}")
    st.write(f"- Max Utilization: {row['max_utilization']:.2%}")

with col2:
    def plot_route(G, path, title):
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            edge_x.extend([pos[edge[0]][0], pos[edge[1]][0], None])
            edge_y.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        base_edges = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
        
        path_x, path_y = [], []
        if isinstance(path, list) and len(path) > 1:
            for i in range(len(path)-1):
                path_x.extend([pos[path[i]][0], pos[path[i+1]][0], None])
                path_y.extend([pos[path[i]][1], pos[path[i+1]][1], None])
        active_edges = go.Scatter(x=path_x, y=path_y, line=dict(width=4, color='purple'), hoverinfo='none', mode='lines')
        
        nodes = go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()], 
                           mode='markers+text', text=[str(n) for n in G.nodes()], 
                           marker=dict(size=20, color='plum'), textposition="bottom center")
                           
        fig = go.Figure(data=[base_edges, active_edges, nodes], layout=go.Layout(title=title, showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        return fig

    fig = plot_route(G, row['selected_path'], "QAOA Selected Path")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("QUBO Objective Value History")
fig2 = go.Figure(data=go.Scatter(x=df_C.index, y=df_C['qubo_fval'], mode='lines+markers', name='FVAL'))
fig2.update_layout(title="Objective Value per Request", xaxis_title="Request Index", yaxis_title="QUBO FVal")
st.plotly_chart(fig2, use_container_width=True)
