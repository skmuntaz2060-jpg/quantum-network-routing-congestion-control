import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import sys
import os

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_topology, render_sidebar_info

st.set_page_config(page_title="Network Overview", page_icon="🕸️", layout="wide")
render_sidebar_info()

st.title("🕸️ Network Overview")
st.markdown("Visualizing the physical network topology and baseline link capacities.")

G = load_topology()

if G is None:
    st.warning("No topology data found. Please generate the dataset or run the experiment first.")
    st.stop()

# Use NetworkX spring layout for visualization
pos = nx.spring_layout(G, seed=42)

edge_x = []
edge_y = []
edge_text = []

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    
    cap = edge[2].get('capacity', 0)
    lat = edge[2].get('latency', 0)
    # The text goes in the middle of the edge roughly
    edge_text.append(f"Capacity: {cap:.1f} Mbps<br>Latency: {lat:.1f} ms")

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# We can also add invisible nodes for edge hover text if desired, but node hover is simpler.

node_x = []
node_y = []
node_text = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"Router {node}")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[str(n) for n in G.nodes()],
    textposition="bottom center",
    hoverinfo='text',
    hovertext=node_text,
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[G.degree(n) for n in G.nodes()],
        size=30,
        colorbar=dict(
            thickness=15,
            title=dict(text='Node Degree', side='right'),
            xanchor='left'
        ),
        line_width=2))

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(text='<br>Simulated Quantum Network Topology', font=dict(size=16)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="NetworkX Barabasi-Albert Graph",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

st.plotly_chart(fig, use_container_width=True)

st.subheader("Link Specifications")
st.write(f"Total Routers (Nodes): {G.number_of_nodes()}")
st.write(f"Total Links (Edges): {G.number_of_edges()}")

# Show a dataframe of links
edge_data = []
for u, v, data in G.edges(data=True):
    edge_data.append({
        'Source': u,
        'Destination': v,
        'Capacity (Mbps)': round(data.get('capacity', 0), 2),
        'Latency (ms)': round(data.get('latency', 0), 2),
        'Packet Loss Rate': round(data.get('packet_loss', 0), 4)
    })

import pandas as pd
st.dataframe(pd.DataFrame(edge_data), use_container_width=True)
