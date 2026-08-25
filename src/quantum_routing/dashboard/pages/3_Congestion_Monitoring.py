import streamlit as st
import plotly.express as px
import pandas as pd
import networkx as nx
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from quantum_routing.dashboard.utils import load_topology, load_routing_scenario, render_sidebar_info

st.set_page_config(page_title="Congestion Monitoring", page_icon="🔥", layout="wide")
render_sidebar_info()

st.title("🔥 Congestion Monitoring")
st.markdown("Compare the link utilization across different routing scenarios.")

G = load_topology()
if G is None:
    st.stop()

scenario = st.radio("Select Scenario:", ["Shortest-Path (A)", "Congestion-Aware (B)", "AI + QAOA (C)"], horizontal=True)
scenario_key = scenario.split("(")[1][0]

df_raw = load_routing_scenario(scenario_key)

if df_raw is None:
    st.warning(f"Data for Scenario {scenario_key} not found.")
    st.stop()

st.write(f"Total Requests Processed: {len(df_raw)}")

# Create a heatmap of max utilization per request?
# Since we didn't save the full link state per step, we only saved 'max_utilization' of the *selected path*.
fig = px.line(df_raw, x='step', y='max_utilization', title=f"Max Utilization of Selected Path Over Time - Scenario {scenario_key}", 
              markers=True)
# Add a threshold line
fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Congestion Threshold (0.8)")
fig.update_layout(yaxis_range=[0, 1.05])
st.plotly_chart(fig, width="stretch")

st.subheader("Raw Request Log")
st.dataframe(df_raw, width="stretch")
