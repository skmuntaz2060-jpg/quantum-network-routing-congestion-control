import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from quantum_routing.dashboard.utils import load_metrics_summary, render_sidebar_info

st.set_page_config(page_title="Performance Benchmark", page_icon="📊", layout="wide")
render_sidebar_info()

st.title("📊 Performance Benchmark")
st.markdown("Compare the aggregated metrics across Shortest-Path, Congestion-Aware, and Quantum AI routing.")

summary = load_metrics_summary()

if summary is None:
    st.warning("Metrics summary not found.")
    st.stop()

# Convert summary dict to DataFrame for Plotly
df_summary = pd.DataFrame.from_dict(summary, orient='index').reset_index()
df_summary.rename(columns={'index': 'Strategy'}, inplace=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Latency (ms)")
    fig1 = px.bar(df_summary, x='Strategy', y='avg_latency', color='Strategy', text_auto='.2f')
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Total Packet Loss")
    fig3 = px.bar(df_summary, x='Strategy', y='total_packet_loss', color='Strategy', text_auto='.4f')
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("Throughput")
    fig2 = px.bar(df_summary, x='Strategy', y='throughput', color='Strategy', text_auto='.1f')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Routing Computation Time (s)")
    fig4 = px.bar(df_summary, x='Strategy', y='avg_route_time_sec', color='Strategy', text_auto='.4f')
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

st.subheader("Radar Chart Comparison")
# Normalize metrics for radar chart
metrics = ['avg_latency', 'total_packet_loss', 'avg_utilization', 'load_imbalance']

fig_radar = go.Figure()

for i, row in df_summary.iterrows():
    # We want smaller values to be better (closer to center or further out depending on perspective)
    # Usually, larger area = better, so we can invert costs, but for raw visualization we just plot raw.
    # Actually, min-max scaling is better for radar charts.
    fig_radar.add_trace(go.Scatterpolar(
        r=[row[m] for m in metrics] + [row[metrics[0]]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name=row['Strategy']
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
        )),
    showlegend=True
)

st.plotly_chart(fig_radar, use_container_width=True)

st.subheader("Raw Summary Data")
st.dataframe(df_summary)
