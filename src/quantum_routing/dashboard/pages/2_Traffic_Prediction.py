import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from quantum_routing.dashboard.utils import load_traffic, load_ml_metrics, render_sidebar_info

st.set_page_config(page_title="Traffic Prediction", page_icon="📈", layout="wide")
render_sidebar_info()

st.title("📈 Traffic Prediction (Machine Learning)")
st.markdown("Evaluating the performance of the XGBoost / Random Forest models on historical traffic.")

traffic_df = load_traffic()
metrics = load_ml_metrics()

if traffic_df is None or metrics is None:
    st.warning("ML Metrics or Traffic data not found.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance")
    rf_metrics = metrics.get('regression', {}).get('RandomForest', {})
    xgb_metrics = metrics.get('regression', {}).get('XGBoost', {})
    
    if rf_metrics:
        st.metric("Random Forest RMSE", f"{rf_metrics.get('RMSE', 0):.4f}")
    if xgb_metrics:
        st.metric("XGBoost RMSE", f"{xgb_metrics.get('RMSE', 0):.4f}")

with col2:
    st.subheader("Classification Performance")
    rf_clf = metrics.get('classification', {}).get('RandomForest', {})
    if rf_clf:
        st.write("**Random Forest**")
        st.write(f"- Accuracy: {rf_clf.get('Accuracy', 0):.2%}")
        st.write(f"- F1 Score: {rf_clf.get('F1', 0):.2f}")

st.divider()
st.subheader("Traffic Visualization")

st.markdown("Select a Source-Destination pair to view its historical traffic demands.")

pairs = traffic_df.groupby(['source', 'destination']).size().reset_index()
pair_options = [f"Source {row['source']} -> Dest {row['destination']}" for _, row in pairs.iterrows()]

selected_pair = st.selectbox("Select Pair", pair_options)
selected_src = int(selected_pair.split(" ")[1])
selected_dst = int(selected_pair.split(" ")[4])

filtered_df = traffic_df[(traffic_df['source'] == selected_src) & (traffic_df['destination'] == selected_dst)]

fig = px.line(filtered_df, x='time_step', y='demand', title=f"Demand over Time for {selected_pair}")
# highlight congestion events
congestion_events = filtered_df[filtered_df['is_congestion_event'] == True]
if not congestion_events.empty:
    fig.add_scatter(x=congestion_events['time_step'], y=congestion_events['demand'], mode='markers', 
                    marker=dict(color='red', size=10, symbol='x'), name='Congestion Event')

st.plotly_chart(fig, use_container_width=True)
