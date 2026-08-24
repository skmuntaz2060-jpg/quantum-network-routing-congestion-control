import streamlit as st

pages = {
    "Dashboards": [
        st.Page("quantum_routing/dashboard/app.py", title="Home", icon="🏠"),
        st.Page("quantum_routing/dashboard/pages/1_Network_Overview.py", title="Network Overview", icon="🌐"),
        st.Page("quantum_routing/dashboard/pages/2_Traffic_Prediction.py", title="Traffic Prediction", icon="📈"),
        st.Page("quantum_routing/dashboard/pages/3_Congestion_Monitoring.py", title="Congestion Monitoring", icon="🚦"),
        st.Page("quantum_routing/dashboard/pages/4_Classical_Routing.py", title="Classical Routing", icon="🛣️"),
        st.Page("quantum_routing/dashboard/pages/5_Quantum_QAOA_Routing.py", title="Quantum QAOA Routing", icon="🌌"),
        st.Page("quantum_routing/dashboard/pages/6_Performance_Benchmark.py", title="Performance Benchmark", icon="⏱️"),
        st.Page("quantum_routing/dashboard/pages/7_Experiment_Configuration.py", title="Experiment Configuration", icon="⚙️")
    ]
}

pg = st.navigation(pages)
pg.run()
