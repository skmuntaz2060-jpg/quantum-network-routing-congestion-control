import streamlit as st
import sys
import os
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import render_sidebar_info

st.set_page_config(page_title="Experiment Configuration", page_icon="⚙️", layout="wide")
render_sidebar_info()

st.title("⚙️ Experiment Configuration")
st.markdown("Configure and run a new End-to-End simulation pipeline.")

with st.form("config_form"):
    st.subheader("Simulation Parameters")
    demo_mode = st.checkbox("Demo Mode (Fast & Safe for Streamlit Cloud)", value=True, help="Limits graph size and depth to prevent Out-Of-Memory errors.")
    
    max_routers = 10 if demo_mode else 30
    max_steps = 3 if demo_mode else 10
    max_reps = 1 if demo_mode else 3
    
    routers = st.number_input("Number of Routers", min_value=5, max_value=max_routers, value=8, step=1)
    steps = st.number_input("Time Steps to Simulate (Out-of-sample)", min_value=1, max_value=max_steps, value=2, step=1)
    qaoa_reps = st.number_input("QAOA Repetitions (Depth)", min_value=1, max_value=max_reps, value=1, step=1)
    
    if not demo_mode:
        st.warning("⚠️ Research Mode Enabled: High number of routers (>12) using Exact Statevector simulation may crash Streamlit Community Cloud due to memory limits (1GB).")
    else:
        st.info("Note: Running the full experiment triggers data generation, ML training, and exact Qiskit Statevector simulation. This may take several minutes depending on the topology size and QAOA reps.")
    
    submitted = st.form_submit_button("Run Experiment Pipeline")
    
if submitted:
    with st.spinner("Executing simulation pipeline... Please wait."):
        # We invoke the CLI orchestrator to ensure a fresh clean state
        main_script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'main.py')
        
        try:
            # Inject src directory into PYTHONPATH so relative imports in main.py work
            env = os.environ.copy()
            src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
            
            # We capture output to show in UI
            result = subprocess.run(
                [sys.executable, main_script, "--routers", str(routers), "--steps", str(steps)],
                capture_output=True, text=True, check=True, env=env
            )
            st.success("Experiment completed successfully!")
            with st.expander("View Logs"):
                st.code(result.stdout)
                
            st.info("Navigate to the other pages to view the updated results.")
        except subprocess.CalledProcessError as e:
            st.error("Experiment failed!")
            with st.expander("Error Logs"):
                st.code(e.stderr)
                st.code(e.stdout)
