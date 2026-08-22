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
    routers = st.number_input("Number of Routers", min_value=5, max_value=50, value=8, step=1)
    steps = st.number_input("Time Steps to Simulate (Out-of-sample)", min_value=1, max_value=20, value=2, step=1)
    qaoa_reps = st.number_input("QAOA Repetitions (Depth)", min_value=1, max_value=5, value=1, step=1)
    
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
                ["python", main_script, "--routers", str(routers), "--steps", str(steps)],
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
