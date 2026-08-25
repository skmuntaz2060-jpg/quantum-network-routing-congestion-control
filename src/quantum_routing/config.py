import os
from pathlib import Path

def get_project_root() -> Path:
    """
    Robustly determine the project root.
    If the app is running in Streamlit, Path.cwd() is usually the project root.
    We verify by checking if 'src/quantum_routing' exists.
    """
    cwd = Path.cwd()
    if (cwd / "src" / "quantum_routing").exists() or (cwd / "pyproject.toml").exists():
        return cwd
    
    # Fallback for local scripts running directly inside the package
    src_dir = Path(__file__).resolve().parent
    return src_dir.parent.parent

PROJECT_ROOT = get_project_root()

# Centralized artifact directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
