# coordinator_agents.py
import os
import subprocess
import sys
import json

# Base folder (works in notebooks too)
BASE_DIR = os.getcwd()
SHARED = os.path.join(BASE_DIR, "shared_data")
os.makedirs(SHARED, exist_ok=True)

ONTOLOGY_PATH = os.path.join(BASE_DIR, "ontology.json")

def load_ontology():
    """Load shared ontology for all agents."""
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_agent(script_name):
    """Run a single agent script."""
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_name} not found in {BASE_DIR}")
    
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with code {result.returncode}")
    print(f"{script_name} finished successfully.\n")

def main():
    ontology = load_ontology()
    print("Coordinator loaded ontology.")
    print("Starting multi-agent pipeline...\n")

    # List of agents in order
    agents = [
        "agent_1.py",
        "agent_2.py",
        "agent_3.py",
        "agent_4.py",
        "agent_5.py"
    ]

    for agent in agents:
        run_agent(agent)

    print("Pipeline finished! All outputs are in the 'shared_data' folder.")

if __name__ == "__main__":
    main()
