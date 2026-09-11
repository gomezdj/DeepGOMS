import os
import yaml
import pandas as pd
import numpy as np

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_and_preprocess_cohorts():
    config = load_config()
    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    
    print("Initializing cohort data ingestion pipeline...")
    for cohort in config["data"]["cohorts"]:
        print(f"Processing cohort: {cohort}")
        # Simulated placeholder for downloading and normalizing matrix data (e.g., via UCSCXenaTools / GEO)
        mock_expression_matrix = pd.DataFrame(
            np.random.randn(100, config["model"]["input_dim"]),
            columns=[f"gene_{i}" for i in range(config["model"]["input_dim"])]
        )
        mock_metadata = pd.DataFrame({
            "patient_id": [f"{cohort}_pat_{i}" for i in range(100)],
            "cohort": cohort,
            "response": np.random.randint(0, 2, size=100)
        })
        
        # Save processed outputs
        mock_expression_matrix.to_csv(f"{config['data']['processed_dir']}_{cohort}_expression.csv", index=False)
        mock_metadata.to_csv(f"{config['data']['processed_dir']}_{cohort}_metadata.csv", index=False)

if __name__ == "__main__":
    download_and_preprocess_cohorts()
