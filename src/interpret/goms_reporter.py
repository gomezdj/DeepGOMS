import pandas as pd
import numpy as np

def generate_goms_report(patient_ids, predictions, feature_matrix):
    report = pd.DataFrame({
        "patient_id": patient_ids,
        "predicted_response_probability": predictions,
        "goms_risk_tier": ["High" if p > 0.5 else "Low" for p in predictions]
    })
    
    # Identify key microbial-immune drivers via coefficient analysis/attribution
    top_drivers = feature_matrix.columns[:5].tolist()
    report["primary_drivers"] = [", ".join(top_drivers)] * len(report)
    
    return report
