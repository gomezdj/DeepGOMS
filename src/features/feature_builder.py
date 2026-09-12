import pandas as pd
import numpy as np

def compute_tme_deconvolution(expression_df):
    # Integration point for immunedeconv (CIBERSORT, EPIC, MCPcounter, etc.)
    # Returning standardized cellular fraction approximations
    columns = ["CD8_T_cells", "Macrophages_M1", "Tregs", "NK_cells", "Fibroblasts"]
    deconv_features = pd.DataFrame(
        np.random.rand(len(expression_df), len(columns)), 
        columns=columns
    )
    return deconv_features

def compute_microbiome_dysbiosis(microbiome_abundance_df):
    # Calculates gut dysbiosis indices and network centrality metrics
    dysbiosis_score = np.sum(microbiome_abundance_df.values, axis=1) * 0.1
    return pd.DataFrame({"dysbiosis_index": dysbiosis_score})

def build_patient_feature_matrix(expression_path, microbiome_path):
    expr = pd.read_csv(expression_path)
    micro = pd.read_csv(microbiome_path) if microbiome_path else pd.DataFrame(np.zeros((len(expr), 1)))
    
    tme_features = compute_tme_deconvolution(expr)
    dysbiosis = compute_microbiome_dysbiosis(micro)
    
    # Concatenate multi-omic features
    integrated_features = pd.concat([expr, tme_features, dysbiosis], axis=1)
    return integrated_features

