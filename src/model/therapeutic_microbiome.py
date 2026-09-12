import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class TherapeuticMicrobiomeEngine(nn.Module):
    """
    Simulates the addition of therapeutic microbial taxa and dietary/metabolic 
    interventions to optimize response rates, athletic endurance (marathon profile), 
    and systemic metabolic fitness.
    """
    def __init__(self, num_taxa=150, metabolic_pathways=45):
        super(TherapeuticMicrobiomeEngine, self).__init__()
        self.num_taxa = num_taxa
        self.metabolic_pathways = metabolic_pathways
        
        # Generator for synthetic therapeutic consortium adjustments
        self.consortia_encoder = nn.Sequential(
            nn.Linear(num_taxa + metabolic_pathways, 128),
            nn.ReLU(),
            nn.Linear(128, num_taxa)
        )
        
        # Phenotypic success evaluator (Marathon recovery, Immunotherapy Response, Conception success)
        self.phenotype_head = nn.Sequential(
            nn.Linear(num_taxa, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # [Immunotherapy Response, Marathon Endurance Score, Conception Probability Index]
        )

    def forward(self, baseline_microbiome, baseline_metabolites):
        combined_input = torch.cat([baseline_microbiome, baseline_metabolites], dim=-1)
        # Predict optimal microbial abundance shifts after therapeutic augmentation
        delta_taxa = torch.tanh(self.consortia_encoder(combined_input))
        optimized_microbiome = torch.clamp(baseline_microbiome + delta_taxa, min=0.0, max=1.0)
        
        phenotype_scores = torch.sigmoid(self.phenotype_head(optimized_microbiome))
        return optimized_microbiome, phenotype_scores
