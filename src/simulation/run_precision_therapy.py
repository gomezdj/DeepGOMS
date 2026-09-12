import torch
from src.model.microbiome_therapy import TherapeuticMicrobiomeEngine
from src.simulation.precision_intervention import simulate_precision_intervention

def main():
    print("Initializing DEEP-GOMS Precision Microbiome & Metabolic Tuning Module...")
    
    # Instantiate engine
    engine = TherapeuticMicrobiomeEngine(num_taxa=150, metabolic_pathways=45)
    engine.eval()
    
    # Mock patient baseline features
    dummy_microbiome = torch.rand(1, 150)
    dummy_metabolites = torch.rand(1, 45)
    
    with torch.no_grad():
        optimized_taxa, phenotypes = engine(dummy_microbiome, dummy_metabolites)
        
    print("\n--- Phenotypic Optimization Outputs ---")
    print(f"Immunotherapy Response Probability: {phenotypes[0][0].item():.4f}")
    print(f"Marathon Endurance Optimization Score: {phenotypes[0][1].item():.4f}")
    print(f"Conception Success Index: {phenotypes[0][2].item():.4f}")
    
    # Run precise intervention report
    report = simulate_precision_intervention("PATIENT_001", optimized_taxa, target_goal="marathon")
    print("\n--- Precision Intervention Prescription ---")
    for k, v in report.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
