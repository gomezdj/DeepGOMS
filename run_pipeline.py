import torch
from torch.utils.data import TensorDataset, DataLoader
from src.model.deepgoms import DeepGOMSEnsemble

def main():
    print("Starting DEEP-GOMS Training Pipeline...")
    
    # Generate dummy tensor data to simulate multi-cohort matrix loading
    X_train = torch.randn(200, 512)
    y_train = torch.randint(0, 2, (200,))
    
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize and train ensemble
    ensemble = DeepGOMSEnsemble(num_models=3, input_dim=512)
    ensemble.fit(dataloader, epochs=5, lr=0.001)
    
    # Run prediction
    X_test = torch.randn(10, 512)
    probs = ensemble.predict_proba(X_test)
    print("Sample Prediction Probabilities:", probs)

if __name__ == "__main__":
    main()
