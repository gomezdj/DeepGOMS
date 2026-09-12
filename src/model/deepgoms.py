import torch
import torch.nn as nn
import torch.nn.functional as F

class DEEPGOMS(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(DEEPGOMS, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return torch.sigmoid(logits)

class DeepGOMSEnsemble:
    def __init__(self, num_models=5, input_dim=512):
        self.models = [DEEPGOMS(input_dim=input_dim) for _ in range(num_models)]
        
    def fit(self, dataloader, epochs=10, lr=0.001):
        # Evolutionary or standard bagging ensemble training loop across sub-models
        criterion = nn.BCELoss()
        for idx, model in enumerate(self.models):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y.float())
                    loss.backward()
                    optimizer.step()

    def predict_proba(self, x):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds = model(x)
                predictions.append(preds.numpy())
        return np.mean(predictions, axis=0)
