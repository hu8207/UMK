import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def to_device(data, device):
    """Move data to device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

class UsadModel(nn.Module):
    """USAD Model implementation"""
    def __init__(self, w_size, z_size):
        super(UsadModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(w_size, w_size//2),
            nn.ReLU(),
            nn.Linear(w_size//2, z_size),
            nn.ReLU()
        )
        
        self.decoder1 = nn.Sequential(
            nn.Linear(z_size, w_size//2),
            nn.ReLU(),
            nn.Linear(w_size//2, w_size)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Linear(z_size, w_size//2),
            nn.ReLU(),
            nn.Linear(w_size//2, w_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded1 = self.decoder1(encoded)
        decoded2 = self.decoder2(encoded)
        return decoded1, decoded2, encoded

def training(n_epochs, model, train_loader, test_loader):
    """Training function for USAD"""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    roc_max = 0
    ap_max = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (x, x_aug, label, idx) in enumerate(train_loader):
            x = x.to(device).view(x.size(0), -1)
            x_aug = x_aug.to(device).view(x_aug.size(0), -1)
            
            optimizer.zero_grad()
            
            # Forward pass
            decoded1, decoded2, encoded = model(x)
            decoded1_aug, decoded2_aug, encoded_aug = model(x_aug)
            
            # USAD loss
            loss1 = torch.mean((x - decoded1) ** 2)
            loss2 = torch.mean((x - decoded2) ** 2)
            
            # Combined loss (simplified USAD loss)
            alpha = 0.5  # Can be scheduled
            loss = alpha * loss1 + (1 - alpha) * loss2
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Testing
        model.eval()
        test_losses = []
        test_labels = []
        
        with torch.no_grad():
            for x, x_aug, label, idx in test_loader:
                x = x.to(device).view(x.size(0), -1)
                
                decoded1, decoded2, encoded = model(x)
                
                # Anomaly score (reconstruction error)
                recon_error = torch.mean((x - decoded1) ** 2, dim=1)
                test_losses.extend(recon_error.cpu().numpy())
                test_labels.extend(label.numpy())
        
        # Calculate metrics
        test_losses = np.array(test_losses)
        test_labels = np.array(test_labels)
        
        if len(np.unique(test_labels)) > 1:  # Check if we have both normal and anomaly
            roc_test = roc_auc_score(test_labels, test_losses)
            ap_test = average_precision_score(test_labels, test_losses)
            
            if roc_test > roc_max:
                roc_max = roc_test
                ap_max = ap_test
            
            print(f'Epoch {epoch+1}/{n_epochs}, ROC: {roc_test:.4f}, AP: {ap_test:.4f}, Best ROC: {roc_max:.4f}')
        else:
            print(f'Epoch {epoch+1}/{n_epochs}, No anomalies in test set')
    
    return roc_max, ap_max
