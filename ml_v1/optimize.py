import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import optuna

# Import your dataset
from dataset import MHeightDataset

DB_PATH = "data/matrices.db"

# 1. The Dynamic Architecture
class Conv1dResBlock(nn.Module):
    def __init__(self, channels):
        super(Conv1dResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return F.relu(out)

class DynamicResColumnCNN(nn.Module):
    def __init__(self, n, k, width, depth, pool_type):
        super(DynamicResColumnCNN, self).__init__()
        self.n = n
        self.k = k
        self.cols = n - k
        
        self.init_conv = nn.Conv1d(in_channels=k, out_channels=width, kernel_size=3, padding=1)
        
        # Stack blocks dynamically based on Optuna's 'depth' suggestion
        self.res_blocks = nn.ModuleList([Conv1dResBlock(width) for _ in range(depth)])
        
        # Dynamic pooling
        if pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(2)
        else:
            self.pool = nn.AdaptiveAvgPool1d(2)
            
        self.fc1 = nn.Linear(width * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, self.k, self.cols)
        x = F.relu(self.init_conv(x))
        
        for block in self.res_blocks:
            x = block(x)
            
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MHeightModelWrapper(nn.Module):
    def __init__(self, backbone):
        super(MHeightModelWrapper, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return 1.0 + F.relu(self.backbone(x))

def custom_log2_cost(y_pred, y_true):
    return torch.mean((torch.log2(y_true) - torch.log2(y_pred)) ** 2)

# 2. The Optuna Objective Function
def objective(trial, n, k, m, train_loader, val_loader, device):
    # Suggest Hyperparameters
    width = trial.suggest_categorical("width", [32, 64, 128, 256, 512])
    depth = trial.suggest_int("depth", 1, 5) # Test 1 to 5 ResBlocks
    pool_type = trial.suggest_categorical("pool_type", ["avg", "max"])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    
    # Initialize dynamic model
    backbone = DynamicResColumnCNN(n, k, width, depth, pool_type)
    model = MHeightModelWrapper(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train for a shorter period to find good parameters quickly (e.g., 10 epochs)
    epochs = 20
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = custom_log2_cost(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
    # Evaluate on Validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = custom_log2_cost(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            
    avg_val_loss = val_loss / len(val_loader.dataset)
    return avg_val_loss

# 3. Execution Setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    args = parser.parse_args()
    
    # Setup Data
    dataset = MHeightDataset(args.n, args.k, args.m, DB_PATH)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    # Batch size 512 for fast sweeping
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Optuna optimization on {device}...")

    # Create Optuna Study
    study = optuna.create_study(direction="minimize")
    
    # Optimize! (Runs 30 different configurations)
    study.optimize(lambda trial: objective(trial, args.n, args.k, args.m, train_loader, val_loader, device), n_trials=30)
    
    print("\n=== Best Hyperparameters Found ===")
    print(f"Best Val Loss: {study.best_value:.6f}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")