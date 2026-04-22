import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import optuna

from dataset import MHeightDataset
from architectures import AdvancedResMLP
from features import get_feature_dim

DB_PATH = "data/matrices.db"

def objective(trial, n, k, m, train_loader, val_loader, input_dim, device):
    # Phase 4 Search Space
    width = trial.suggest_categorical("width", [128, 256, 512])
    depth = trial.suggest_int("depth", 3, 6)
    dropout = trial.suggest_float("dropout", 0.01, 0.05)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    
    model = AdvancedResMLP(input_dim, width, depth, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Phase 1: Train on standard MSE loss in log2 space
    criterion = nn.MSELoss()
    
    epochs = 20 # Keep short for quick hyperparam sweeps
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            
    return val_loss / len(val_loader.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    args = parser.parse_args()
    
    # Load standardized dataset
    dataset = MHeightDataset(args.n, args.k, args.m, DB_PATH, is_training=True)
    input_dim = get_feature_dim(args.n, args.k, args.m)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Optuna optimization on {device}...")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args.n, args.k, args.m, train_loader, val_loader, input_dim, device), n_trials=30)
    
    print("\n=== Best Hyperparameters Found ===")
    print(f"Best Val MSE: {study.best_value:.6f}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")