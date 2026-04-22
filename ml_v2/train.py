import sys
import os
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import MHeightDataset
from architectures import AdvancedResMLP
from features import get_feature_dim

DB_PATH = "data/matrices.db"
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0

def train_ensemble_member(seed, n, k, m, train_ds, val_ds, input_dim, width, depth, dropout, device, save_dir):
    torch.manual_seed(seed)
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    model = AdvancedResMLP(input_dim=input_dim, width=width, depth=depth, dropout=dropout).to(device)
    
    # Phase 1: MSE Loss in log2 space, and LR Scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Train up to 250 epochs
    epochs = 250 
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        avg_train = train_loss / len(train_ds)
        avg_val = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = model.state_dict()
            
        early_stopping(avg_val)
        if early_stopping.early_stop:
            logger.info(f"Seed {seed} | Early stopping triggered at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            logger.info(f"Seed {seed} - Epoch {epoch+1:03d} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}")

    # Save the best model for this seed
    save_path = os.path.join(save_dir, f"model_seed_{seed}.pth")
    torch.save(best_model_state, save_path)
    logger.info(f"Seed {seed} finished. Best Val MSE: {best_val_loss:.6f}")
    
    return model, best_val_loss

def train_combo(n, k, m, sync_time, num_seeds=3, is_hard_combo=False):
    # Phase 4 Strategy: Use larger models for hard combinations
    width = 512 if is_hard_combo else 256
    depth = 6 if is_hard_combo else 4
    dropout = 0.05 if is_hard_combo else 0.02
    
    full_dataset = MHeightDataset(n, k, m, DB_PATH, is_training=True)
    if len(full_dataset) == 0:
        logger.error(f"No data for n={n}, k={k}, m={m}")
        return

    input_dim = get_feature_dim(n, k, m)
    logger.info(f"Feature Dimension for ({n}, {k}, {m}): {input_dim}")

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    # Disable augmentations on val/test sets by modifying their underlying dataset reference flag
    # (A cleaner PyTorch way is creating separate datasets, but this is a quick patch for random_split)
    val_ds.dataset.is_training = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = f"model/ensemble_n{n}_k{k}_m{m}_{sync_time}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Phase 5: Train an ensemble of N models
    models = []
    for seed in range(num_seeds):
        logger.info(f"--- Training Ensemble Member {seed+1}/{num_seeds} ---")
        model, _ = train_ensemble_member(seed, n, k, m, train_ds, val_ds, input_dim, width, depth, dropout, device, save_dir)
        models.append(model)
        
    # Evaluate Ensemble on Test Set
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    for m_idx, model in enumerate(models):
        model.eval()
        
    total_ensemble_mse = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Test-Time Augmentation (TTA) / Ensemble Averaging
            predictions = torch.zeros_like(y_batch)
            for model in models:
                predictions += model(X_batch)
            ensemble_pred = predictions / num_seeds
            
            # Phase 5: Ensure bounds (transforming back to raw space logic for checks, though calculating MSE in log space)
            # actual_height_pred = torch.clamp(2 ** ensemble_pred, min=1.0)
            
            loss = criterion(ensemble_pred, y_batch)
            total_ensemble_mse += loss.item() * X_batch.size(0)

    final_test = total_ensemble_mse / len(test_ds)
    logger.info(f"=== Final ENSEMBLE Test MSE for n={n}, k={k}, m={m}: {final_test:.6f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train separate models with ensemble.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--sync_time", type=str, required=True)
    args = parser.parse_args()
    
    log_sub_dir = f"logs/ensemble_n{args.n}_k{args.k}_m{args.m}"
    os.makedirs(log_sub_dir, exist_ok=True)
    log_path = os.path.join(log_sub_dir, f"train_{args.sync_time}.log")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    
    is_hard = (args.m == args.n - args.k)
    train_combo(args.n, args.k, args.m, args.sync_time, num_seeds=3, is_hard_combo=is_hard)