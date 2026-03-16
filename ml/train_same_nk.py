import sys
import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import CombinedMHeightDataset
import architectures

DB_PATH = "data/matrices.db"
logger = logging.getLogger(__name__)

class TheoryModelWrapper(nn.Module):
    def __init__(self, backbone):
        super(TheoryModelWrapper, self).__init__()
        self.backbone = backbone

    def forward(self, x, m_val):
        raw_output = self.backbone(x, m_val)
        return 1.0 + F.relu(raw_output)

def theory_informed_loss(model, X_batch, m_batch, y_true, lambda_penalty=0.5):
    y_pred = model(X_batch, m_batch)
    base_cost = torch.mean((torch.log2(y_true) - torch.log2(y_pred)) ** 2)
    
    m_minus_1_batch = torch.clamp(m_batch - 1, min=0) 
    y_pred_prev = model(X_batch, m_minus_1_batch)
    
    inversion_penalty = torch.mean(F.relu(y_pred_prev - y_pred))
    return base_cost + (lambda_penalty * inversion_penalty)

def evaluate_theory_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, m_batch, y_batch in dataloader:
            X_batch, m_batch, y_batch = X_batch.to(device), m_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch, m_batch)
            # Eval just uses standard cost, not the penalty
            loss = torch.mean((torch.log2(y_batch) - torch.log2(y_pred)) ** 2)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0

def train_theory_model(n, k, arch_name):
    dataset = CombinedMHeightDataset(n, k, DB_PATH)
    if len(dataset) == 0:
        logger.error(f"No data for n={n}, k={k}")
        return

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        BackboneClass = getattr(architectures, arch_name)
    except AttributeError:
        logger.error(f"Architecture '{arch_name}' not found.")
        return
        
    backbone = BackboneClass(n=n, k=k) 
    model = TheoryModelWrapper(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(20):
        model.train()
        train_loss = 0.0
        for X_batch, m_batch, y_batch in train_loader:
            X_batch, m_batch, y_batch = X_batch.to(device), m_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            # Use custom theory loss for training
            loss = theory_informed_loss(model, X_batch, m_batch, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        avg_train = train_loss / train_size
        avg_val = evaluate_theory_model(model, val_loader, device)
        logger.info(f"Epoch {epoch+1:02d} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

    final_test = evaluate_theory_model(model, test_loader, device)
    logger.info(f"Final Test Loss for {arch_name}: {final_test:.6f}")

    # Save format reflects the combined nature of the model
    save_path = f"model/n{n}_k{k}_theory/{arch_name}_n{n}_k{k}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train theory-informed models.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--sync_time", type=str, required=True)
    parser.add_argument("--arch", type=str, required=True)
    args = parser.parse_args()
    
    log_sub_dir = f"logs/n{args.n}_k{args.k}_theory"
    os.makedirs(log_sub_dir, exist_ok=True)
    log_path = os.path.join(log_sub_dir, f"{args.arch}_{args.sync_time}.log")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    
    train_theory_model(args.n, args.k, args.arch)