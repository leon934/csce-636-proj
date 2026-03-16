import sys
import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import MHeightDataset
import architectures

DB_PATH = "data/matrices.db"
logger = logging.getLogger(__name__)

class MHeightModelWrapper(nn.Module):
    def __init__(self, backbone):
        super(MHeightModelWrapper, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        raw_output = self.backbone(x)
        return 1.0 + F.relu(raw_output)

def custom_log2_cost(y_pred, y_true):
    return torch.mean((torch.log2(y_true) - torch.log2(y_pred)) ** 2)

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = custom_log2_cost(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0

def train_model(n, k, m, arch_name):
    dataset = MHeightDataset(n, k, m, DB_PATH)
    if len(dataset) == 0:
        logger.error(f"No data for n={n}, k={k}, m={m}")
        return

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        BackboneClass = getattr(architectures, arch_name)
    except AttributeError:
        logger.error(f"Architecture '{arch_name}' not found.")
        return
        
    backbone = BackboneClass(n=n, k=k) 
    model = MHeightModelWrapper(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(20):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = custom_log2_cost(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        avg_train = train_loss / train_size
        avg_val = evaluate_model(model, val_loader, device)
        logger.info(f"Epoch {epoch+1:02d} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

    final_test = evaluate_model(model, test_loader, device)
    logger.info(f"Final Test Loss for {arch_name}: {final_test:.6f}")

    save_path = f"model/n{n}_k{k}_m{m}/{arch_name}_n{n}_k{k}_m{m}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train standard standard models.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--sync_time", type=str, required=True)
    parser.add_argument("--arch", type=str, required=True)
    args = parser.parse_args()
    
    log_sub_dir = f"logs/n{args.n}_k{args.k}_m{args.m}"
    os.makedirs(log_sub_dir, exist_ok=True)
    log_path = os.path.join(log_sub_dir, f"{args.arch}_{args.sync_time}.log")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    
    train_model(args.n, args.k, args.m, args.arch)