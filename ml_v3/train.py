import sys
import os
import logging
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from dataset import MHeightDataset
from architectures import DeepSetsNet

DB_PATH = "data/matrices.db"
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def make_context(n, k, m, batch_size, device):
    ctx = torch.tensor([[float(n), float(k), float(m)]], dtype=torch.float32, device=device)
    return ctx.expand(batch_size, -1)


def evaluate(model, loader, criterion, n, k, m, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for P_batch, y_batch in loader:
            P_batch, y_batch = P_batch.to(device), y_batch.to(device)
            ctx = make_context(n, k, m, P_batch.size(0), device)
            pred = model(P_batch, ctx)
            total += criterion(pred, y_batch).item() * P_batch.size(0)
    return total / len(loader.dataset)


def train_member(seed, n, k, m, train_ds, val_ds, width, depth, dropout, device, save_dir):
    torch.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)

    model = DeepSetsNet(
        k=k,
        n_minus_k=n - k,
        col_embed_dim=128,
        head_width=width,
        head_depth=depth,
        dropout=dropout,
    ).to(device)

    criterion    = nn.MSELoss()
    optimizer    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stop   = EarlyStopping(patience=15)

    best_val  = float("inf")
    best_state = None

    for epoch in range(250):
        model.train()
        train_loss = 0.0
        for P_batch, y_batch in train_loader:
            P_batch, y_batch = P_batch.to(device), y_batch.to(device)
            ctx = make_context(n, k, m, P_batch.size(0), device)
            optimizer.zero_grad()
            pred = model(P_batch, ctx)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * P_batch.size(0)

        avg_train = train_loss / len(train_ds)
        avg_val   = evaluate(model, val_loader, criterion, n, k, m, device)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val   = avg_val
            best_state = {k_: v.cpu() for k_, v in model.state_dict().items()}

        early_stop(avg_val)
        if early_stop.early_stop:
            logger.info(f"Seed {seed} | Early stop at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            logger.info(f"Seed {seed} | Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

    save_path = os.path.join(save_dir, f"model_seed_{seed}.pth")
    torch.save(best_state, save_path)
    logger.info(f"Seed {seed} done. Best val MSE: {best_val:.6f}")
    return model, best_val


def train_combo(n, k, m, sync_time, num_seeds=3):
    is_hard = (m == n - k)
    width   = 512 if is_hard else 256
    depth   = 6   if is_hard else 4
    dropout = 0.05 if is_hard else 0.02

    full_ds = MHeightDataset(n, k, m, DB_PATH, is_training=True)
    if len(full_ds) == 0:
        logger.error(f"No data for n={n}, k={k}, m={m}")
        return

    train_size = int(0.8 * len(full_ds))
    val_size   = int(0.1 * len(full_ds))
    test_size  = len(full_ds) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        full_ds, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.is_training = False

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = f"model/v3_n{n}_k{k}_m{m}_{sync_time}"
    os.makedirs(save_dir, exist_ok=True)

    models = []
    for seed in range(num_seeds):
        logger.info(f"--- Training seed {seed + 1}/{num_seeds} for n={n}, k={k}, m={m} ---")
        model, _ = train_member(seed, n, k, m, train_ds, val_ds, width, depth, dropout, device, save_dir)
        models.append(model)

    # Evaluate ensemble on held-out test set
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    criterion   = nn.MSELoss()
    total       = 0.0

    for model in models:
        model.eval()

    with torch.no_grad():
        for P_batch, y_batch in test_loader:
            P_batch, y_batch = P_batch.to(device), y_batch.to(device)
            ctx = make_context(n, k, m, P_batch.size(0), device)
            preds = torch.stack([m_model(P_batch, ctx) for m_model in models]).mean(dim=0)
            total += criterion(preds, y_batch).item() * P_batch.size(0)

    logger.info(f"=== Ensemble test MSE for n={n}, k={k}, m={m}: {total / len(test_ds):.6f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--sync_time", type=str, required=True)
    args = parser.parse_args()

    log_dir = f"logs/v3_n{args.n}_k{args.k}_m{args.m}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{args.sync_time}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )

    train_combo(args.n, args.k, args.m, args.sync_time)
