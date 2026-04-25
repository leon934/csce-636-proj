import os
import logging
import subprocess
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    import psutil
    PSUTIL = True
except ImportError:
    PSUTIL = False

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False

DB_PATH     = "data/matrices.db"
SAVE_ROOT   = "model"
BATCH_SIZE  = 512
NUM_WORKERS = min(8, os.cpu_count() or 4)
MODEL_TYPE  = "DeepSetsNet_v2"
logger      = logging.getLogger(__name__)


def augment_batch(P: torch.Tensor, n_minus_k: int) -> torch.Tensor:
    perm = torch.randperm(n_minus_k, device=P.device)
    P = P[:, :, perm]
    signs = torch.randint(0, 2, (1, 1, n_minus_k), device=P.device).float() * 2 - 1
    return P * signs


def gpu_stats() -> str:
    parts = []
    if torch.cuda.is_available():
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,utilization.memory",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            ).decode().strip()
            gpu_util, mem_util = out.split(", ")
            parts.append(f"GPU {gpu_util}% | mem util {mem_util}%")
        except Exception:
            pass
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        parts.append(f"VRAM {alloc:.1f}/{total:.1f} GB")
    if PSUTIL:
        parts.append(f"CPU {psutil.cpu_percent(interval=None):.0f}%")
        parts.append(f"RAM {psutil.virtual_memory().used / 1e9:.1f} GB")
    return "  ".join(parts) if parts else "no stats"


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
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
            P_batch = P_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            ctx  = make_context(n, k, m, P_batch.size(0), device)
            pred = model(P_batch, ctx)
            total += criterion(pred, y_batch).item() * P_batch.size(0)
    return total / len(loader.dataset)


def train_member(seed, n, k, m, train_ds, val_ds, width, depth, dropout, device, save_dir, sync_time):
    torch.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = DeepSetsNet(
        k=k, n_minus_k=n - k, col_embed_dim=128,
        head_width=width, head_depth=depth, dropout=dropout,
    ).to(device)

    criterion  = nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stop = EarlyStopping(patience=15)

    best_val  = float("inf")
    save_path = os.path.join(save_dir, f"model_seed_{seed}.pth")

    run = None
    if WANDB:
        run = wandb.init(
            project="636-mheight",
            name=f"{MODEL_TYPE}_n{n}_k{k}_m{m}_seed{seed}_{sync_time}",
            config={
                "model": MODEL_TYPE,
                "n": n, "k": k, "m": m, "seed": seed,
                "width": width, "depth": depth, "dropout": dropout,
                "batch_size": BATCH_SIZE, "sync_time": sync_time,
            },
            reinit=True,
        )

    for epoch in range(250):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for P_batch, y_batch in train_loader:
            P_batch = P_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            P_batch = augment_batch(P_batch, n - k)
            ctx     = make_context(n, k, m, P_batch.size(0), device)
            optimizer.zero_grad()
            loss = criterion(model(P_batch, ctx), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * P_batch.size(0)

        epoch_secs = time.time() - t0
        avg_train  = train_loss / len(train_ds)
        avg_val    = evaluate(model, val_loader, criterion, n, k, m, device)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save({k_: v.cpu() for k_, v in model.state_dict().items()}, save_path)
            logger.info(f"Seed {seed} | Epoch {epoch+1:03d} | New best — saved ({best_val:.6f})")

        if WANDB and run is not None:
            run.log({
                "train_mse":  avg_train,
                "val_mse":    avg_val,
                "best_val":   best_val,
                "epoch_secs": epoch_secs,
                "lr":         optimizer.param_groups[0]["lr"],
            }, step=epoch)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Seed {seed} | Epoch {epoch+1:03d} | "
                f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                f"Best: {best_val:.6f} | {epoch_secs:.1f}s | {gpu_stats()}"
            )

        early_stop(avg_val)
        if early_stop.early_stop:
            logger.info(f"Seed {seed} | Early stop at epoch {epoch + 1}")
            break

    logger.info(f"Seed {seed} done. Best val MSE: {best_val:.6f}")

    if WANDB and run is not None:
        run.summary["best_val_mse"] = best_val
        run.finish()

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, best_val


def train_combo(n, k, m, sync_time, num_seeds=3, save_root=SAVE_ROOT):
    is_hard = (m == n - k)
    width   = 512 if is_hard else 256
    depth   = 6   if is_hard else 4
    dropout = 0.05 if is_hard else 0.02

    full_ds = MHeightDataset(n, k, m, DB_PATH)
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

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(save_root, MODEL_TYPE, f"n{n}_k{k}_m{m}_{sync_time}")
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "train.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    batches_per_epoch = (train_size + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        f"n={n}, k={k}, m={m} | device={device} | batch={BATCH_SIZE} | workers={NUM_WORKERS} | "
        f"train={train_size:,} val={val_size:,} test={test_size:,} | "
        f"~{batches_per_epoch:,} batches/epoch | {gpu_stats()}"
    )

    models = []
    for seed in range(num_seeds):
        logger.info(f"--- Training seed {seed + 1}/{num_seeds} for n={n}, k={k}, m={m} ---")
        model, _ = train_member(
            seed, n, k, m, train_ds, val_ds, width, depth, dropout, device, save_dir, sync_time
        )
        models.append(model)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    criterion = nn.MSELoss()
    total     = 0.0

    for model in models:
        model.eval()

    with torch.no_grad():
        for P_batch, y_batch in test_loader:
            P_batch = P_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            ctx   = make_context(n, k, m, P_batch.size(0), device)
            preds = torch.stack([m_model(P_batch, ctx) for m_model in models]).mean(dim=0)
            total += criterion(preds, y_batch).item() * P_batch.size(0)

    ensemble_mse = total / len(test_ds)
    logger.info(f"=== Ensemble test MSE for n={n}, k={k}, m={m}: {ensemble_mse:.6f} ===")

    logger.removeHandler(fh)
    fh.close()


def predict(n, k, m, P_matrix, models, device=None):
    """
    Run ensemble inference on a single P matrix and return the predicted m-height.

    P_matrix : numpy array of shape (k, n-k) with integer values in [-100, 100]
    models   : list of trained DeepSetsNet instances (one per ensemble seed)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    P = torch.tensor(P_matrix, dtype=torch.float32) / 100.0
    P = P.unsqueeze(0).to(device)
    ctx = make_context(n, k, m, 1, device)

    for model in models:
        model.eval()

    with torch.no_grad():
        log_pred = torch.stack([m(P, ctx) for m in models]).mean(dim=0).item()

    return max(1.0, 2 ** log_pred)
