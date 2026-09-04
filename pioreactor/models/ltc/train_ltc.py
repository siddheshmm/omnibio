"""BPTT Training Engine for Liquid Time-Constant (LTC) Yeast Bioreactor Digital Twin.

Trains the continuous-time LTC model across all chemical dosing & multimodal runs,
with masked MSE loss on valid sensor observations, gradient clipping, and stratified
validation monitoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from dataset import (
    LtcDataset,
    NormalizationScalers,
    get_stratified_splits,
    SENSOR_TARGETS,
    INPUT_COLUMNS,
)
from ltc_model import LTCBioreactorTwin

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def masked_mse_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE only over observed sensor bins safely."""
    clean_y_true = torch.where(mask, y_true, torch.zeros_like(y_true))
    clean_y_pred = torch.where(mask, y_pred, torch.zeros_like(y_pred))
    diff_sq = (clean_y_pred - clean_y_true) ** 2
    total_valid = torch.sum(mask.float())
    if total_valid > 0:
        return torch.sum(diff_sq) / total_valid
    return torch.tensor(0.0, device=y_pred.device, requires_grad=True)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        u = batch["u"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        init_od = batch["init_od"].to(device)
        init_vol = batch["init_vol"].to(device)

        optimizer.zero_grad()
        out = model(u, init_od, init_vol)
        loss = masked_mse_loss(out["y_pred"], y, mask)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        u = batch["u"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        init_od = batch["init_od"].to(device)
        init_vol = batch["init_vol"].to(device)

        out = model(u, init_od, init_vol)
        loss = masked_mse_loss(out["y_pred"], y, mask)

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LTC Yeast Digital Twin")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LTC model on device: {device}")

    # Paths
    dataset_dir = SCRIPT_DIR / cfg["paths"]["dataset_dir"]
    manifest_path = SCRIPT_DIR / cfg["paths"]["manifest_path"]
    scalers_path = SCRIPT_DIR / cfg["paths"]["scalers_path"]
    checkpoint_dir = SCRIPT_DIR / cfg["paths"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest and split runs
    df_manifest = pd.read_csv(manifest_path)
    train_keys, val_keys, bench_keys = get_stratified_splits(df_manifest, seed=cfg["training"]["seed"])

    print(f"Dataset summary:")
    print(f"  Training runs: {len(train_keys)}")
    print(f"  Validation runs: {len(val_keys)}")
    print(f"  Held-out benchmarks: {len(bench_keys)}")

    train_dfs = [pd.read_csv(dataset_dir / df_manifest.loc[df_manifest["run_key"] == k, "filename"].values[0]) for k in train_keys]
    val_dfs = [pd.read_csv(dataset_dir / df_manifest.loc[df_manifest["run_key"] == k, "filename"].values[0]) for k in val_keys]

    # 2. Fit and save scalers strictly on training runs
    scalers = NormalizationScalers.fit_from_dataframes(train_dfs)
    scalers.save(scalers_path)
    print(f"Saved normalization scalers to: {scalers_path}")

    # 3. Create datasets and dataloaders
    w_size = cfg["training"].get("window_size", 36)
    stride = cfg["training"].get("stride", 6)
    train_ds = LtcDataset(train_dfs, scalers, window_size=w_size, stride=stride)
    val_ds = LtcDataset(val_dfs, scalers, window_size=w_size, stride=stride)

    batch_size = cfg["training"].get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"Generated {len(train_ds)} train windows and {len(val_ds)} val windows ({w_size * 5} min each).")

    # 4. Build model
    m_cfg = cfg["model"]
    model = LTCBioreactorTwin(
        input_dim=m_cfg.get("input_dim", 6),
        hidden_dim=m_cfg.get("hidden_dim", 32),
        num_sensors=m_cfg.get("num_sensors", 14),
        unfolding_steps=m_cfg.get("unfolding_steps", 2),
        dt_min=m_cfg.get("dt_min", 5.0),
        tau_min=m_cfg.get("tau_min", 1.0),
        tau_max=m_cfg.get("tau_max", 60.0),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LTC model instantiated with {total_params:,} trainable parameters.")

    # 5. Optimizer
    lr = float(cfg["training"].get("learning_rate", 1e-3))
    wd = float(cfg["training"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 6. Training loop
    epochs = int(cfg["training"].get("epochs", 80))
    grad_clip = float(cfg["training"].get("grad_clip_norm", 1.0))
    best_val_loss = float("inf")
    history = []

    print("\nStarting BPTT optimization:")
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device, grad_clip=grad_clip)
        val_loss = evaluate_epoch(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
        })

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, checkpoint_dir / "ltc_best_checkpoint.pt")

    # Save final checkpoint
    torch.save({
        "epoch": epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "final_val_loss": val_loss,
        "config": cfg,
    }, checkpoint_dir / "ltc_checkpoint.pt")

    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {checkpoint_dir / 'ltc_best_checkpoint.pt'}")


if __name__ == "__main__":
    main()
