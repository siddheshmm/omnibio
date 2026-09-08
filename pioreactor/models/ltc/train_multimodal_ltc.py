"""Comprehensive Multimodal Training Script for Liquid Time-Constant (LTC) Yeast Digital Twin.

Trains the continuous-time LTC model using ALL available experimental runs (39 runs total):
  1. Pulse Experiments (26 runs):
     - Chemical dosing: glucose (x3), nitrogen (x5), salt (x5), sulfur (x5), control (x1), uracil (x1)
     - Temperature modulation (x3: 20260904085950, 20260904090547, 20260904091054)
     - UV irradiation (x3: 20260905150354, 20260905150930, 20260905151517)
  2. Sine Wave Encoding (7 runs):
     - Chemical dosing (x5: glucose, nitro, salt x2, sulfur)
     - Temperature (x1)
     - UV (x1)
  3. Mackey-Glass Encoding (6 runs):
     - Chemical dosing (x4: glucose, nitro, salt, sulfur)
     - Temperature (x1)
     - UV (x1)

Evaluation:
  - Stratified validation set (6 runs covering glucose, nitrogen, salt, sulfur, temp, uv)
  - Unseen reservoir computing evaluation on continuous 6.8h Mackey-Glass and 12.1h 4D Rössler runs
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
    """Safe masked MSE ignoring unobserved bins without NaN propagation."""
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
    parser = argparse.ArgumentParser(description="Train Multimodal LTC Yeast Digital Twin")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Multimodal LTC model on device: {device}", flush=True)

    # Paths
    dataset_dir = SCRIPT_DIR / cfg["paths"]["dataset_dir"]
    manifest_path = SCRIPT_DIR / cfg["paths"]["manifest_path"]
    scalers_path = SCRIPT_DIR / "artifacts" / "dataset" / "scalers_multimodal.json"
    checkpoint_dir = SCRIPT_DIR / cfg["paths"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest and create stratified multimodal split
    df_manifest = pd.read_csv(manifest_path)
    train_keys, val_keys, _ = get_stratified_splits(df_manifest, seed=cfg["training"]["seed"], include_all_modalities=True)

    print(f"\nMultimodal Dataset Ingestion Summary:", flush=True)
    print(f"  Total experimental runs: {len(df_manifest)}", flush=True)
    print(f"  Training runs: {len(train_keys)}", flush=True)
    print(f"  Validation runs: {len(val_keys)} ({val_keys})", flush=True)

    train_dfs = [pd.read_csv(dataset_dir / df_manifest.loc[df_manifest["run_key"] == k, "filename"].values[0]) for k in train_keys]
    val_dfs = [pd.read_csv(dataset_dir / df_manifest.loc[df_manifest["run_key"] == k, "filename"].values[0]) for k in val_keys]

    # 2. Fit and save scalers
    scalers = NormalizationScalers.fit_from_dataframes(train_dfs)
    scalers.save(scalers_path)
    print(f"Saved multimodal normalization scalers to: {scalers_path}", flush=True)

    # 3. Create datasets and dataloaders
    w_size = cfg["training"].get("window_size", 36)
    stride = cfg["training"].get("stride", 6)
    train_ds = LtcDataset(train_dfs, scalers, window_size=w_size, stride=stride)
    val_ds = LtcDataset(val_dfs, scalers, window_size=w_size, stride=stride)

    batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"Generated {len(train_ds)} train windows and {len(val_ds)} val windows ({w_size * 5} min each).", flush=True)

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
    print(f"Multimodal LTC model instantiated with {total_params:,} trainable parameters.", flush=True)

    # 5. Optimizer
    lr = args.lr
    wd = float(cfg["training"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 6. Training loop
    epochs = args.epochs
    grad_clip = float(cfg["training"].get("grad_clip_norm", 1.0))
    best_val_loss = float("inf")
    history = []

    best_checkpoint_path = checkpoint_dir / "ltc_multimodal_best_checkpoint.pt"
    final_checkpoint_path = checkpoint_dir / "ltc_multimodal_final_checkpoint.pt"

    print("\nStarting Multimodal BPTT optimization:", flush=True)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
                "scalers_path": str(scalers_path),
            }, best_checkpoint_path)

    torch.save({
        "epoch": epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "final_val_loss": val_loss,
        "config": cfg,
        "scalers_path": str(scalers_path),
    }, final_checkpoint_path)

    with open(checkpoint_dir / "multimodal_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nMultimodal training completed. Best validation loss: {best_val_loss:.4f}", flush=True)
    print(f"Best checkpoint saved to: {best_checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
