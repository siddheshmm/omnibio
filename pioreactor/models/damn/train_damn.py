"""Train the continuous Dynamic Artificial Metabolic Network (dAMN) model with PyTorch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from damn_ode import DAMN
from dataset import load_dataset_runs, NormalizationScalers

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "model"


def train_epoch(
    model: DAMN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_stoich: float = 0.1,
    lambda_smooth: float = 0.01,
) -> tuple[float, float, float]:
    model.train()
    total_loss, total_sensor_loss, total_stoich_loss = 0.0, 0.0, 0.0
    num_batches = 0

    for batch in loader:
        u = batch["u"]          # (batch, W, 4)
        y = batch["y"]          # (batch, W, 14)
        mask = batch["mask"]    # (batch, W, 14)
        v0 = batch["init_volume"].unsqueeze(1).to(dtype=torch.float32)
        init_biomass = (batch["init_od"].unsqueeze(1) * 0.15).to(dtype=torch.float32)

        optimizer.zero_grad()

        # Forward integration
        out = model(u_trajectory=u, init_biomass=init_biomass, init_volume=v0)
        y_pred = out["y_pred"]
        stoich_loss = out["stoich_loss"]
        fluxes = out["fluxes"]

        # Masked sensor reconstruction loss (MSE on observed entries)
        diff_sq = (y_pred - y) ** 2
        sensor_loss = torch.sum(diff_sq * mask) / torch.clamp(mask.sum(), min=1.0)

        # Flux smoothness regularization: ||v_{t+1} - v_t||^2
        flux_diff = fluxes[:, 1:, :] - fluxes[:, :-1, :]
        smooth_loss = torch.mean(flux_diff ** 2)

        loss = sensor_loss + (lambda_stoich * stoich_loss) + (lambda_smooth * smooth_loss)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        total_sensor_loss += sensor_loss.item()
        total_stoich_loss += stoich_loss.item()
        num_batches += 1

    return (
        total_loss / max(num_batches, 1),
        total_sensor_loss / max(num_batches, 1),
        total_stoich_loss / max(num_batches, 1),
    )


@torch.no_grad()
def evaluate_loss(
    model: DAMN,
    loader: DataLoader,
    lambda_stoich: float = 0.1,
) -> tuple[float, float, float]:
    model.eval()
    total_loss, total_sensor_loss, total_stoich_loss = 0.0, 0.0, 0.0
    num_batches = 0

    for batch in loader:
        u = batch["u"]
        y = batch["y"]
        mask = batch["mask"]
        v0 = batch["init_volume"].unsqueeze(1).to(dtype=torch.float32)
        init_biomass = (batch["init_od"].unsqueeze(1) * 0.15).to(dtype=torch.float32)

        out = model(u_trajectory=u, init_biomass=init_biomass, init_volume=v0)
        y_pred = out["y_pred"]
        stoich_loss = out["stoich_loss"]

        diff_sq = (y_pred - y) ** 2
        sensor_loss = torch.sum(diff_sq * mask) / torch.clamp(mask.sum(), min=1.0)
        loss = sensor_loss + (lambda_stoich * stoich_loss)

        total_loss += loss.item()
        total_sensor_loss += sensor_loss.item()
        total_stoich_loss += stoich_loss.item()
        num_batches += 1

    return (
        total_loss / max(num_batches, 1),
        total_sensor_loss / max(num_batches, 1),
        total_stoich_loss / max(num_batches, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.config.open() as handle:
        config = yaml.safe_load(handle)

    train_ds, test_ds, scalers = load_dataset_runs(config_path=args.config)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(train_ds)} train windows and {len(test_ds)} test windows across {len(scalers.sensor_names)} sensors.")

    # Instantiate model
    amn_cfg = config.get("amn", {})
    model = DAMN(
        num_sensors=len(scalers.sensor_names),
        num_metabolites=6,
        input_dim=len(scalers.input_names),
        latent_dim=int(amn_cfg.get("latent_dim", 8)),
        hidden_dim=int(amn_cfg.get("hidden_dim", 64)),
        yield_glucose=0.0811,
        dt_min=float(config.get("bin_minutes", 5.0)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    lambda_stoich = float(amn_cfg.get("stoichiometric_penalty_weight", 0.1))
    lambda_smooth = float(amn_cfg.get("flux_smoothness_weight", 0.01))

    best_val_loss = float("inf")
    history = []
    checkpoint_path = args.output_dir / "damn_checkpoint.pt"

    print(f"\nStarting dAMN training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_sens, tr_stoich = train_epoch(
            model, train_loader, optimizer, lambda_stoich=lambda_stoich, lambda_smooth=lambda_smooth
        )
        val_loss, val_sens, val_stoich = evaluate_loss(
            model, test_loader, lambda_stoich=lambda_stoich
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_sensor_loss": tr_sens,
                "train_stoich_loss": tr_stoich,
                "val_loss": val_loss,
                "val_sensor_loss": val_sens,
                "val_stoich_loss": val_stoich,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                    "scalers": {
                        "sensor_mean": scalers.sensor_mean.tolist(),
                        "sensor_std": scalers.sensor_std.tolist(),
                        "sensor_names": scalers.sensor_names,
                        "input_mean": scalers.input_mean.tolist(),
                        "input_std": scalers.input_std.tolist(),
                        "input_names": scalers.input_names,
                    },
                },
                checkpoint_path,
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:3d}/{args.epochs:3d} | "
                f"Train Loss: {tr_loss:.4f} (sens: {tr_sens:.4f}, stoich: {tr_stoich:.4f}) | "
                f"Val Loss: {val_loss:.4f} (sens: {val_sens:.4f}) | "
                f"Best Val: {best_val_loss:.4f}",
                flush=True,
            )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best checkpoint saved to {checkpoint_path}")

    # Save training history
    history_file = args.output_dir / "training_history.json"
    with history_file.open("w") as handle:
        json.dump(history, handle, indent=2)
    print(f"Training history saved to {history_file}")


if __name__ == "__main__":
    main()
