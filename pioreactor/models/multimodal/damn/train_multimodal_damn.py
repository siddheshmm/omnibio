"""Training script for continuous Multimodal dAMN across all 39 bioreactor runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from multimodal_damn_ode import MultimodalDAMN
from dataset import MultimodalTrajectoryDataset, load_dataset_runs

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = SCRIPT_DIR.parents[0] / "artifacts" / "dataset" / "runs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "model"


def train_epoch(
    model: MultimodalDAMN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_stoich: float = 0.1,
) -> tuple[float, float, float]:
    model.train()
    total_loss, total_sensor_loss, total_stoich_loss = 0.0, 0.0, 0.0
    num_batches = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        sensors = batch["sensors"].to(device)
        mask = batch["mask"].to(device)
        init_od = batch["init_od"].to(device)
        init_vol = batch["init_volume"].to(device)

        optimizer.zero_grad()

        pred_sensors, pred_states, stoich_penalty = model.rollout(
            inputs=inputs,
            initial_od=init_od,
            initial_volume=init_vol,
        )

        diff_sq = (pred_sensors - sensors) ** 2
        sensor_loss = torch.sum(diff_sq * mask) / torch.clamp(mask.sum(), min=1.0)
        loss = sensor_loss + (lambda_stoich * stoich_penalty)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        total_sensor_loss += sensor_loss.item()
        total_stoich_loss += stoich_penalty.item()
        num_batches += 1

    return (
        total_loss / max(num_batches, 1),
        total_sensor_loss / max(num_batches, 1),
        total_stoich_loss / max(num_batches, 1),
    )


@torch.no_grad()
def evaluate_epoch(
    model: MultimodalDAMN,
    loader: DataLoader,
    device: torch.device,
    lambda_stoich: float = 0.1,
) -> tuple[float, float, float]:
    model.eval()
    total_loss, total_sensor_loss, total_stoich_loss = 0.0, 0.0, 0.0
    num_batches = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        sensors = batch["sensors"].to(device)
        mask = batch["mask"].to(device)
        init_od = batch["init_od"].to(device)
        init_vol = batch["init_volume"].to(device)

        pred_sensors, _, stoich_penalty = model.rollout(
            inputs=inputs,
            initial_od=init_od,
            initial_volume=init_vol,
        )

        diff_sq = (pred_sensors - sensors) ** 2
        sensor_loss = torch.sum(diff_sq * mask) / torch.clamp(mask.sum(), min=1.0)
        loss = sensor_loss + (lambda_stoich * stoich_penalty)

        total_loss += loss.item()
        total_sensor_loss += sensor_loss.item()
        total_stoich_loss += stoich_penalty.item()
        num_batches += 1

    return (
        total_loss / max(num_batches, 1),
        total_sensor_loss / max(num_batches, 1),
        total_stoich_loss / max(num_batches, 1),
    )


def train_multimodal_damn(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    window_steps: int = 24,
    device_name: str = "cpu",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    print(f"Training Multimodal dAMN on device: {device}")

    tables = load_dataset_runs(runs_dir)
    print(f"Loaded {len(tables)} runs from {runs_dir}")

    full_dataset = MultimodalTrajectoryDataset(tables, window_steps=window_steps, stride=3)
    print(f"Constructed {len(full_dataset)} trajectory windows (window_steps={window_steps})")

    # Train / val split (80 / 20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = MultimodalDAMN(
        num_sensors=14,
        num_metabolites=6,
        input_dim=6,
        latent_dim=8,
        hidden_dim=64,
        yield_glucose=0.0811,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_loss": [], "val_sensor_loss": []}
    best_val_loss = float("inf")
    best_ckpt_path = output_dir / "multimodal_damn_best.pt"

    model.stoich_constraint.debug_print_stoichiometry()

    print(f"\nStarting BPTT training across {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_sens, tr_stoich = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_sens, val_stoich = evaluate_epoch(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_sensor_loss"].append(val_sens)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "scalers": {
                        "sensor_mean": full_dataset.scalers.sensor_mean.tolist(),
                        "sensor_std": full_dataset.scalers.sensor_std.tolist(),
                        "sensor_names": full_dataset.scalers.sensor_names,
                        "input_mean": full_dataset.scalers.input_mean.tolist(),
                        "input_std": full_dataset.scalers.input_std.tolist(),
                        "input_names": full_dataset.scalers.input_names,
                    },
                },
                best_ckpt_path,
            )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:02d}/{epochs}] "
                f"Train Loss: {tr_loss:.4f} (sens: {tr_sens:.4f}, stoich: {tr_stoich:.4f}) | "
                f"Val Loss: {val_loss:.4f} (sens: {val_sens:.4f}) | "
                f"Best: {best_val_loss:.4f}"
            )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds.")
    print(f"Best model checkpoint saved to: {best_ckpt_path}")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, full_dataset.scalers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_multimodal_damn(args.runs_dir, args.output_dir, args.epochs, args.batch_size, args.lr)
