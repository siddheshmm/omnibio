"""PyTorch Dataset and DataLoader for continuous-time dAMN training across all 14 sensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"


@dataclass
class NormalizationScalers:
    sensor_mean: np.ndarray
    sensor_std: np.ndarray
    sensor_names: list[str]
    input_mean: np.ndarray
    input_std: np.ndarray
    input_names: list[str]

    def normalize_sensors(self, y: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(y, torch.Tensor):
            mean = torch.tensor(self.sensor_mean, dtype=y.dtype, device=y.device)
            std = torch.tensor(self.sensor_std, dtype=y.dtype, device=y.device)
            return (y - mean) / std
        return (y - self.sensor_mean) / self.sensor_std

    def denormalize_sensors(self, y_norm: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(y_norm, torch.Tensor):
            mean = torch.tensor(self.sensor_mean, dtype=y_norm.dtype, device=y_norm.device)
            std = torch.tensor(self.sensor_std, dtype=y_norm.dtype, device=y_norm.device)
            return y_norm * std + mean
        return y_norm * self.sensor_std + self.sensor_mean

    def normalize_inputs(self, u: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(u, torch.Tensor):
            mean = torch.tensor(self.input_mean, dtype=u.dtype, device=u.device)
            std = torch.tensor(self.input_std, dtype=u.dtype, device=u.device)
            return (u - mean) / std
        return (u - self.input_mean) / self.input_std


class PioreactorTrajectoryDataset(Dataset):
    """Slices continuous bioreactor runs into multi-step trajectory sequences for Neural ODE / dAMN."""

    def __init__(
        self,
        tables: list[pd.DataFrame],
        sensor_targets: list[str],
        input_columns: list[str],
        window_steps: int = 24,
        scalers: NormalizationScalers | None = None,
        stride: int = 4,
    ):
        self.sensor_targets = sensor_targets
        self.input_columns = input_columns
        self.window_steps = window_steps

        # Compute scalers if not provided
        if scalers is None:
            all_sensors = []
            all_inputs = []
            for t in tables:
                all_sensors.append(t[sensor_targets].to_numpy(dtype=np.float32))
                all_inputs.append(t[input_columns].to_numpy(dtype=np.float32))
            cat_s = np.concatenate(all_sensors, axis=0)
            cat_u = np.concatenate(all_inputs, axis=0)

            s_mean = np.nanmean(cat_s, axis=0)
            s_std = np.nanstd(cat_s, axis=0)
            s_std[s_std < 1e-5] = 1.0

            u_mean = np.nanmean(cat_u, axis=0)
            u_std = np.nanstd(cat_u, axis=0)
            u_std[u_std < 1e-5] = 1.0

            self.scalers = NormalizationScalers(
                sensor_mean=s_mean,
                sensor_std=s_std,
                sensor_names=sensor_targets,
                input_mean=u_mean,
                input_std=u_std,
                input_names=input_columns,
            )
        else:
            self.scalers = scalers

        # Slice runs into trajectory windows
        self.samples = []
        obs_cols = [f"observed_{c}" for c in sensor_targets]

        for run_df in tables:
            n_steps = len(run_df)
            if n_steps < window_steps:
                continue

            raw_y = run_df[sensor_targets].to_numpy(dtype=np.float32)
            raw_u = run_df[input_columns].to_numpy(dtype=np.float32)

            # Clean any NaNs in inputs and targets
            clean_norm_od = (
                run_df["norm_od"].bfill().ffill().fillna(0.5).to_numpy(dtype=np.float32)
                if "norm_od" in run_df
                else np.full(n_steps, 0.5, dtype=np.float32)
            )

            # Mask indicates whether each sensor point was directly observed
            if set(obs_cols).issubset(run_df.columns):
                raw_m = run_df[obs_cols].fillna(False).to_numpy(dtype=bool)
            else:
                raw_m = np.isfinite(raw_y)

            norm_y = self.scalers.normalize_sensors(raw_y)
            norm_y = np.nan_to_num(norm_y, nan=0.0)
            norm_u = self.scalers.normalize_inputs(raw_u)
            norm_u = np.nan_to_num(norm_u, nan=0.0)
            raw_v = (
                run_df["volume_ml"].bfill().ffill().fillna(13.5).to_numpy(dtype=np.float32)
                if "volume_ml" in run_df
                else np.full(n_steps, 13.5, dtype=np.float32)
            )

            for start in range(0, n_steps - window_steps + 1, stride):
                end = start + window_steps
                init_od_val = float(clean_norm_od[start])
                init_v_val = float(raw_v[start])
                self.samples.append(
                    {
                        "y": torch.tensor(norm_y[start:end], dtype=torch.float32),
                        "y_raw": torch.tensor(np.nan_to_num(raw_y[start:end], nan=0.0), dtype=torch.float32),
                        "u": torch.tensor(norm_u[start:end], dtype=torch.float32),
                        "u_raw": torch.tensor(np.nan_to_num(raw_u[start:end], nan=0.0), dtype=torch.float32),
                        "mask": torch.tensor(raw_m[start:end], dtype=torch.bool),
                        "volume": torch.tensor(raw_v[start:end], dtype=torch.float32),
                        "init_od": init_od_val,
                        "init_volume": init_v_val,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def load_dataset_runs(
    config_path: Path = DEFAULT_CONFIG,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[PioreactorTrajectoryDataset, PioreactorTrajectoryDataset, NormalizationScalers]:
    """Load, split by run_key, and return PyTorch train and test datasets."""
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    dataset_dir = (SCRIPT_DIR / config["dataset_dir"]).resolve()
    csv_paths = sorted(dataset_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No run tables found in {dataset_dir}")

    # Load tables
    tables = [pd.read_csv(p, parse_dates=["timestamp"]) for p in csv_paths]

    # Split by run_key to prevent data leakage
    rng = np.random.RandomState(seed)
    n_runs = len(tables)
    indices = rng.permutation(n_runs)
    n_train = max(1, int(n_runs * train_ratio))

    train_tables = [tables[i] for i in indices[:n_train]]
    test_tables = [tables[i] for i in indices[n_train:]]

    sensors = config["sensor_targets"]
    inputs = config["input_columns"]
    window_steps = int(config.get("training", {}).get("window_steps", 24))

    # Build datasets
    train_ds = PioreactorTrajectoryDataset(
        tables=train_tables,
        sensor_targets=sensors,
        input_columns=inputs,
        window_steps=window_steps,
        scalers=None,  # Will compute scalers from train_tables
        stride=4,
    )

    test_ds = PioreactorTrajectoryDataset(
        tables=test_tables,
        sensor_targets=sensors,
        input_columns=inputs,
        window_steps=window_steps,
        scalers=train_ds.scalers,  # Reuse training scalers
        stride=4,
    )

    return train_ds, test_ds, train_ds.scalers


if __name__ == "__main__":
    train_ds, test_ds, scalers = load_dataset_runs()
    print(f"Created Train Dataset with {len(train_ds)} windows, Test Dataset with {len(test_ds)} windows.")
    print(f"Tracking {len(scalers.sensor_names)} sensors: {scalers.sensor_names}")
    sample = train_ds[0]
    print(f"Sample shapes -> y: {sample['y'].shape}, u: {sample['u'].shape}, mask: {sample['mask'].shape}")
