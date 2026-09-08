"""Dataset loader and trajectory batching for Multimodal dAMN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SENSOR_TARGETS = [
    "norm_od",
    "growth_rate",
    "co2_ppm",
    "od_45",
    "od_90",
    "od_135",
    "nm_415",
    "nm_445",
    "nm_480",
    "nm_515",
    "nm_555",
    "nm_590",
    "nm_630",
    "nm_680",
]

INPUT_COLUMNS = [
    "add_media_ml",
    "add_alt_media_ml",
    "remove_waste_ml",
    "dose_total_ml",
    "temp_c",
    "uv_intensity",
]


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


class MultimodalTrajectoryDataset(Dataset):
    """Slices continuous bioreactor runs into multi-step trajectories for multimodal dAMN."""

    def __init__(
        self,
        tables: list[pd.DataFrame],
        window_steps: int = 24,
        scalers: NormalizationScalers | None = None,
        stride: int = 4,
    ):
        self.window_steps = window_steps
        self.sensor_targets = SENSOR_TARGETS
        self.input_columns = INPUT_COLUMNS

        if scalers is None:
            all_s = [t[SENSOR_TARGETS].to_numpy(dtype=np.float32) for t in tables]
            all_u = [t[INPUT_COLUMNS].to_numpy(dtype=np.float32) for t in tables]
            cat_s = np.concatenate(all_s, axis=0)
            cat_u = np.concatenate(all_u, axis=0)

            s_mean = np.nan_to_num(np.nanmean(cat_s, axis=0), nan=0.0)
            s_std = np.nan_to_num(np.nanstd(cat_s, axis=0), nan=1.0)
            s_std[s_std < 1e-5] = 1.0

            u_mean = np.nan_to_num(np.nanmean(cat_u, axis=0), nan=0.0)
            u_std = np.nan_to_num(np.nanstd(cat_u, axis=0), nan=1.0)
            u_std[u_std < 1e-5] = 1.0

            self.scalers = NormalizationScalers(
                sensor_mean=s_mean,
                sensor_std=s_std,
                sensor_names=SENSOR_TARGETS,
                input_mean=u_mean,
                input_std=u_std,
                input_names=INPUT_COLUMNS,
            )
        else:
            self.scalers = scalers

        self.samples: list[dict[str, Any]] = []
        for df in tables:
            n_rows = len(df)
            if n_rows < window_steps:
                continue

            inputs = df[INPUT_COLUMNS].to_numpy(dtype=np.float32)
            sensors_raw = df[SENSOR_TARGETS].to_numpy(dtype=np.float32)
            obs_mask = np.isfinite(sensors_raw)
            clean_sensors = np.nan_to_num(sensors_raw, nan=0.0)
            norm_sensors = self.scalers.normalize_sensors(clean_sensors)

            vol_raw = df["volume_ml"].to_numpy(dtype=np.float32) if "volume_ml" in df else np.full(n_rows, 13.5, dtype=np.float32)
            od_raw = df["norm_od"].to_numpy(dtype=np.float32) if "norm_od" in df else np.full(n_rows, 0.5, dtype=np.float32)

            for start in range(0, n_rows - window_steps + 1, stride):
                end = start + window_steps
                # Initial conditions at start of window
                init_od = float(od_raw[start]) if np.isfinite(od_raw[start]) else 0.5
                init_vol = float(vol_raw[start]) if np.isfinite(vol_raw[start]) else 13.5

                self.samples.append(
                    {
                        "inputs": torch.tensor(inputs[start:end], dtype=torch.float32),
                        "sensors": torch.tensor(norm_sensors[start:end], dtype=torch.float32),
                        "mask": torch.tensor(obs_mask[start:end], dtype=torch.float32),
                        "init_od": torch.tensor([init_od], dtype=torch.float32),
                        "init_volume": torch.tensor([init_vol], dtype=torch.float32),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def load_dataset_runs(runs_dir: Path) -> list[pd.DataFrame]:
    csv_paths = sorted(runs_dir.glob("*.csv"))
    tables = [pd.read_csv(p) for p in csv_paths]
    return tables
