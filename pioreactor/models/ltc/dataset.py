"""Dataset ingestion, synchronization, and PyTorch Dataset for Liquid Time-Constant (LTC) Yeast Modeling.

Supports:
1. Chemical Dosing Experiments:
   - Pulse experiments (20 runs): control (x1), glucose (x3), nitrogen (x5), salt (x5), sulfur (x5), uracil (x1)
   - Sine wave dosing (5 runs): glucose (x1), nitro (x1), salt (x2), sulfur (x1)
   - Mackey-Glass chemical dosing (4 runs): glucose (x1), nitro (x1), salt (x1), sulfur (x1)
2. Multimodal Inputs:
   - Temperature modulation (temp_c in °C, baseline 30.0)
   - UV irradiation (uv_intensity in %, baseline 0.0)
3. Target Sensors (14 dimensions):
   - norm_od, growth_rate, co2_ppm
   - od_45, od_90, od_135
   - nm_415, nm_445, nm_480, nm_515, nm_555, nm_590, nm_630, nm_680

Held-out biocomputing benchmarks (4D Rössler hyperchaotic and continuous 25 May Mackey-Glass)
remain strictly unseen for downstream reservoir evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parents[1] / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "dataset"
RUN_ID_RE = re.compile(r"all_units-(\d+)\.csv$")
FREQUENCY = "5min"
MAX_FORWARD_FILL_BINS = 2
BASELINE_TEMP_C = 30.0
BASELINE_UV_INTENSITY = 0.0

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


@dataclass(frozen=True)
class DiscoveredRun:
    path: Path
    run_id: str
    condition: str
    modality: str
    input_type: str  # 'chem', 'temp', 'uv'
    anchor_path: Path


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise ValueError(f"{path} has no timestamp column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp")


def discover_all_runs(data_root: Path) -> list[DiscoveredRun]:
    """Discover pulse, sine, and mackey-glass runs across chem, temp, and uv."""
    runs: list[DiscoveredRun] = []

    # 1. Pulse experiments (chemical dosing)
    pulse_root = data_root / "pulse exp"
    if pulse_root.exists():
        for dosing_path in pulse_root.rglob("dosing_events-*.csv"):
            match = RUN_ID_RE.search(dosing_path.name)
            if not match:
                continue
            rel = dosing_path.relative_to(pulse_root)
            condition = rel.parts[0]
            runs.append(DiscoveredRun(dosing_path.parent, match.group(1), condition, "pulse", "chem", dosing_path))

    # 2. Sine wave encoding
    sine_root = data_root / "sine wave encoding"
    if sine_root.exists():
        # chem
        sine_chem = sine_root / "chem"
        if sine_chem.exists():
            for dosing_path in sine_chem.rglob("dosing_events-*.csv"):
                match = RUN_ID_RE.search(dosing_path.name)
                if not match:
                    continue
                rel = dosing_path.relative_to(sine_chem)
                condition = rel.parts[0]
                runs.append(DiscoveredRun(dosing_path.parent, match.group(1), condition, "sine", "chem", dosing_path))
        # temp
        sine_temp = sine_root / "temperature"
        if sine_temp.exists():
            for anchor in sine_temp.rglob("temperature_readings-*.csv"):
                match = RUN_ID_RE.search(anchor.name)
                if match:
                    runs.append(DiscoveredRun(anchor.parent, match.group(1), "temp", "sine", "temp", anchor))
        # uv
        sine_uv = sine_root / "UV"
        if sine_uv.exists():
            for anchor in sine_uv.rglob("led_change_events-*.csv"):
                match = RUN_ID_RE.search(anchor.name)
                if match:
                    runs.append(DiscoveredRun(anchor.parent, match.group(1), "uv", "sine", "uv", anchor))

    # 3. Mackey-Glass encoding
    mg_root = data_root / "mackey glass"
    if mg_root.exists():
        # chem
        mg_chem = mg_root / "chem"
        if mg_chem.exists():
            for dosing_path in mg_chem.rglob("dosing_events-*.csv"):
                match = RUN_ID_RE.search(dosing_path.name)
                if not match:
                    continue
                rel = dosing_path.relative_to(mg_chem)
                condition = rel.parts[0]
                runs.append(DiscoveredRun(dosing_path.parent, match.group(1), condition, "mackey_glass", "chem", dosing_path))
        # temp
        mg_temp = mg_root / "temp"
        if mg_temp.exists():
            for anchor in mg_temp.rglob("temperature_readings-*.csv"):
                match = RUN_ID_RE.search(anchor.name)
                if match:
                    runs.append(DiscoveredRun(anchor.parent, match.group(1), "temp", "mackey_glass", "temp", anchor))
        # uv
        mg_uv = mg_root / "uv"
        if mg_uv.exists():
            for anchor in mg_uv.rglob("led_change_events-*.csv"):
                match = RUN_ID_RE.search(anchor.name)
                if match:
                    runs.append(DiscoveredRun(anchor.parent, match.group(1), "uv", "mackey_glass", "uv", anchor))

    return sorted(runs, key=lambda r: (r.modality, r.input_type, r.condition, r.run_id))


def sibling(run: DiscoveredRun, prefix: str) -> Path | None:
    candidates = sorted(run.path.glob(f"{prefix}*all_units-{run.run_id}.csv"))
    return candidates[0] if candidates else None


def numeric_stream(path: Path | None, value: str, name: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(index=index, data={name: np.nan, f"observed_{name}": False})
    frame = read_csv(path)
    if value not in frame.columns:
        return pd.DataFrame(index=index, data={name: np.nan, f"observed_{name}": False})
    values = pd.to_numeric(frame[value], errors="coerce")
    series = pd.Series(values.to_numpy(), index=frame["timestamp"]).resample(FREQUENCY).mean().reindex(index)
    observed = series.notna()
    return pd.DataFrame({name: series.ffill(limit=MAX_FORWARD_FILL_BINS), f"observed_{name}": observed}, index=index)


def spectrum_stream(path: Path | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(index=index)
    frame = read_csv(path)
    if "reading" not in frame or "band" not in frame:
        return pd.DataFrame(index=index)
    frame["reading"] = pd.to_numeric(frame["reading"], errors="coerce")
    frame["band"] = pd.to_numeric(frame["band"], errors="coerce")
    frame = frame.dropna(subset=["reading", "band"])
    if frame.empty:
        return pd.DataFrame(index=index)
    pivot = frame.pivot_table(index="timestamp", columns="band", values="reading", aggfunc="mean")
    pivot.columns = [f"nm_{int(col)}" for col in pivot.columns]
    pivot = pivot.resample(FREQUENCY).mean().reindex(index)
    observed = pivot.notna().rename(columns=lambda col: f"observed_{col}")
    return pd.concat([pivot.ffill(limit=MAX_FORWARD_FILL_BINS), observed], axis=1)


def angled_od_stream(path: Path | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(index=index)
    frame = read_csv(path)
    if "od_reading" not in frame or "angle" not in frame:
        return pd.DataFrame(index=index)
    frame["od_reading"] = pd.to_numeric(frame["od_reading"], errors="coerce")
    frame["angle"] = pd.to_numeric(frame["angle"], errors="coerce")
    frame = frame.dropna(subset=["od_reading", "angle"])
    if frame.empty:
        return pd.DataFrame(index=index)
    pivot = frame.pivot_table(index="timestamp", columns="angle", values="od_reading", aggfunc="mean")
    pivot.columns = [f"od_{int(col)}" for col in pivot.columns]
    pivot = pivot.resample(FREQUENCY).mean().reindex(index)
    observed = pivot.notna().rename(columns=lambda col: f"observed_{col}")
    return pd.concat([pivot.ffill(limit=MAX_FORWARD_FILL_BINS), observed], axis=1)


def build_run_table(run: DiscoveredRun, initial_volume_ml: float = 13.5) -> pd.DataFrame:
    # 1. Determine run time boundaries
    anchor_df = read_csv(run.anchor_path)
    start = anchor_df["timestamp"].min().floor(FREQUENCY)
    end = anchor_df["timestamp"].max().ceil(FREQUENCY)
    index = pd.date_range(start, end, freq=FREQUENCY, tz="UTC", name="timestamp")

    # 2. Fluid pump inputs
    pumps = pd.DataFrame(0.0, index=index, columns=["add_media_ml", "add_alt_media_ml", "remove_waste_ml"])
    dosing_path = sibling(run, "dosing_events-")
    if dosing_path and dosing_path.exists():
        dosing = read_csv(dosing_path)
        dosing["volume_change_ml"] = pd.to_numeric(dosing["volume_change_ml"], errors="coerce").fillna(0.0)
        for event_name, col in (("add_media", "add_media_ml"), ("add_alt_media", "add_alt_media_ml"), ("remove_waste", "remove_waste_ml")):
            subset = dosing[dosing["event"] == event_name]
            if not subset.empty:
                totals = subset.set_index("timestamp")["volume_change_ml"].resample(FREQUENCY).sum()
                pumps[col] = totals.reindex(index, fill_value=0.0)

    pumps["dose_total_ml"] = pumps["add_media_ml"] + pumps["add_alt_media_ml"]
    net = pumps["add_media_ml"] + pumps["add_alt_media_ml"] - pumps["remove_waste_ml"]
    volume_series = initial_volume_ml + net.cumsum()

    # 3. Multimodal inputs (temperature and UV)
    temp_path = sibling(run, "temperature_readings-")
    if temp_path and temp_path.exists():
        temp_df = numeric_stream(temp_path, "temperature_c", "temp_c", index)
        temp_series = temp_df["temp_c"].fillna(BASELINE_TEMP_C)
    else:
        temp_series = pd.Series(BASELINE_TEMP_C, index=index, name="temp_c")

    uv_path = sibling(run, "led_change_events-")
    if uv_path and uv_path.exists():
        uv_raw = read_csv(uv_path)
        if "intensity" in uv_raw.columns:
            uv_resampled = (
                pd.to_numeric(uv_raw["intensity"], errors="coerce")
                .groupby(uv_raw["timestamp"])
                .mean()
                .resample(FREQUENCY)
                .mean()
                .reindex(index)
                .ffill(limit=MAX_FORWARD_FILL_BINS)
                .fillna(BASELINE_UV_INTENSITY)
            )
            uv_series = pd.Series(uv_resampled.values, index=index, name="uv_intensity")
        else:
            uv_series = pd.Series(BASELINE_UV_INTENSITY, index=index, name="uv_intensity")
    else:
        uv_series = pd.Series(BASELINE_UV_INTENSITY, index=index, name="uv_intensity")

    # 4. Sensor streams
    norm_od = numeric_stream(sibling(run, "od_readings_filtered"), "normalized_od_reading", "norm_od", index)
    growth = numeric_stream(sibling(run, "growth_rates"), "rate", "growth_rate", index)
    co2 = numeric_stream(sibling(run, "co2_readings"), "CO2_ppm", "co2_ppm", index)
    angles = angled_od_stream(sibling(run, "od_readings-"), index)
    spectrum = spectrum_stream(sibling(run, "as7341_spectrum_readings"), index)

    table = pd.concat([pumps, temp_series, uv_series, norm_od, growth, co2, angles, spectrum], axis=1)
    table["volume_ml"] = volume_series
    table["condition"] = run.condition
    table["modality"] = run.modality
    table["input_type"] = run.input_type
    table["run_id"] = run.run_id
    table["run_key"] = f"{run.modality}:{run.input_type}:{run.condition}:{run.run_id}"

    # Ensure all sensor targets exist
    for target in SENSOR_TARGETS:
        if target not in table.columns:
            table[target] = np.nan
        if f"observed_{target}" not in table.columns:
            table[f"observed_{target}"] = False

    return table.reset_index()


def build_and_save_all_runs(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    output_runs_dir = output_dir / "runs"
    output_runs_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_all_runs(data_root)
    manifest = []
    print(f"Found {len(runs)} total runs across pulse, sine, and mackey-glass experiments.")

    for idx, r in enumerate(runs, start=1):
        filename = f"{r.modality}__{r.input_type}__{r.condition}__{r.run_id}.csv"
        target_path = output_runs_dir / filename
        table = build_run_table(r)
        table.to_csv(target_path, index=False)

        manifest.append({
            "run_key": f"{r.modality}:{r.input_type}:{r.condition}:{r.run_id}",
            "modality": r.modality,
            "input_type": r.input_type,
            "condition": r.condition,
            "run_id": r.run_id,
            "rows": len(table),
            "filename": filename,
            "path": str(target_path),
        })
        print(f"  [{idx:02d}/{len(runs):02d}] {filename} ({len(table)} bins, input={r.input_type})")

    df_manifest = pd.DataFrame(manifest)
    df_manifest.to_csv(output_dir / "manifest.csv", index=False)
    print(f"\nManifest saved to {output_dir / 'manifest.csv'}")
    return df_manifest


class NormalizationScalers:
    """Computes and stores standard normalization statistics for inputs and sensors."""

    def __init__(self, stats: dict[str, Any] | None = None):
        self.stats = stats or {}

    @classmethod
    def fit_from_dataframes(cls, dfs: list[pd.DataFrame]) -> NormalizationScalers:
        combined = pd.concat(dfs, ignore_index=True)
        stats = {}

        # 1. Inputs: scale to [0, 1] or mean/std
        for col in INPUT_COLUMNS:
            vals = combined[col].to_numpy(dtype=np.float32)
            stats[col] = {
                "min": float(np.nanmin(vals)),
                "max": float(max(np.nanmax(vals), 1e-4)),
                "mean": float(np.nanmean(vals)),
                "std": float(max(np.nanstd(vals), 1e-4)),
            }

        # 2. Sensors: z-score with robust clipping
        for col in SENSOR_TARGETS:
            vals = combined[col].to_numpy(dtype=np.float32)
            valid = vals[np.isfinite(vals)]
            if len(valid) == 0:
                stats[col] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
            else:
                stats[col] = {
                    "min": float(np.percentile(valid, 1)),
                    "max": float(np.percentile(valid, 99)),
                    "mean": float(np.mean(valid)),
                    "std": float(max(np.std(valid), 1e-4)),
                }
        return cls(stats)

    def normalize_inputs(self, u: np.ndarray) -> np.ndarray:
        # u: (..., 6)
        out = np.zeros_like(u, dtype=np.float32)
        for i, col in enumerate(INPUT_COLUMNS):
            col_max = self.stats[col]["max"]
            if col == "temp_c":
                # Center around 30°C and scale by 10°C
                out[..., i] = (u[..., i] - 30.0) / 10.0
            elif col == "uv_intensity":
                # Scale 0..100% to 0..1
                out[..., i] = u[..., i] / 100.0
            else:
                out[..., i] = u[..., i] / col_max
        return out

    def normalize_sensors(self, y: np.ndarray) -> np.ndarray:
        # y: (..., 14)
        out = np.zeros_like(y, dtype=np.float32)
        for i, col in enumerate(SENSOR_TARGETS):
            col_mean = self.stats[col]["mean"]
            col_std = self.stats[col]["std"]
            out[..., i] = (y[..., i] - col_mean) / col_std
        return out

    def denormalize_sensors(self, y_norm: np.ndarray) -> np.ndarray:
        out = np.zeros_like(y_norm, dtype=np.float32)
        for i, col in enumerate(SENSOR_TARGETS):
            col_mean = self.stats[col]["mean"]
            col_std = self.stats[col]["std"]
            out[..., i] = (y_norm[..., i] * col_std) + col_mean
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> NormalizationScalers:
        with open(path) as f:
            return cls(json.load(f))


class LtcDataset(Dataset):
    """PyTorch Dataset yielding windowed continuous-time sequences for BPTT."""

    def __init__(
        self,
        run_dfs: list[pd.DataFrame],
        scalers: NormalizationScalers,
        window_size: int = 36,  # 36 bins = 3 hours
        stride: int = 6,        # stride of 30 minutes
    ):
        self.scalers = scalers
        self.window_size = window_size
        self.samples = []

        for df in run_dfs:
            n_rows = len(df)
            if n_rows < window_size:
                continue

            u_raw = df[INPUT_COLUMNS].to_numpy(dtype=np.float32)
            u_norm = np.nan_to_num(scalers.normalize_inputs(u_raw), nan=0.0)

            y_raw = df[SENSOR_TARGETS].to_numpy(dtype=np.float32)
            y_norm = np.nan_to_num(scalers.normalize_sensors(y_raw), nan=0.0)

            mask_cols = [f"observed_{s}" for s in SENSOR_TARGETS]
            obs_mask = df[mask_cols].fillna(False).to_numpy(dtype=bool)

            volume = df["volume_ml"].to_numpy(dtype=np.float32)
            norm_od = df["norm_od"].to_numpy(dtype=np.float32)
            condition = df["condition"].iloc[0]
            modality = df["modality"].iloc[0]

            for start_idx in range(0, n_rows - window_size + 1, stride):
                end_idx = start_idx + window_size
                self.samples.append({
                    "u": torch.from_numpy(u_norm[start_idx:end_idx]),
                    "y": torch.from_numpy(y_norm[start_idx:end_idx]),
                    "mask": torch.from_numpy(obs_mask[start_idx:end_idx]),
                    "init_vol": torch.tensor([volume[start_idx]], dtype=torch.float32),
                    "init_od": torch.tensor([norm_od[start_idx] if np.isfinite(norm_od[start_idx]) else 0.5], dtype=torch.float32),
                    "condition": condition,
                    "modality": modality,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def get_stratified_splits(
    manifest_df: pd.DataFrame,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Partition runs into Training, In-Distribution Validation, and Held-Out Benchmarks.
    
    Rules:
      - Training & Validation strictly draw from pulse and sine runs.
      - Uracil (x1) and Control (x1) stay in Training.
      - 1 pulse run each of Glucose, Salt, Nitrogen, Sulfur are held out for Validation.
      - Mackey-Glass runs can form an unseen benchmark split.
    """
    np.random.seed(seed)

    # 1. Benchmark set: all mackey_glass runs (kept strictly unseen if requested)
    bench_keys = manifest_df[manifest_df["modality"] == "mackey_glass"]["run_key"].tolist()

    # 2. Candidate train/val runs: pulse and sine
    dosing_manifest = manifest_df[manifest_df["modality"].isin(["pulse", "sine"])]
    pulse_runs = dosing_manifest[dosing_manifest["modality"] == "pulse"]

    val_keys = []
    for cond in ["glucose", "nitrogen", "salt", "sulfur"]:
        cond_keys = pulse_runs[pulse_runs["condition"] == cond]["run_key"].tolist()
        if cond_keys:
            chosen = np.random.choice(cond_keys)
            val_keys.append(chosen)

    train_keys = [
        k for k in dosing_manifest["run_key"].tolist()
        if k not in val_keys
    ]

    return train_keys, val_keys, bench_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build synchronized LTC dataset")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df_manifest = build_and_save_all_runs(args.data_root, args.output_dir)
    train_k, val_k, bench_k = get_stratified_splits(df_manifest)

    print("\nDataset Split Summary:")
    print(f"  Training Runs ({len(train_k)}): {train_k}")
    print(f"  Validation Runs ({len(val_k)}): {val_k}")
    print(f"  Unseen Benchmark Runs ({len(bench_k)}): {bench_k}")
