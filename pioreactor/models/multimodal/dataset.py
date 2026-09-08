"""Unified Multimodal Dataset Ingestion for Pioreactor Digital Twins.

Ingests and synchronizes all 39 physical bioreactor runs into unified 5-minute binned trajectories:
1. Pulse Experiments (26 runs):
   - Chem (20 runs): control (x1), glucose (x3), nitrogen (x5), salt (x5), sulfur (x5), uracil (x1)
   - Temp (3 runs): pulse exp/temperature/1, 2, 3
   - UV (3 runs): pulse exp/uv/1, 2, 3
2. Sine Wave Modulation (7 runs):
   - Chem (5 runs): glucose (x1), nitro (x1), salt (x2), sulfur (x1)
   - Temp (1 run): sine wave encoding/temperature
   - UV (1 run): sine wave encoding/UV
3. Mackey-Glass Dynamics (6 runs):
   - Chem (4 runs): glucose (x1), nitro (x1), salt (x1), sulfur (x1)
   - Temp (1 run): mackey glass/temp
   - UV (1 run): mackey glass/uv

Held-Out Evaluation Benchmarks (strictly unseen during training):
- 4D Rössler hyperchaotic experiment (pioreactor/data/rossler)
- Continuous 25 May Mackey-Glass experiment (results/25th may)
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

    # 1. Pulse experiments
    pulse_root = data_root / "pulse exp"
    if pulse_root.exists():
        for dosing_path in pulse_root.rglob("dosing_events-*.csv"):
            match = RUN_ID_RE.search(dosing_path.name)
            if not match:
                continue
            rel = dosing_path.relative_to(pulse_root)
            condition = rel.parts[0]
            if condition in ("temperature", "uv"):
                continue
            runs.append(DiscoveredRun(dosing_path.parent, match.group(1), condition, "pulse", "chem", dosing_path))

        pulse_temp = pulse_root / "temperature"
        if pulse_temp.exists():
            for anchor in pulse_temp.rglob("temperature_readings-*.csv"):
                match = RUN_ID_RE.search(anchor.name)
                if match:
                    runs.append(DiscoveredRun(anchor.parent, match.group(1), "temperature", "pulse", "temp", anchor))

        pulse_uv = pulse_root / "uv"
        if pulse_uv.exists():
            for anchor in pulse_uv.rglob("led_change_events-*.csv"):
                match = RUN_ID_RE.search(anchor.name)
                if match:
                    runs.append(DiscoveredRun(anchor.parent, match.group(1), "uv", "pulse", "uv", anchor))

    # 2. Sine wave encoding
    sine_root = data_root / "sine wave encoding"
    if sine_root.exists():
        sine_chem = sine_root / "chem"
        if sine_chem.exists():
            for dosing_path in sine_chem.rglob("dosing_events-*.csv"):
                match = RUN_ID_RE.search(dosing_path.name)
                if not match:
                    continue
                rel = dosing_path.relative_to(sine_chem)
                condition = rel.parts[0]
                runs.append(DiscoveredRun(dosing_path.parent, match.group(1), condition, "sine", "chem", dosing_path))

        sine_temp = sine_root / "temperature"
        if sine_temp.exists():
            for anchor in sine_temp.rglob("temperature_readings-*.csv"):
                match = RUN_ID_RE.search(anchor.name)
                if match:
                    runs.append(DiscoveredRun(anchor.parent, match.group(1), "temp", "sine", "temp", anchor))

        sine_uv = sine_root / "UV"
        if sine_uv.exists():
            for anchor in sine_uv.rglob("led_change_events-*.csv"):
                match = RUN_ID_RE.search(anchor.name)
                if match:
                    runs.append(DiscoveredRun(anchor.parent, match.group(1), "uv", "sine", "uv", anchor))

    # 3. Mackey-Glass encoding
    mg_root = data_root / "mackey glass"
    if mg_root.exists():
        mg_chem = mg_root / "chem"
        if mg_chem.exists():
            for dosing_path in mg_chem.rglob("dosing_events-*.csv"):
                match = RUN_ID_RE.search(dosing_path.name)
                if not match:
                    continue
                rel = dosing_path.relative_to(mg_chem)
                condition = rel.parts[0]
                runs.append(DiscoveredRun(dosing_path.parent, match.group(1), condition, "mackey_glass", "chem", dosing_path))

        mg_temp = mg_root / "temp"
        if mg_temp.exists():
            for anchor in mg_temp.rglob("temperature_readings-*.csv"):
                match = RUN_ID_RE.search(anchor.name)
                if match:
                    runs.append(DiscoveredRun(anchor.parent, match.group(1), "temp", "mackey_glass", "temp", anchor))

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


def numeric_stream(path: Path | None, candidate_cols: list[str], name: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(index=index, data={name: np.nan, f"observed_{name}": False})
    frame = read_csv(path)
    found_col = None
    for col in candidate_cols:
        if col in frame.columns:
            found_col = col
            break
    if found_col is None:
        return pd.DataFrame(index=index, data={name: np.nan, f"observed_{name}": False})
    values = pd.to_numeric(frame[found_col], errors="coerce")
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
    anchor_df = read_csv(run.anchor_path)
    start = anchor_df["timestamp"].min().floor(FREQUENCY)
    end = anchor_df["timestamp"].max().ceil(FREQUENCY)
    index = pd.date_range(start, end, freq=FREQUENCY, tz="UTC", name="timestamp")

    # 1. Fluid pumps
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

    # 2. Temperature: check temperature_readings- first, then co2_readings- fallback
    temp_path = sibling(run, "temperature_readings-")
    if temp_path and temp_path.exists():
        temp_df = numeric_stream(temp_path, ["temperature_c"], "temp_c", index)
        temp_series = temp_df["temp_c"].fillna(BASELINE_TEMP_C)
    else:
        co2_fallback = sibling(run, "co2_readings-")
        if co2_fallback and co2_fallback.exists():
            temp_df = numeric_stream(co2_fallback, ["temperature_c"], "temp_c", index)
            temp_series = temp_df["temp_c"].fillna(BASELINE_TEMP_C)
        else:
            temp_series = pd.Series(BASELINE_TEMP_C, index=index, name="temp_c")

    # 3. UV Intensity
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
    norm_od = numeric_stream(sibling(run, "od_readings_filtered-"), ["normalized_od_reading"], "norm_od", index)
    growth = numeric_stream(sibling(run, "growth_rates-"), ["rate"], "growth_rate", index)
    co2 = numeric_stream(sibling(run, "co2_readings-"), ["co2_reading_ppm", "co2_ppm"], "co2_ppm", index)
    angles = angled_od_stream(sibling(run, "od_readings-"), index)
    spectrum = spectrum_stream(sibling(run, "as7341_spectrum_readings-"), index)

    metadata = pd.DataFrame(
        {
            "run_key": f"{run.modality}:{run.input_type}:{run.condition}:{run.run_id}",
            "modality": run.modality,
            "input_type": run.input_type,
            "condition": run.condition,
            "run_id": run.run_id,
            "elapsed_minutes": (index - index[0]).total_seconds() / 60.0,
            "volume_ml": volume_series,
        },
        index=index,
    )

    inputs = pd.concat([pumps, temp_series, uv_series], axis=1)
    combined = pd.concat([metadata, inputs, norm_od, growth, co2, angles, spectrum], axis=1)

    for sensor in SENSOR_TARGETS:
        if sensor not in combined.columns:
            combined[sensor] = np.nan
        obs_col = f"observed_{sensor}"
        if obs_col not in combined.columns:
            combined[obs_col] = combined[sensor].notna()

    return combined.reset_index()


def compute_scalers(tables: list[pd.DataFrame]) -> dict[str, Any]:
    all_rows = pd.concat(tables, ignore_index=True)
    scalers: dict[str, Any] = {"inputs": {}, "sensors": {}, "initial_volume_ml": 13.5}

    for col in INPUT_COLUMNS:
        scalers["inputs"][col] = {
            "mean": float(all_rows[col].mean()),
            "std": float(max(all_rows[col].std(), 1e-4)),
            "min": float(all_rows[col].min()),
            "max": float(all_rows[col].max()),
        }

    for col in SENSOR_TARGETS:
        valid = all_rows[col].dropna()
        mean_val = float(valid.mean()) if not valid.empty else 0.0
        std_val = float(valid.std()) if not valid.empty and valid.std() > 1e-5 else 1.0
        scalers["sensors"][col] = {
            "mean": 0.0 if np.isnan(mean_val) else mean_val,
            "std": 1.0 if (np.isnan(std_val) or std_val < 1e-5) else std_val,
            "min": float(valid.min() if not valid.empty else 0.0),
            "max": float(valid.max() if not valid.empty else 1.0),
        }

    return scalers


def build_dataset(data_root: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_all_runs(data_root)
    print(f"Discovered {len(discovered)} total runs across chem, temp, and uv.")

    manifest_rows = []
    tables = []

    for run in discovered:
        table = build_run_table(run)
        slug = f"{run.modality}__{run.input_type}__{run.condition}__{run.run_id}.csv"
        out_path = runs_dir / slug
        table.to_csv(out_path, index=False)
        tables.append(table)
        manifest_rows.append(
            {
                "run_key": f"{run.modality}:{run.input_type}:{run.condition}:{run.run_id}",
                "modality": run.modality,
                "input_type": run.input_type,
                "condition": run.condition,
                "run_id": run.run_id,
                "rows": len(table),
                "filename": slug,
                "path": str(out_path),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    scalers = compute_scalers(tables)
    with open(output_dir / "scalers.json", "w", encoding="utf-8") as f:
        json.dump(scalers, f, indent=2)

    print(f"Ingestion complete: {len(manifest)} runs written to {runs_dir}")
    print(f"Manifest saved to {output_dir / 'manifest.csv'}")
    print(f"Scalers saved to {output_dir / 'scalers.json'}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Pioreactor Dataset Ingestion")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build_dataset(args.data_root, args.output_dir)
