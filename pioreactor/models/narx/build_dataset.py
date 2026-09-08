"""Build synchronized 5-minute dosing/sensor state tables for the NARX twin."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parents[1] / "data"
DEFAULT_INITIAL_VOLUME_ML = 13.5
RUN_ID_RE = re.compile(r"all_units-(\d+)\.csv$")
FREQUENCY = "5min"
MAX_FORWARD_FILL_BINS = 2


@dataclass(frozen=True)
class Run:
    path: Path
    run_id: str
    condition: str
    modality: str
    dosing_path: Path


def read_csv(path: Path) -> pd.DataFrame:
    """Read an export and normalise its timestamp into a UTC index."""
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise ValueError(f"{path} has no timestamp column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp")


def discover_runs(data_root: Path) -> list[Run]:
    """Find one run for every dosing export, including duplicate exports in a folder."""
    sources = (
        ("pulse", data_root / "pulse exp"),
        ("sine", data_root / "sine wave encoding" / "chem"),
        ("mackey_glass", data_root / "mackey glass" / "chem"),
    )
    runs: list[Run] = []
    for modality, root in sources:
        if not root.exists():
            continue
        for dosing_path in root.rglob("dosing_events-*.csv"):
            match = RUN_ID_RE.search(dosing_path.name)
            if not match:
                continue
            relative = dosing_path.relative_to(root)
            condition = relative.parts[0]
            runs.append(Run(dosing_path.parent, match.group(1), condition, modality, dosing_path))
    return sorted(runs, key=lambda run: (run.modality, run.condition, run.run_id))


def sibling(run: Run, prefix: str) -> Path | None:
    """Return the matching exported stream for this run identifier, if available."""
    candidates = sorted(run.path.glob(f"{prefix}*all_units-{run.run_id}.csv"))
    return candidates[0] if candidates else None


def numeric_stream(path: Path | None, value: str, name: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(index=index, data={name: np.nan, f"observed_{name}": False})
    frame = read_csv(path)
    values = pd.to_numeric(frame[value], errors="coerce")
    series = pd.Series(values.to_numpy(), index=frame["timestamp"]).resample(FREQUENCY).mean().reindex(index)
    observed = series.notna()
    return pd.DataFrame({name: series.ffill(limit=MAX_FORWARD_FILL_BINS), f"observed_{name}": observed}, index=index)


def spectrum_stream(path: Path | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(index=index)
    frame = read_csv(path)
    frame["reading"] = pd.to_numeric(frame["reading"], errors="coerce")
    frame["band"] = pd.to_numeric(frame["band"], errors="coerce")
    frame = frame.dropna(subset=["reading", "band"])
    if frame.empty:
        return pd.DataFrame(index=index)
    pivot = frame.pivot_table(index="timestamp", columns="band", values="reading", aggfunc="mean")
    pivot.columns = [f"nm_{int(column)}" for column in pivot.columns]
    pivot = pivot.resample(FREQUENCY).mean().reindex(index)
    observed = pivot.notna().rename(columns=lambda column: f"observed_{column}")
    return pd.concat([pivot.ffill(limit=MAX_FORWARD_FILL_BINS), observed], axis=1)


def angled_od_stream(path: Path | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load raw OD values as separate angle-specific optical states."""
    if path is None:
        return pd.DataFrame(index=index)
    frame = read_csv(path)
    frame["od_reading"] = pd.to_numeric(frame["od_reading"], errors="coerce")
    frame["angle"] = pd.to_numeric(frame["angle"], errors="coerce")
    frame = frame.dropna(subset=["od_reading", "angle"])
    if frame.empty:
        return pd.DataFrame(index=index)
    pivot = frame.pivot_table(index="timestamp", columns="angle", values="od_reading", aggfunc="mean")
    pivot.columns = [f"od_{int(column)}" for column in pivot.columns]
    pivot = pivot.resample(FREQUENCY).mean().reindex(index)
    observed = pivot.notna().rename(columns=lambda column: f"observed_{column}")
    return pd.concat([pivot.ffill(limit=MAX_FORWARD_FILL_BINS), observed], axis=1)


def event_stream(run: Run, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = read_csv(run.dosing_path)
    frame["volume_change_ml"] = pd.to_numeric(frame["volume_change_ml"], errors="coerce").fillna(0.0)
    events = frame.pivot_table(
        index="timestamp", columns="event", values="volume_change_ml", aggfunc="sum", fill_value=0.0
    ).resample(FREQUENCY).sum().reindex(index, fill_value=0.0)
    output = pd.DataFrame(index=index)
    for event, column in (("add_media", "add_media_ml"), ("add_alt_media", "add_alt_media_ml"), ("remove_waste", "remove_waste_ml")):
        output[column] = events[event] if event in events else 0.0
    output["dose_total_ml"] = output["add_media_ml"] + output["add_alt_media_ml"]
    return output


def add_reactor_state(table: pd.DataFrame, initial_volume_ml: float = DEFAULT_INITIAL_VOLUME_ML) -> pd.DataFrame:
    """Track cumulative dosing and estimated working volume."""
    output = table.copy()
    net_change = output["add_media_ml"] + output["add_alt_media_ml"] - output["remove_waste_ml"]
    output["volume_ml"] = float(initial_volume_ml) + net_change.cumsum()
    output["cum_add_media_ml"] = output["add_media_ml"].cumsum()
    output["cum_add_alt_media_ml"] = output["add_alt_media_ml"].cumsum()
    output["cum_remove_waste_ml"] = output["remove_waste_ml"].cumsum()
    output["cum_dose_total_ml"] = output["dose_total_ml"].cumsum()
    return output


def make_table(run: Run) -> pd.DataFrame:
    """Synchronize all sensor streams and actual fluid events for one run."""
    paths = [run.dosing_path]
    for prefix in ("od_readings_filtered-", "od_readings-", "growth_rates-", "co2_readings-", "as7341_spectrum_readings-"):
        candidate = sibling(run, prefix)
        if candidate is not None:
            paths.append(candidate)
    bounds = []
    for path in paths:
        timestamps = read_csv(path)["timestamp"]
        bounds.extend((timestamps.min(), timestamps.max()))
    index = pd.date_range(min(bounds).floor(FREQUENCY), max(bounds).ceil(FREQUENCY), freq=FREQUENCY, tz="UTC")
    streams = [
        event_stream(run, index),
        numeric_stream(sibling(run, "od_readings_filtered-"), "normalized_od_reading", "norm_od", index),
        numeric_stream(sibling(run, "growth_rates-"), "rate", "growth_rate", index),
        numeric_stream(sibling(run, "co2_readings-"), "co2_reading_ppm", "co2_ppm", index),
        angled_od_stream(sibling(run, "od_readings-"), index),
        spectrum_stream(sibling(run, "as7341_spectrum_readings-"), index),
    ]
    table = pd.concat(streams, axis=1).reset_index(names="timestamp")
    table = add_reactor_state(table)
    table.insert(0, "run_key", f"{run.modality}:{run.condition}:{run.run_id}")
    table.insert(1, "modality", run.modality)
    table.insert(2, "condition", run.condition)
    table.insert(3, "run_id", run.run_id)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "artifacts" / "dataset")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "runs"
    tables_dir.mkdir(exist_ok=True)

    manifest_rows, coverage_rows = [], []
    for run in discover_runs(args.data_root):
        table = make_table(run)
        run_key = table.at[0, "run_key"]
        table.to_csv(tables_dir / f"{run_key.replace(':', '__')}.csv", index=False)
        manifest_rows.append({
            "run_key": run_key, "modality": run.modality, "condition": run.condition, "run_id": run.run_id,
            "source_directory": str(run.path), "stock_concentration": "", "concentration_units": "",
            "initial_volume_ml": "", "notes": "",
        })
        coverage_rows.append({
            "run_key": run_key, "rows": len(table),
            "start": table["timestamp"].min(), "end": table["timestamp"].max(),
            **{column: int(table[column].sum()) for column in table if column.startswith("observed_")},
        })
        print(f"built {run_key}: {len(table)} bins")
    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "experiment_manifest.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(args.output_dir / "coverage_report.csv", index=False)


if __name__ == "__main__":
    main()
