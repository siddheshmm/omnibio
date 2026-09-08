"""Compare dFBA biomass proxy against observed Pioreactor sensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_BATCH_DIR = HERE / "artifacts" / "batch"


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if mask.sum() < 2:
        return {"n": int(mask.sum()), "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    y = observed[mask]
    yhat = predicted[mask]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "mae": float(np.mean(np.abs(y - yhat))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def compare_file(path: Path) -> dict[str, float | str]:
    frame = pd.read_csv(path)
    row: dict[str, float | str] = {"run": path.stem.replace("_dfba", "")}
    if {"predicted_norm_od", "observed_norm_od"}.issubset(frame.columns):
        od_metrics = metrics(frame["observed_norm_od"].to_numpy(), frame["predicted_norm_od"].to_numpy())
        row.update({f"norm_od_{key}": value for key, value in od_metrics.items()})
    row["final_biomass_gdw_per_l"] = float(frame["biomass_gdw_per_l"].iloc[-1])
    row["final_glucose_mmol_per_l"] = float(frame["glucose_mmol_per_l"].iloc[-1])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_BATCH_DIR / "observation_metrics.csv")
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("*_dfba.csv"))
    if not files:
        raise FileNotFoundError(f"No dFBA outputs found in {args.input_dir}")
    summary = pd.DataFrame([compare_file(path) for path in files])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"wrote observation metrics for {len(summary)} runs to {args.output}")


if __name__ == "__main__":
    main()
