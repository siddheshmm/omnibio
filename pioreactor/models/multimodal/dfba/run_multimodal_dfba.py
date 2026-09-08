"""Run multimodal dFBA batch evaluation across all 39 training runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from multimodal_dfba import MultimodalDFBA

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = SCRIPT_DIR.parents[0] / "artifacts" / "dataset" / "runs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "batch"


def run_batch(runs_dir: Path = DEFAULT_RUNS_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(runs_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} runs for multimodal dFBA evaluation.")

    model = MultimodalDFBA()
    summary = []

    for idx, path in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] Simulating {path.name}...")
        df = pd.read_csv(path)
        sim = model.simulate_trajectory(df)

        out_sim = output_dir / f"{path.stem}__dfba.csv"
        sim.to_csv(out_sim, index=False)

        # Compute OD error metrics against physical culture
        valid = df["norm_od"].dropna()
        if not valid.empty and len(valid) == len(sim):
            y_true = df["norm_od"].to_numpy(float)
            y_pred = sim["norm_od"].to_numpy(float)
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() > 5:
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                r2 = r2_score(y_true[mask], y_pred[mask])
            else:
                mae, r2 = np.nan, np.nan
        else:
            mae, r2 = np.nan, np.nan

        summary.append(
            {
                "run_file": path.name,
                "rows": len(df),
                "mae_norm_od": mae,
                "r2_norm_od": r2,
                "sim_path": str(out_sim),
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "multimodal_dfba_summary.csv", index=False)
    print(f"Evaluation complete! Summary saved to {output_dir / 'multimodal_dfba_summary.csv'}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_batch(args.runs_dir, args.output_dir)
