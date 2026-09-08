"""Run dFBA across all canonical NARX dosing run tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_dfba import DEFAULT_CONFIG, DEFAULT_MODEL, DEFAULT_OUTPUT, load_config, load_run, simulate


HERE = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = HERE.parent / "narx" / "artifacts" / "dataset" / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "batch")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of runs")
    args = parser.parse_args()

    run_tables = sorted(args.runs_dir.glob("*.csv"))
    if args.limit > 0:
        run_tables = run_tables[: args.limit]
    if not run_tables:
        raise FileNotFoundError(f"No run tables found in {args.runs_dir}")

    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for run_table in run_tables:
        print(f"simulating {run_table.name}")
        try:
            run = load_run(run_table)
            result, diagnostics = simulate(args.model, run, config)
            stem = run_table.stem
            result.to_csv(args.output_dir / f"{stem}_dfba.csv", index=False)
            with (args.output_dir / f"{stem}_diagnostics.json").open("w", encoding="utf-8") as handle:
                json.dump(diagnostics, handle, indent=2)
            row = {
                "run_table": run_table.name,
                "condition": diagnostics["condition"],
                "points": len(result),
                "solver_success": bool(result["solver_success"].all()),
                "final_biomass_gdw_per_l": float(result["biomass_gdw_per_l"].iloc[-1]),
                "final_volume_ml": float(result["volume_ml"].iloc[-1]),
                "status": "ok",
            }
        except Exception as exc:  # noqa: BLE001 - batch runner should continue
            row = {"run_table": run_table.name, "status": "error", "error": str(exc)}
            print(f"  failed: {exc}")
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "batch_summary.csv", index=False)
    print(f"wrote batch summary for {len(summary_rows)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
