"""Create visual evaluation reports from dFBA batch observation metrics and trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_DIR = SCRIPT_DIR / "artifacts" / "batch"
DEFAULT_METRICS_PATH = DEFAULT_BATCH_DIR / "observation_metrics.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "plots"

COLOR_OBSERVED = "#1f2937"    # Dark slate
COLOR_PREDICTED = "#dc2626"   # Red / crimson
COLOR_GLUCOSE = "#059669"     # Emerald green
COLOR_BIOMASS = "#2563eb"     # Blue


def parse_run_identifier(run_key: str) -> dict[str, str]:
    """Parse run string into modality, condition, and run_id."""
    parts = run_key.replace(".csv", "").split("__")
    if len(parts) >= 3:
        return {"modality": parts[0], "condition": parts[1], "run_id": parts[2], "label": f"{parts[0]}:{parts[1]}"}
    if len(parts) == 2:
        return {"modality": parts[0], "condition": parts[1], "run_id": "", "label": f"{parts[0]}:{parts[1]}"}
    return {"modality": "unknown", "condition": run_key, "run_id": "", "label": run_key}


def load_trajectories(batch_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all dFBA run output tables from the batch directory."""
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(batch_dir.glob("*_dfba.csv")):
        run_name = path.stem.replace("_dfba", "")
        df = pd.read_csv(path)
        if not df.empty:
            tables[run_name] = df
    return tables


def plot_metrics_overview(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Generate summary dashboard of dFBA observation metrics."""
    df = metrics_df.copy()
    parsed = [parse_run_identifier(r) for r in df["run"]]
    df["modality"] = [p["modality"] for p in parsed]
    df["condition"] = [p["condition"] for p in parsed]
    df["label"] = [f"{p['label']}\n({p['run_id'][-6:]})" if p['run_id'] else p['label'] for p in parsed]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Panel 1: MAE & RMSE per run
    ax1 = axes[0, 0]
    y_pos = np.arange(len(df))
    has_large_outlier = (df["norm_od_rmse"] > 5.0).any()
    if has_large_outlier:
        ax1.set_xscale("symlog", linthresh=0.1)
        ax1.set_xlabel("Error (OD units, symlog scale)")
    else:
        ax1.set_xlabel("Error (OD units)")

    width = 0.38
    ax1.barh(y_pos - width / 2, df["norm_od_mae"], height=width, label="MAE", color="#3b82f6", alpha=0.9)
    ax1.barh(y_pos + width / 2, df["norm_od_rmse"], height=width, label="RMSE", color="#ef4444", alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df["label"], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_title("Normalized OD Prediction Error by Run", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    for idx, row in df.iterrows():
        if row["norm_od_mae"] > 2.0:
            ax1.text(row["norm_od_mae"] * 1.1, idx - width / 2, f"outlier: {row['norm_od_mae']:.1f}",
                     va="center", fontsize=8, color="#991b1b", fontweight="bold")

    # Panel 2: Mean MAE grouped by dosing condition
    ax2 = axes[0, 1]
    condition_stats = df.groupby("condition")["norm_od_mae"].agg(["mean", "count"]).reset_index()
    condition_stats = condition_stats.sort_values("mean", ascending=True)
    bars = ax2.bar(condition_stats["condition"], condition_stats["mean"], color="#10b981", edgecolor="#047857", alpha=0.85)
    ax2.set_ylabel("Mean MAE (OD units)", fontsize=9)
    ax2.set_title("Average Error by Nutrient / Stressor Condition", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    if (condition_stats["mean"] > 2.0).any():
        ax2.set_yscale("symlog", linthresh=0.1)
    for bar, count, mean_val in zip(bars, condition_stats["count"], condition_stats["mean"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean_val:.2f}\n(n={count})",
                 ha="center", va="bottom", fontsize=8)

    # Panel 3: Final Biomass vs Glucose
    ax3 = axes[1, 0]
    conditions = df["condition"].unique()
    cmap = plt.get_cmap("tab10")
    for i, cond in enumerate(conditions):
        sub = df[df["condition"] == cond]
        ax3.scatter(sub["final_biomass_gdw_per_l"], sub["final_glucose_mmol_per_l"],
                    label=cond, color=cmap(i), s=65, edgecolors="#1f2937", linewidth=0.8)
    ax3.set_xlabel("Final Biomass (gDW / L)", fontsize=9)
    ax3.set_ylabel("Final Extracellular Glucose (mmol / L)", fontsize=9)
    ax3.set_title("Metabolic Endpoint: Biomass vs Glucose Depletion", fontsize=11, fontweight="bold")
    ax3.grid(alpha=0.3, linestyle="--")
    ax3.legend(loc="best", fontsize=8)

    # Panel 4: Goodness-of-Fit Summary
    ax4 = axes[1, 1]
    clean_r2 = df["norm_od_r2"].clip(lower=-25.0)
    colors = ["#22c55e" if r >= 0 else "#f97316" for r in clean_r2]
    ax4.bar(range(len(df)), clean_r2, color=colors, alpha=0.8, edgecolor="#374151")
    ax4.axhline(0, color="#111827", linestyle="-", linewidth=0.8)
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=7.5)
    ax4.set_ylabel("Normalized OD R² (clipped at -25)", fontsize=9)
    ax4.set_title("Coefficient of Determination (R²) Baseline", fontsize=11, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Yeast dFBA Digital Twin: Batch Observation Metrics Evaluation", fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trajectory_grid(
    trajectories: dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    output_path: Path,
    max_runs: int = 6,
) -> None:
    """Generate multi-panel comparison of observed vs predicted trajectories for lowest MAE runs."""
    # Merge MAE into trajectory list and sort by lowest MAE
    mae_lookup = dict(zip(metrics_df["run"], metrics_df["norm_od_mae"]))
    rmse_lookup = dict(zip(metrics_df["run"], metrics_df["norm_od_rmse"]))

    sorted_keys = sorted(
        trajectories.keys(),
        key=lambda k: mae_lookup.get(k, float("inf")),
    )
    selected_keys = sorted_keys[:max_runs]
    runs = [(k, trajectories[k]) for k in selected_keys]

    n_runs = len(runs)
    cols = 2
    rows = int(np.ceil(n_runs / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3.5 * rows, 8)), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx, (run_key, frame) in enumerate(runs):
        row_idx = idx // cols
        col_idx = idx % cols
        ax = axes[row_idx, col_idx]

        time_min = frame["time_min"] if "time_min" in frame else np.arange(len(frame)) * 5.0

        # Primary axis: Observed vs Predicted OD
        line_obs = ax.plot(time_min, frame["observed_norm_od"], color=COLOR_OBSERVED, lw=1.8, label="Observed sensor norm_od")
        line_pred = ax.plot(time_min, frame["predicted_norm_od"], color=COLOR_PREDICTED, lw=1.6, ls="--", label="dFBA predicted norm_od")
        ax.set_ylabel("Normalized OD", color=COLOR_OBSERVED, fontsize=9.5)
        ax.grid(alpha=0.25, linestyle=":")

        # Zoom y-axis to data range so variations are visible
        valid_obs = frame["observed_norm_od"].dropna()
        valid_pred = frame["predicted_norm_od"].dropna()
        if not valid_obs.empty and not valid_pred.empty:
            y_min = min(valid_obs.min(), valid_pred.min())
            y_max = max(valid_obs.max(), valid_pred.max())
            y_range = max(y_max - y_min, 0.05)
            ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

        # Secondary axis: Extracellular Glucose or Biomass
        ax2 = ax.twinx()
        if "glucose_mmol_per_l" in frame:
            line_met = ax2.plot(time_min, frame["glucose_mmol_per_l"], color=COLOR_GLUCOSE, lw=1.2, ls=":", label="Glucose (mmol/L)")
            ax2.set_ylabel("Glucose (mmol/L)", color=COLOR_GLUCOSE, fontsize=8)
        elif "biomass_gdw_per_l" in frame:
            line_met = ax2.plot(time_min, frame["biomass_gdw_per_l"], color=COLOR_BIOMASS, lw=1.2, ls=":", label="Biomass (gDW/L)")
            ax2.set_ylabel("Biomass (gDW/L)", color=COLOR_BIOMASS, fontsize=8)
        else:
            line_met = []

        parsed = parse_run_identifier(run_key)
        mae = mae_lookup.get(run_key, np.nan)
        rmse = rmse_lookup.get(run_key, np.nan)
        ax.set_title(
            f"Rank #{idx+1} (Lowest MAE) | {parsed['modality'].upper()} : {parsed['condition'].capitalize()} ({parsed['run_id']})",
            fontsize=9.5, fontweight="bold"
        )
        ax.set_xlabel("Time (min)", fontsize=8.5)

        # Add annotation box with metrics
        ax.text(
            0.98, 0.94,
            f"MAE: {mae:.3f} OD\nRMSE: {rmse:.3f} OD",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.9),
        )

        if idx == 0:
            lines = line_obs + line_pred + line_met
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper left", fontsize=8, framealpha=0.9)

    for idx in range(n_runs, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Lowest MAE Runs: dFBA In Silico Dynamics vs Observed Pioreactor Telemetry", fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_parity(trajectories: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Generate observed vs predicted parity scatter plot across all batch observations."""
    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)

    conditions: dict[str, list[float]] = {}
    for run_key, frame in trajectories.items():
        if not {"observed_norm_od", "predicted_norm_od"}.issubset(frame.columns):
            continue
        parsed = parse_run_identifier(run_key)
        cond = parsed["condition"]
        valid = frame["observed_norm_od"].notna() & frame["predicted_norm_od"].notna()
        obs = frame.loc[valid, "observed_norm_od"].tolist()
        pred = frame.loc[valid, "predicted_norm_od"].tolist()
        if cond not in conditions:
            conditions[cond] = {"obs": [], "pred": []}
        conditions[cond]["obs"].extend(obs)
        conditions[cond]["pred"].extend(pred)

    all_obs, all_pred = [], []
    cmap = plt.get_cmap("tab10")
    for i, (cond, data) in enumerate(conditions.items()):
        obs_arr = np.array(data["obs"])
        pred_arr = np.array(data["pred"])
        all_obs.extend(obs_arr)
        all_pred.extend(pred_arr)
        ax.scatter(obs_arr, pred_arr, label=cond, color=cmap(i), alpha=0.6, s=28, edgecolors="none")

    all_obs = np.array(all_obs)
    all_pred = np.array(all_pred)
    min_val = min(all_obs.min(), all_pred.min()) if len(all_obs) else 0.0
    max_val = max(all_obs.max(), all_pred.max()) if len(all_obs) else 3.0

    ax.plot([min_val, max_val], [min_val, max_val], color="#ef4444", lw=1.5, ls="--", label="Ideal Parity (y=x)")
    ax.set_xlabel("Observed Normalized OD", fontsize=10)
    ax.set_ylabel("dFBA Predicted Normalized OD", fontsize=10)
    ax.set_title("Parity Plot: Observed vs dFBA Predicted Optical Density", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=8.5)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {args.metrics_path}; run compare_observations.py first.")

    metrics_df = pd.read_csv(args.metrics_path)
    trajectories = load_trajectories(args.batch_dir)

    print(f"Loaded metrics for {len(metrics_df)} runs and {len(trajectories)} trajectory tables.")

    metrics_plot = args.output_dir / "dfba_metrics_summary.png"
    plot_metrics_overview(metrics_df, metrics_plot)
    print(f"Wrote metrics overview to {metrics_plot}")

    if trajectories:
        trajectory_plot = args.output_dir / "dfba_trajectories_comparison.png"
        plot_trajectory_grid(trajectories, metrics_df, trajectory_plot, max_runs=6)
        print(f"Wrote trajectories grid to {trajectory_plot}")

        parity_plot = args.output_dir / "dfba_parity_fit.png"
        plot_parity(trajectories, parity_plot)
        print(f"Wrote parity plot to {parity_plot}")


if __name__ == "__main__":
    main()
