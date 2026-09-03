"""Visualizations for dAMN multi-sensor rollouts and biological reservoir dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROLLOUTS_DIR = SCRIPT_DIR / "artifacts" / "rollouts"
DEFAULT_PLOTS_DIR = SCRIPT_DIR / "artifacts" / "plots"


def plot_multisensor_trajectories(rollouts_dir: Path, plots_dir: Path, n_runs: int = 4) -> None:
    metrics_file = rollouts_dir / "rollout_metrics.csv"
    if not metrics_file.exists():
        print(f"Metrics file {metrics_file} not found.")
        return

    m_df = pd.read_csv(metrics_file)
    top_runs = m_df.sort_values("norm_od_mae").head(n_runs)["run"].tolist()

    fig, axes = plt.subplots(n_runs, 3, figsize=(18, 4 * n_runs), sharex="row")
    if n_runs == 1:
        axes = np.expand_dims(axes, 0)

    for i, run_name in enumerate(top_runs):
        csv_file = rollouts_dir / f"{run_name}_damn.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        t = df["time_min"].to_numpy()

        # Panel 1: norm_od (observed vs dAMN)
        ax1 = axes[i, 0]
        ax1.plot(t, df["norm_od"], label="Observed Sensor", color="#1f77b4", linewidth=2.0)
        ax1.plot(t, df["pred_norm_od"], label="dAMN Predicted", color="#d62728", linestyle="--", linewidth=2.0)
        mae = float(df["norm_od"] - df["pred_norm_od"]).__abs__() if False else np.nanmean(np.abs(df["norm_od"] - df["pred_norm_od"]))
        ax1.set_title(f"{run_name[:35]}...\nNormalized OD (MAE: {mae:.3f})", fontsize=11, fontweight="bold")
        ax1.set_ylabel("norm_od")
        ax1.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax1.legend(loc="best", frameon=True)

        # Panel 2: od_45 and od_90
        ax2 = axes[i, 1]
        if "od_45" in df and "pred_od_45" in df:
            ax2.plot(t, df["od_45"], label="Observed OD 45°", color="#2ca02c", linewidth=1.8)
            ax2.plot(t, df["pred_od_45"], label="dAMN OD 45°", color="#ff7f0e", linestyle="--", linewidth=1.8)
        if "od_90" in df and "pred_od_90" in df:
            ax2.plot(t, df["od_90"], label="Observed OD 90°", color="#9467bd", alpha=0.7, linewidth=1.5)
            ax2.plot(t, df["pred_od_90"], label="dAMN OD 90°", color="#8c564b", linestyle=":", linewidth=1.5)
        ax2.set_title("Photodiode Scattering (OD 45° & 90°)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("OD reading")
        ax2.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax2.legend(loc="best", frameon=True)

        # Panel 3: co2_ppm and latent growth
        ax3 = axes[i, 2]
        if "co2_ppm" in df and "pred_co2_ppm" in df:
            ax3.plot(t, df["co2_ppm"], label="Observed CO2 (ppm)", color="#333333", linewidth=1.8)
            ax3.plot(t, df["pred_co2_ppm"], label="dAMN CO2", color="#e377c2", linestyle="--", linewidth=1.8)
            ax3.set_ylabel("CO2 (ppm)")
        ax3.set_title("Gas Exchange (CO2 ppm)", fontsize=11, fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax3.legend(loc="best", frameon=True)

        axes[i, 0].set_xlabel("Time (min)")
        axes[i, 1].set_xlabel("Time (min)")
        axes[i, 2].set_xlabel("Time (min)")

    plt.suptitle("dAMN Multi-Sensor Closed-Loop Continuous Rollouts", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_file = plots_dir / "damn_multisensor_trajectories.png"
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote multisensor trajectories to {out_file}")


def plot_metrics_overview(rollouts_dir: Path, plots_dir: Path) -> None:
    metrics_file = rollouts_dir / "rollout_metrics.csv"
    if not metrics_file.exists():
        return
    df = pd.read_csv(metrics_file)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. MAE by condition
    ax1 = axes[0]
    cond_grp = df.groupby("condition")["norm_od_mae"].mean().sort_values()
    colors = plt.cm.Set2(np.linspace(0, 1, len(cond_grp)))
    cond_grp.plot(kind="bar", ax=ax1, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_title("Mean norm_od MAE by Condition", fontsize=12, fontweight="bold")
    ax1.set_ylabel("MAE")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. norm_od MAE distribution
    ax2 = axes[1]
    ax2.hist(df["norm_od_mae"].dropna(), bins=12, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax2.axvline(df["norm_od_mae"].median(), color="red", linestyle="--", label=f"Median: {df['norm_od_mae'].median():.3f}")
    ax2.set_title("norm_od MAE Distribution Across Runs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("norm_od MAE")
    ax2.set_ylabel("Run Count")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. od_45 vs co2 MAE scatter
    ax3 = axes[2]
    if "od_45_mae" in df.columns and "co2_ppm_mae" in df.columns:
        ax3.scatter(df["od_45_mae"], df["co2_ppm_mae"], color="#2ca02c", s=70, edgecolors="black", alpha=0.8)
        ax3.set_title("OD 45° vs CO2 Prediction Errors", fontsize=12, fontweight="bold")
        ax3.set_xlabel("OD 45° MAE")
        ax3.set_ylabel("CO2 ppm MAE")
        ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_file = plots_dir / "damn_metrics_overview.png"
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote metrics overview to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts-dir", type=Path, default=DEFAULT_ROLLOUTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    args = parser.parse_args()

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_multisensor_trajectories(args.rollouts_dir, args.plots_dir, n_runs=4)
    plot_metrics_overview(args.rollouts_dir, args.plots_dir)


if __name__ == "__main__":
    main()
