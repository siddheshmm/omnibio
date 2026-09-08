"""Create visual evaluation reports from direct NARX metrics and recursive rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CORE_TARGETS = ("norm_od", "od_45", "od_90", "od_135", "growth_rate", "co2_ppm")


def load_rollouts(rollout_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in rollout_dir.glob("*.csv"):
        if path.name == "rollout_metrics.csv":
            continue
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        if not frame.empty:
            tables[str(frame.at[0, "run_key"])] = frame
    return tables


def direct_horizon_chart(metrics: pd.DataFrame, output: Path) -> None:
    selected = metrics.loc[metrics["target"].isin(CORE_TARGETS)].copy()
    selected["mae_ratio"] = selected["mae"] / selected["persistence_mae"]
    pivot = selected.pivot(index="target", columns="horizon_minutes", values="mae_ratio").reindex(CORE_TARGETS)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn_r", vmin=0.6, vmax=1.6)
    ax.set_xticks(range(len(pivot.columns)), [f"{value} min" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Direct forecast MAE relative to persistence\n(< 1.0 means NARX is better)")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            if pd.notna(value):
                ax.text(column, row, f"{value:.2f}×", ha="center", va="center", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("NARX MAE / persistence MAE")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def rollout_trajectory_chart(rollouts: dict[str, pd.DataFrame], output: Path) -> None:
    runs = list(rollouts.items())
    targets = ("norm_od", "od_45", "co2_ppm")
    fig, axes = plt.subplots(len(runs), len(targets), figsize=(15, max(3.0 * len(runs), 8)), sharex=False, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, (run_key, frame) in enumerate(runs):
        minutes = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds() / 60
        for column, target in enumerate(targets):
            ax = axes[row, column]
            ax.plot(minutes, frame[target], color="#1f2937", lw=1.5, label="observed")
            ax.plot(minutes, frame[f"predicted_{target}"], color="#dc2626", lw=1.3, ls="--", label="recursive NARX")
            if row == 0:
                ax.set_title(target)
            if column == 0:
                ax.set_ylabel(run_key.replace(":", "\n"), fontsize=8)
            if row == len(runs) - 1:
                ax.set_xlabel("minutes from run start")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Recursive NARX rollouts on held-out dosing runs", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=SCRIPT_DIR / "artifacts" / "model")
    parser.add_argument("--rollout-dir", type=Path, default=SCRIPT_DIR / "artifacts" / "rollouts")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "artifacts" / "plots")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.model_dir / "metrics.csv")
    direct_horizon_chart(metrics, args.output_dir / "direct_horizon_vs_persistence.png")
    rollouts = load_rollouts(args.rollout_dir)
    if not rollouts:
        raise FileNotFoundError("No rollout tables found; run evaluate_rollouts.py first.")
    rollout_trajectory_chart(rollouts, args.output_dir / "held_out_recursive_rollouts.png")
    print(f"wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
