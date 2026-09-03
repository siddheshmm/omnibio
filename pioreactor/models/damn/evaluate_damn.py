"""Evaluate continuous full-trajectory rollouts of dAMN across all runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import yaml

from damn_ode import DAMN
from dataset import NormalizationScalers

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "artifacts" / "model" / "damn_checkpoint.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "rollouts"


def evaluate_run(
    model: DAMN,
    run_df: pd.DataFrame,
    scalers: NormalizationScalers,
    sensor_names: list[str],
    input_names: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    n_steps = len(run_df)
    raw_u = run_df[input_names].to_numpy(dtype=np.float32)
    norm_u = scalers.normalize_inputs(raw_u)
    u_tensor = torch.tensor(norm_u, dtype=torch.float32).unsqueeze(0)  # (1, T, 4)

    # Initial state
    init_od = float(run_df["norm_od"].dropna().iloc[0]) if "norm_od" in run_df else 0.5
    init_biomass = torch.tensor([[max(init_od * 0.15, 0.01)]], dtype=torch.float32)
    init_volume = torch.tensor([[float(run_df["volume_ml"].iloc[0]) if "volume_ml" in run_df else 13.5]], dtype=torch.float32)

    with torch.no_grad():
        out = model(u_trajectory=u_tensor, init_biomass=init_biomass, init_volume=init_volume)

    # Denormalize predicted sensors back to physical units
    y_pred_norm = out["y_pred"].squeeze(0).cpu().numpy()  # (T, 14)
    y_pred_phys = scalers.denormalize_sensors(y_pred_norm)

    pred_df = pd.DataFrame(index=run_df.index)
    pred_df["timestamp"] = run_df["timestamp"]
    pred_df["run_key"] = run_df["run_key"] if "run_key" in run_df else ""
    pred_df["time_min"] = np.arange(n_steps) * float(model.dt_min)
    pred_df["biomass_gdw_per_l"] = out["biomass"].squeeze(0).squeeze(-1).cpu().numpy()
    pred_df["volume_ml"] = out["volume"].squeeze(0).squeeze(-1).cpu().numpy()
    pred_df["growth_rate_damn"] = out["growth_rate"].squeeze(0).squeeze(-1).cpu().numpy()

    # Add predicted and observed sensors
    metrics = {}
    obs_cols = [f"observed_{s}" for s in sensor_names]

    for idx, s_name in enumerate(sensor_names):
        pred_col = f"pred_{s_name}"
        obs_col = s_name
        pred_df[pred_col] = y_pred_phys[:, idx]
        pred_df[obs_col] = run_df[obs_col].to_numpy()

        # Compute metrics on valid observed values
        mask_col = f"observed_{s_name}"
        if mask_col in run_df.columns:
            valid_mask = run_df[mask_col].fillna(False).to_numpy(dtype=bool)
        else:
            valid_mask = np.isfinite(run_df[obs_col].to_numpy())

        if np.sum(valid_mask) > 1:
            y_true = run_df.loc[valid_mask, obs_col].to_numpy(dtype=np.float64)
            y_pred = pred_df.loc[valid_mask, pred_col].to_numpy(dtype=np.float64)

            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            try:
                r2 = float(r2_score(y_true, y_pred))
            except ValueError:
                r2 = float("nan")

            metrics[f"{s_name}_mae"] = mae
            metrics[f"{s_name}_rmse"] = rmse
            metrics[f"{s_name}_r2"] = r2

    return pred_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    scaler_info = checkpoint["scalers"]

    scalers = NormalizationScalers(
        sensor_mean=np.array(scaler_info["sensor_mean"], dtype=np.float32),
        sensor_std=np.array(scaler_info["sensor_std"], dtype=np.float32),
        sensor_names=scaler_info["sensor_names"],
        input_mean=np.array(scaler_info["input_mean"], dtype=np.float32),
        input_std=np.array(scaler_info["input_std"], dtype=np.float32),
        input_names=scaler_info["input_names"],
    )

    amn_cfg = config.get("amn", {})
    model = DAMN(
        num_sensors=len(scalers.sensor_names),
        num_metabolites=6,
        input_dim=len(scalers.input_names),
        latent_dim=int(amn_cfg.get("latent_dim", 8)),
        hidden_dim=int(amn_cfg.get("hidden_dim", 64)),
        yield_glucose=0.0811,
        dt_min=float(config.get("bin_minutes", 5.0)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset_dir = (SCRIPT_DIR / config["dataset_dir"]).resolve()
    run_paths = sorted(dataset_dir.glob("*.csv"))

    summary_rows = []
    print(f"Evaluating {len(run_paths)} runs with dAMN continuous rollouts...")

    for r_path in run_paths:
        run_df = pd.read_csv(r_path, parse_dates=["timestamp"])
        run_key = run_df["run_key"].iloc[0] if "run_key" in run_df else r_path.stem
        condition = run_df["condition"].iloc[0] if "condition" in run_df else ""
        modality = run_df["modality"].iloc[0] if "modality" in run_df else ""

        pred_df, metrics = evaluate_run(
            model=model,
            run_df=run_df,
            scalers=scalers,
            sensor_names=scalers.sensor_names,
            input_names=scalers.input_names,
        )

        out_run_file = args.output_dir / f"{r_path.stem}_damn.csv"
        pred_df.to_csv(out_run_file, index=False)

        row = {
            "run": r_path.stem,
            "run_key": run_key,
            "condition": condition,
            "modality": modality,
            "steps": len(run_df),
            **metrics,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    metrics_file = args.output_dir / "rollout_metrics.csv"
    summary_df.to_csv(metrics_file, index=False)

    print(f"\nWrote rollouts to {args.output_dir}")
    print(f"Wrote rollout metrics to {metrics_file}")
    print("\nTop Lowest MAE on norm_od:")
    if "norm_od_mae" in summary_df.columns:
        print(summary_df.sort_values("norm_od_mae")[["run", "condition", "modality", "norm_od_mae", "norm_od_rmse", "norm_od_r2"]].head(10).to_string())


if __name__ == "__main__":
    main()
