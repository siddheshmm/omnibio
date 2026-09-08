"""Evaluate trained Multimodal dAMN on open-loop continuous rollouts across all runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from multimodal_damn_ode import MultimodalDAMN
from dataset import SENSOR_TARGETS, INPUT_COLUMNS, NormalizationScalers

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = SCRIPT_DIR.parents[0] / "artifacts" / "dataset" / "runs"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "artifacts" / "model" / "multimodal_damn_best.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "rollouts"


def evaluate_rollouts(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    scalers_dict = ckpt["scalers"]
    scalers = NormalizationScalers(
        sensor_mean=np.array(scalers_dict["sensor_mean"], dtype=np.float32),
        sensor_std=np.array(scalers_dict["sensor_std"], dtype=np.float32),
        sensor_names=scalers_dict["sensor_names"],
        input_mean=np.array(scalers_dict["input_mean"], dtype=np.float32),
        input_std=np.array(scalers_dict["input_std"], dtype=np.float32),
        input_names=scalers_dict["input_names"],
    )

    model = MultimodalDAMN(
        num_sensors=14,
        num_metabolites=6,
        input_dim=6,
        latent_dim=8,
        hidden_dim=64,
        yield_glucose=0.0811,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    csv_paths = sorted(runs_dir.glob("*.csv"))
    metrics_summary = []

    print(f"Evaluating {len(csv_paths)} continuous rollouts...")

    for path in csv_paths:
        df = pd.read_csv(path)
        u_arr = df[INPUT_COLUMNS].to_numpy(dtype=np.float32)
        y_true = df[SENSOR_TARGETS].to_numpy(dtype=np.float32)

        init_od = float(df["norm_od"].dropna().iloc[0]) if "norm_od" in df and df["norm_od"].dropna().shape[0] > 0 else 0.5
        init_vol = float(df["volume_ml"].iloc[0]) if "volume_ml" in df else 13.5

        u_tensor = torch.tensor(u_arr, dtype=torch.float32).unsqueeze(0)
        od_tensor = torch.tensor([[init_od]], dtype=torch.float32)
        vol_tensor = torch.tensor([[init_vol]], dtype=torch.float32)

        with torch.no_grad():
            pred_sensors_norm, pred_states, _ = model.rollout(u_tensor, od_tensor, vol_tensor)

        pred_sensors = scalers.denormalize_sensors(pred_sensors_norm.squeeze(0)).cpu().numpy()

        out_df = pd.DataFrame(pred_sensors, columns=[f"{s}_pred" for s in SENSOR_TARGETS])
        out_df["timestamp"] = df["timestamp"]
        out_df["norm_od_true"] = df["norm_od"]
        out_df["growth_rate_true"] = df["growth_rate"]
        out_df["co2_ppm_true"] = df["co2_ppm"]
        out_df.to_csv(output_dir / f"{path.stem}__damn_rollout.csv", index=False)

        # Metrics for norm_od
        valid_mask = np.isfinite(y_true[:, 0]) & np.isfinite(pred_sensors[:, 0])
        if valid_mask.sum() > 5:
            od_mae = mean_absolute_error(y_true[valid_mask, 0], pred_sensors[valid_mask, 0])
            od_r2 = r2_score(y_true[valid_mask, 0], pred_sensors[valid_mask, 0])
        else:
            od_mae, od_r2 = np.nan, np.nan

        metrics_summary.append(
            {
                "run": path.name,
                "steps": len(df),
                "od_mae": od_mae,
                "od_r2": od_r2,
            }
        )

    summary_df = pd.DataFrame(metrics_summary)
    summary_df.to_csv(output_dir / "multimodal_damn_metrics.csv", index=False)
    print(f"Evaluation complete! Summary saved to {output_dir / 'multimodal_damn_metrics.csv'}")
    print(f"Mean OD MAE across all 39 runs: {summary_df['od_mae'].mean():.4f}")
    print(f"Median OD R2 across all 39 runs: {summary_df['od_r2'].median():.4f}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    evaluate_rollouts(args.runs_dir, args.checkpoint, args.output_dir)
