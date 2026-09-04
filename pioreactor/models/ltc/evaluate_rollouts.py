"""Full multi-hour continuous rollout evaluation across all runs for LTC Digital Twin.

Computes MAE, RMSE, and R2 across all 14 physical sensor targets for every run,
and writes artifacts/rollouts/rollout_metrics.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

from dataset import NormalizationScalers, SENSOR_TARGETS, INPUT_COLUMNS
from ltc_model import LTCBioreactorTwin

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "artifacts" / "dataset" / "manifest.csv"
CHECKPOINT_PATH = SCRIPT_DIR / "artifacts" / "model" / "ltc_best_checkpoint.pt"
SCALERS_PATH = SCRIPT_DIR / "artifacts" / "dataset" / "scalers.json"
OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "rollouts"


def evaluate_single_run(
    model: LTCBioreactorTwin,
    run_df: pd.DataFrame,
    scalers: NormalizationScalers,
) -> tuple[pd.DataFrame, dict[str, float]]:
    n_steps = len(run_df)
    u_raw = run_df[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    u_norm = np.nan_to_num(scalers.normalize_inputs(u_raw), nan=0.0)
    u_tensor = torch.from_numpy(u_norm).unsqueeze(0)

    init_od_val = float(run_df["norm_od"].dropna().iloc[0]) if "norm_od" in run_df and not run_df["norm_od"].dropna().empty else 0.5
    init_vol_val = float(run_df["volume_ml"].iloc[0]) if "volume_ml" in run_df and not run_df["volume_ml"].empty else 13.5

    with torch.no_grad():
        out = model(
            u_seq=u_tensor,
            init_od=torch.tensor([[init_od_val]], dtype=torch.float32),
            init_vol=torch.tensor([[init_vol_val]], dtype=torch.float32),
        )

    y_pred_norm = out["y_pred"].squeeze(0).numpy()
    y_pred_phys = scalers.denormalize_sensors(y_pred_norm)

    pred_df = pd.DataFrame(index=run_df.index)
    pred_df["timestamp"] = run_df["timestamp"]
    pred_df["run_key"] = run_df["run_key"]
    pred_df["volume_ml"] = out["volume"].squeeze(0).squeeze(-1).numpy()

    metrics: dict[str, float] = {}
    for idx, s_name in enumerate(SENSOR_TARGETS):
        pred_col = f"pred_{s_name}"
        obs_col = s_name
        pred_df[pred_col] = y_pred_phys[:, idx]
        pred_df[obs_col] = run_df[obs_col].to_numpy()

        mask_col = f"observed_{s_name}"
        if mask_col in run_df.columns:
            valid_mask = run_df[mask_col].fillna(False).to_numpy(dtype=bool)
        else:
            valid_mask = np.isfinite(run_df[obs_col].to_numpy())

        if np.sum(valid_mask) > 1:
            y_true = run_df.loc[valid_mask, obs_col].to_numpy(dtype=np.float64)
            y_pred = pred_df.loc[valid_mask, pred_col].to_numpy(dtype=np.float64)

            metrics[f"{s_name}_mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics[f"{s_name}_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            try:
                metrics[f"{s_name}_r2"] = float(r2_score(y_true, y_pred))
            except ValueError:
                metrics[f"{s_name}_r2"] = float("nan")

    return pred_df, metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(DEFAULT_MANIFEST)
    scalers = NormalizationScalers.load(SCALERS_PATH)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cfg = ckpt.get("config", {})
    m_cfg = cfg.get("model", {})

    model = LTCBioreactorTwin(
        input_dim=m_cfg.get("input_dim", 6),
        hidden_dim=m_cfg.get("hidden_dim", 32),
        num_sensors=m_cfg.get("num_sensors", 14),
        unfolding_steps=m_cfg.get("unfolding_steps", 2),
        dt_min=5.0,
        tau_min=m_cfg.get("tau_min", 1.0),
        tau_max=m_cfg.get("tau_max", 60.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_metrics = []
    print(f"Evaluating LTC rollouts across {len(manifest)} runs:")

    for idx, row in manifest.iterrows():
        run_file = Path(row["path"])
        run_df = pd.read_csv(run_file)

        pred_df, metrics = evaluate_single_run(model, run_df, scalers)
        out_csv = OUTPUT_DIR / f"{Path(row['filename']).stem}__ltc_rollout.csv"
        pred_df.to_csv(out_csv, index=False)

        metrics_entry = {
            "run_key": row["run_key"],
            "modality": row["modality"],
            "input_type": row["input_type"],
            "condition": row["condition"],
            "run_id": row["run_id"],
            "steps": len(run_df),
            **metrics,
        }
        all_metrics.append(metrics_entry)
        od_mae = metrics.get("norm_od_mae", float("nan"))
        co2_mae = metrics.get("co2_ppm_mae", float("nan"))
        print(f"  [{idx + 1:02d}/{len(manifest):02d}] {row['run_key']:40s} | norm_od MAE: {od_mae:.4f} | CO2 MAE: {co2_mae:.1f}")

    df_summary = pd.DataFrame(all_metrics)
    summary_path = OUTPUT_DIR / "rollout_metrics.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved rollout metrics summary to: {summary_path}")


if __name__ == "__main__":
    main()
