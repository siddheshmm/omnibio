"""Plot 12-hour forward ODE simulation comparison on unseen 4D Rössler experiment."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

import sys
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[0] / "dfba"))
sys.path.append(str(HERE.parents[0] / "damn"))

from multimodal_dfba import MultimodalDFBA
from multimodal_damn_ode import MultimodalDAMN
from dataset import NormalizationScalers, SENSOR_TARGETS, INPUT_COLUMNS
from benchmark_rossler_multimodal import load_rossler_data, DEFAULT_ROSSLER_DIR, DEFAULT_DAMN_CKPT, OUTPUT_DIR


def plot_forward_simulation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_rossler_data(DEFAULT_ROSSLER_DIR)

    steps_per_cycle = 5
    n_cycles = data["total_cycles"]
    timeline_hours = np.arange(n_cycles) * (25.0 / 60.0)

    # 1. dFBA simulation
    dfba_model = MultimodalDFBA()
    run_df = pd.DataFrame(data["u_trajectory"], columns=INPUT_COLUMNS)
    run_df["volume_ml"] = 13.5
    run_df["norm_od"] = data["real_od"]
    dfba_sim = dfba_model.simulate_trajectory(run_df, initial_od=float(data["real_od"][0]))
    dfba_cycle_od = dfba_sim["norm_od"].to_numpy(float).reshape(n_cycles, steps_per_cycle).mean(axis=1)

    # 2. dAMN simulation
    ckpt = torch.load(DEFAULT_DAMN_CKPT, map_location="cpu")
    scalers_dict = ckpt["scalers"]
    scalers = NormalizationScalers(
        sensor_mean=np.array(scalers_dict["sensor_mean"], dtype=np.float32),
        sensor_std=np.array(scalers_dict["sensor_std"], dtype=np.float32),
        sensor_names=scalers_dict["sensor_names"],
        input_mean=np.array(scalers_dict["input_mean"], dtype=np.float32),
        input_std=np.array(scalers_dict["input_std"], dtype=np.float32),
        input_names=scalers_dict["input_names"],
    )

    damn_model = MultimodalDAMN(
        num_sensors=14,
        num_metabolites=6,
        input_dim=6,
        latent_dim=8,
        hidden_dim=64,
    )
    damn_model.load_state_dict(ckpt["model_state_dict"])
    damn_model.eval()

    u_tensor = torch.tensor(data["u_trajectory"], dtype=torch.float32).unsqueeze(0)
    od_tensor = torch.tensor([[data["real_od"][0]]], dtype=torch.float32)
    vol_tensor = torch.tensor([[13.5]], dtype=torch.float32)

    with torch.no_grad():
        damn_pred_norm, _, _ = damn_model.rollout(u_tensor, od_tensor, vol_tensor)

    damn_sensors = scalers.denormalize_sensors(damn_pred_norm.squeeze(0)).cpu().numpy()
    damn_cycle_od = damn_sensors[:, 0].reshape(n_cycles, steps_per_cycle).mean(axis=1)

    real_cycle_od = data["real_od"].reshape(n_cycles, steps_per_cycle).mean(axis=1)

    # Metrics
    dfba_mae = mean_absolute_error(real_cycle_od, dfba_cycle_od)
    dfba_r2 = r2_score(real_cycle_od, dfba_cycle_od)
    damn_mae = mean_absolute_error(real_cycle_od, damn_cycle_od)
    damn_r2 = r2_score(real_cycle_od, damn_cycle_od)

    plt.figure(figsize=(11, 6))
    plt.plot(timeline_hours, real_cycle_od, color="#1e40af", linewidth=2.6, marker="o", markersize=5, label="Physical Yeast Culture (Observed)")
    plt.plot(timeline_hours, damn_cycle_od, color="#dc2626", linewidth=2.2, linestyle="--", marker="s", markersize=5, label=f"Multimodal dAMN (MAE={damn_mae:.4f} OD, R²={damn_r2:.4f})")
    plt.plot(timeline_hours, dfba_cycle_od, color="#7c3aed", linewidth=2.0, linestyle="-.", marker="^", markersize=5, label=f"Multimodal dFBA (MAE={dfba_mae:.4f} OD, R²={dfba_r2:.4f})")

    plt.title("4D Hyperchaotic Rössler (12.1h): Continuous Forward Optical Density Simulation", fontsize=13, fontweight="bold")
    plt.xlabel("Timeline (Hours)", fontsize=11)
    plt.ylabel("Normalized Optical Density (OD)", fontsize=11)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_plot = OUTPUT_DIR / "rossler_forward_simulation.png"
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"Forward simulation plot saved to: {out_plot}")


if __name__ == "__main__":
    plot_forward_simulation()
