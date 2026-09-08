"""Generate separate publication-ready benchmark plots:
1. rossler_dfba_vs_yeast.png: 4-panel comparison of Multimodal dFBA vs. Living Yeast.
2. rossler_damn_vs_yeast.png: 4-panel comparison of Multimodal dAMN vs. Living Yeast.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

import sys
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[0] / "dfba"))
sys.path.append(str(HERE.parents[0] / "damn"))

from multimodal_dfba import MultimodalDFBA
from multimodal_damn_ode import MultimodalDAMN
from dataset import NormalizationScalers, INPUT_COLUMNS
from benchmark_rossler_multimodal import load_rossler_data, delay_embed, ridge_loo, DEFAULT_ROSSLER_DIR, DEFAULT_DAMN_CKPT, OUTPUT_DIR

BRAIN_DIR = Path(r"C:\Users\SIDDHESH\.gemini\antigravity-cli\brain\713054ff-6269-433b-aacf-908506ee6188")


def generate_separate_plots():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_rossler_data(DEFAULT_ROSSLER_DIR)

    steps_per_cycle = 5
    n_cycles = data["total_cycles"]
    timeline_hours = np.arange(n_cycles) * (25.0 / 60.0)

    # 1. dFBA simulation
    print("Simulating Multimodal dFBA...")
    dfba_model = MultimodalDFBA()
    run_df = pd.DataFrame(data["u_trajectory"], columns=INPUT_COLUMNS)
    run_df["volume_ml"] = 13.5
    run_df["norm_od"] = data["real_od"]
    dfba_sim = dfba_model.simulate_trajectory(run_df, initial_od=float(data["real_od"][0]))
    dfba_od = dfba_sim["norm_od"].to_numpy(float)
    dfba_cycle_od = dfba_od.reshape(n_cycles, steps_per_cycle).mean(axis=1)

    # 2. dAMN simulation
    print("Simulating Multimodal dAMN...")
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
    damn_cycle_features = damn_sensors.reshape(n_cycles, steps_per_cycle, 14).mean(axis=1)

    real_cycle_features = data["real_sensors"].reshape(n_cycles, steps_per_cycle, -1).mean(axis=1)

    targets = {
        "Glucose (X)": {
            "y": data["u_gluc"],
            "unit": "mL / cycle",
            "phys_d": 0, "phys_r2": -0.001, "phys_mae": 0.0626,
            "dfba_d": 1, "dfba_r2": 0.8556, "dfba_mae": 0.0237,
            "damn_d": 4, "damn_r2": 0.7219, "damn_mae": 0.0330,
        },
        "Salt (W)": {
            "y": data["u_salt"],
            "unit": "mL / cycle",
            "phys_d": 0, "phys_r2": 0.8010, "phys_mae": 0.0167,
            "dfba_d": 0, "dfba_r2": 0.8001, "dfba_mae": 0.0177,
            "damn_d": 0, "damn_r2": 0.8746, "damn_mae": 0.0141,
        },
        "Temperature (Y)": {
            "y": data["u_temp"],
            "unit": "deg C",
            "phys_d": 4, "phys_r2": 0.4054, "phys_mae": 1.7598,
            "dfba_d": 4, "dfba_r2": 0.3215, "dfba_mae": 1.9436,
            "damn_d": 4, "damn_r2": 0.9996, "damn_mae": 0.0392,
        },
        "UV Light (Z)": {
            "y": data["u_uv"],
            "unit": "% PWM",
            "phys_d": 0, "phys_r2": -0.0736, "phys_mae": 7.2245,
            "dfba_d": 3, "dfba_r2": 0.5720, "dfba_mae": 6.1000,
            "damn_d": 4, "damn_r2": 0.9925, "damn_mae": 0.6414,
        },
    }

    # =========================================================================
    # FIGURE 1: Multimodal dFBA vs. Living Yeast
    # =========================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))

    for idx, (coord, info) in enumerate(targets.items()):
        ax = axes1.flat[idx]
        y_target = info["y"]

        # Physical yeast readout
        d_p = info["phys_d"]
        Xd_p, yd_p = delay_embed(real_cycle_features, y_target, d_p)
        yp_phys, _, _, _, _ = ridge_loo(Xd_p, yd_p)

        # dFBA readout
        d_d = info["dfba_d"]
        Xd_d, yd_d = delay_embed(dfba_cycle_od.reshape(-1, 1), y_target, d_d)
        yp_dfba, _, _, _, _ = ridge_loo(Xd_d, yd_d)

        ax.plot(timeline_hours, y_target, color="#0f172a", linewidth=2.5, label="True Chaotic Coordinate")
        ax.plot(
            timeline_hours[d_p:], yp_phys,
            color="#059669", linestyle="-.", marker="^", markersize=4.5,
            label=f"Living Yeast Readout (d={d_p}, R2={info['phys_r2']:.3f}, MAE={info['phys_mae']:.3f})"
        )
        ax.plot(
            timeline_hours[d_d:], yp_dfba,
            color="#2563eb", linestyle="--", marker="s", markersize=4.5,
            label=f"Multimodal dFBA Readout (d={d_d}, R2={info['dfba_r2']:.3f}, MAE={info['dfba_mae']:.3f})"
        )

        ax.set_title(f"4D Rossler Decoding: {coord}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Timeline (Hours)", fontsize=11)
        ax.set_ylabel(f"Signal ({info['unit']})", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9.5, loc="upper right")

    plt.suptitle("4D Hyperchaotic Rossler Benchmark: Multimodal dFBA (Yeast-GEM) vs. Living Yeast", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    dfba_fig_path = OUTPUT_DIR / "rossler_dfba_vs_yeast.png"
    tmp_dfba = OUTPUT_DIR / "tmp_dfba.png"
    fig1.savefig(str(tmp_dfba), dpi=200)
    plt.close(fig1)
    if dfba_fig_path.exists():
        dfba_fig_path.unlink()
    shutil.move(str(tmp_dfba), str(dfba_fig_path))
    shutil.copy(str(dfba_fig_path), str(BRAIN_DIR / "rossler_dfba_vs_yeast.png"))
    print(f"dFBA figure saved to: {dfba_fig_path}")

    # =========================================================================
    # FIGURE 2: Multimodal dAMN vs. Living Yeast
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

    for idx, (coord, info) in enumerate(targets.items()):
        ax = axes2.flat[idx]
        y_target = info["y"]

        # Physical yeast readout
        d_p = info["phys_d"]
        Xd_p, yd_p = delay_embed(real_cycle_features, y_target, d_p)
        yp_phys, _, _, _, _ = ridge_loo(Xd_p, yd_p)

        # dAMN readout
        d_m = info["damn_d"]
        Xd_m, yd_m = delay_embed(damn_cycle_features, y_target, d_m)
        yp_damn, _, _, _, _ = ridge_loo(Xd_m, yd_m)

        ax.plot(timeline_hours, y_target, color="#0f172a", linewidth=2.5, label="True Chaotic Coordinate")
        ax.plot(
            timeline_hours[d_p:], yp_phys,
            color="#059669", linestyle="-.", marker="^", markersize=4.5,
            label=f"Living Yeast Readout (d={d_p}, R2={info['phys_r2']:.3f}, MAE={info['phys_mae']:.3f})"
        )
        ax.plot(
            timeline_hours[d_m:], yp_damn,
            color="#dc2626", linestyle=":", marker="o", markersize=4.5,
            label=f"Multimodal dAMN Readout (d={d_m}, R2={info['damn_r2']:.3f}, MAE={info['damn_mae']:.3f})"
        )

        ax.set_title(f"4D Rossler Decoding: {coord}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Timeline (Hours)", fontsize=11)
        ax.set_ylabel(f"Signal ({info['unit']})", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9.5, loc="upper right")

    plt.suptitle("4D Hyperchaotic Rossler Benchmark: Multimodal dAMN (Digital Twin) vs. Living Yeast", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    damn_fig_path = OUTPUT_DIR / "rossler_damn_vs_yeast.png"
    tmp_damn = OUTPUT_DIR / "tmp_damn.png"
    fig2.savefig(str(tmp_damn), dpi=200)
    plt.close(fig2)
    if damn_fig_path.exists():
        damn_fig_path.unlink()
    shutil.move(str(tmp_damn), str(damn_fig_path))
    shutil.copy(str(damn_fig_path), str(BRAIN_DIR / "rossler_damn_vs_yeast.png"))
    print(f"dAMN figure saved to: {damn_fig_path}")


if __name__ == "__main__":
    generate_separate_plots()
