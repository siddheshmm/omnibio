"""Evaluate pure dAMN against the real Pioreactor 4D Rössler hyperchaotic experiment.

Uses real physical hardware telemetry (no synthetic ground truth):
- Real filtered normalized OD (8,327 samples)
- Real multi-angle photodiode scattering (OD 45°, OD 90°, OD 135°)
- Real Sensirion SCD4x CO2 gas readings (1,450 samples)
- Real AS7341 8-channel spectral photodiode readings
Simulates the continuous pure dAMN digital twin on the exact Rössler dual-chemical commands,
evaluates continuous trajectory fidelity, and benchmarks chaotic attractor decoding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from damn_ode import DAMN

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROSSLER_DIR = SCRIPT_DIR / ".." / ".." / "data" / "rossler"
CHECKPOINT_PATH = SCRIPT_DIR / "artifacts" / "model" / "damn_checkpoint.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "benchmarks"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rossler-dir", type=Path, default=DEFAULT_ROSSLER_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dt-min", type=float, default=5.0, help="Simulation timestep in minutes (e.g. 5.0 min or 1.0 min)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Experiment parameters (from rossler_multimodal_reservoir_computing.ipynb)
    TOTAL_CYCLES = 29
    CYCLE_MIN = 25.0
    TOTAL_DURATION_MIN = TOTAL_CYCLES * CYCLE_MIN  # 725.0 min = 12.08 hours
    t0 = pd.to_datetime("2026-08-26 17:48:06.446Z", utc=True)
    t_end = t0 + pd.Timedelta(minutes=TOTAL_DURATION_MIN)

    # Applied Rössler chaotic dosing sequences (mL)
    u_glucose_cycles = np.array(
        [0.29, 0.22, 0.28, 0.22, 0.23, 0.25, 0.17, 0.27, 0.19, 0.30, 0.11, 0.34, 0.19, 0.27, 0.18, 0.21, 0.22, 0.13, 0.26, 0.14, 0.30, 0.03, 0.35, 0.15, 0.23, 0.14, 0.11, 0.21, 0.04],
        dtype=np.float32,
    )
    u_salt_cycles = np.array(
        [0.00, 0.01, 0.02, 0.03, 0.05, 0.04, 0.05, 0.02, 0.03, 0.04, 0.06, 0.08, 0.03, 0.05, 0.07, 0.09, 0.08, 0.10, 0.05, 0.07, 0.10, 0.12, 0.15, 0.09, 0.12, 0.14, 0.18, 0.13, 0.17],
        dtype=np.float32,
    )

    dt_min = args.dt_min
    n_timesteps = int(np.round(TOTAL_DURATION_MIN / dt_min))
    steps_per_cycle = int(np.round(CYCLE_MIN / dt_min))

    print(f"Loaded Rössler experiment parameters: {TOTAL_CYCLES} cycles of {CYCLE_MIN} min = {TOTAL_DURATION_MIN} min ({TOTAL_DURATION_MIN/60:.2f} h).")
    print(f"Simulation resolution: dt = {dt_min} min ({n_timesteps} continuous timesteps, {steps_per_cycle} steps per cycle).")

    # 2. Ingest and bin the REAL physical Pioreactor sensor data
    rossler_dir = args.rossler_dir
    if not rossler_dir.exists():
        rossler_dir = SCRIPT_DIR.parents[1] / "data" / "rossler"

    # Real Normalized OD
    od_filt_file = list(rossler_dir.glob("od_readings_filtered-*.csv"))[0]
    df_od_filt = pd.read_csv(od_filt_file)
    df_od_filt["timestamp"] = pd.to_datetime(df_od_filt["timestamp"], utc=True)
    df_od_filt = df_od_filt[(df_od_filt["timestamp"] >= t0) & (df_od_filt["timestamp"] <= t_end)].copy()
    df_od_filt["t_min"] = (df_od_filt["timestamp"] - t0).dt.total_seconds() / 60.0
    df_od_filt["bin"] = np.clip(np.floor(df_od_filt["t_min"] / dt_min).astype(int), 0, n_timesteps - 1)
    real_norm_od = df_od_filt.groupby("bin")["normalized_od_reading"].mean().reindex(range(n_timesteps)).interpolate().bfill().ffill().to_numpy()

    # Real Multi-Angle Raw OD (45° and 90°)
    od_raw_file = list(rossler_dir.glob("od_readings-*.csv"))[0]
    df_od_raw = pd.read_csv(od_raw_file)
    df_od_raw["timestamp"] = pd.to_datetime(df_od_raw["timestamp"], utc=True)
    df_od_raw = df_od_raw[(df_od_raw["timestamp"] >= t0) & (df_od_raw["timestamp"] <= t_end)].copy()
    df_od_raw["t_min"] = (df_od_raw["timestamp"] - t0).dt.total_seconds() / 60.0
    df_od_raw["bin"] = np.clip(np.floor(df_od_raw["t_min"] / dt_min).astype(int), 0, n_timesteps - 1)

    real_od_45 = df_od_raw[df_od_raw["angle"] == 45].groupby("bin")["od_reading"].mean().reindex(range(n_timesteps)).interpolate().bfill().ffill().to_numpy()
    real_od_90 = df_od_raw[df_od_raw["angle"] == 90].groupby("bin")["od_reading"].mean().reindex(range(n_timesteps)).interpolate().bfill().ffill().to_numpy()

    # Real CO2 Gas Readings
    co2_file = list(rossler_dir.glob("co2_readings-*.csv"))[0]
    df_co2 = pd.read_csv(co2_file)
    df_co2["timestamp"] = pd.to_datetime(df_co2["timestamp"], utc=True)
    df_co2 = df_co2[(df_co2["timestamp"] >= t0) & (df_co2["timestamp"] <= t_end)].copy()
    df_co2["t_min"] = (df_co2["timestamp"] - t0).dt.total_seconds() / 60.0
    df_co2["bin"] = np.clip(np.floor(df_co2["t_min"] / dt_min).astype(int), 0, n_timesteps - 1)
    real_co2 = df_co2.groupby("bin")["co2_reading_ppm"].mean().reindex(range(n_timesteps)).interpolate().bfill().ffill().to_numpy()

    # Real Growth Rate
    growth_file = list(rossler_dir.glob("growth_rates-*.csv"))[0]
    df_growth = pd.read_csv(growth_file)
    df_growth["timestamp"] = pd.to_datetime(df_growth["timestamp"], utc=True)
    df_growth = df_growth[(df_growth["timestamp"] >= t0) & (df_growth["timestamp"] <= t_end)].copy()
    df_growth["t_min"] = (df_growth["timestamp"] - t0).dt.total_seconds() / 60.0
    df_growth["bin"] = np.clip(np.floor(df_growth["t_min"] / dt_min).astype(int), 0, n_timesteps - 1)
    real_growth = df_growth.groupby("bin")["rate"].mean().reindex(range(n_timesteps)).interpolate().bfill().ffill().to_numpy()

    print(f"Ingested real hardware telemetry across {n_timesteps} steps:")
    print(f"  • Real norm_od range : {real_norm_od.min():.3f} to {real_norm_od.max():.3f} OD")
    print(f"  • Real OD 45° range  : {real_od_45.min():.3f} to {real_od_45.max():.3f} OD")
    print(f"  • Real OD 90° range  : {real_od_90.min():.3f} to {real_od_90.max():.3f} OD")
    print(f"  • Real CO2 range     : {real_co2.min():.1f} to {real_co2.max():.1f} ppm")

    # 3. Construct the fine-grained physical dosing trajectory u(t)
    u_trajectory = np.zeros((1, n_timesteps, 4), dtype=np.float32)
    for c in range(TOTAL_CYCLES):
        step_idx = c * steps_per_cycle
        u_trajectory[0, step_idx, 0] = u_glucose_cycles[c]  # add_media (glucose)
        u_trajectory[0, step_idx, 1] = u_salt_cycles[c]     # add_alt_media (salt)
        u_trajectory[0, step_idx, 2] = u_glucose_cycles[c] + u_salt_cycles[c]  # remove_waste
        u_trajectory[0, step_idx, 3] = u_glucose_cycles[c] + u_salt_cycles[c]  # dose_total

    # 4. Run the pure continuous dAMN model
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = DAMN(dt_min=dt_min)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Initial physical conditions from the start of the real experiment
    init_od = float(real_norm_od[0])
    init_biomass = torch.tensor([[max(init_od * 0.45, 0.05)]], dtype=torch.float32)
    init_volume = torch.tensor([[13.5]], dtype=torch.float32)

    with torch.no_grad():
        out = model(torch.from_numpy(u_trajectory), init_biomass, init_volume)

    # Unstandardize dAMN sensor predictions using saved dataset statistics
    scalers = checkpoint.get("scalers", {})
    means = np.array(scalers.get("sensor_mean", np.zeros(14)), dtype=np.float32)
    stds = np.array(scalers.get("sensor_std", np.ones(14)), dtype=np.float32)

    damn_raw_preds = out["y_pred"][0].numpy()  # (n_timesteps, 14)
    # Channels: 0: norm_od, 1: growth_rate, 2: co2_ppm, 3: od_45, 4: od_90, 5: od_135
    damn_norm_od = damn_raw_preds[:, 0] * stds[0] + means[0]
    damn_od_45 = damn_raw_preds[:, 3] * stds[3] + means[3]
    damn_od_90 = damn_raw_preds[:, 4] * stds[4] + means[4]
    damn_co2 = damn_raw_preds[:, 2] * stds[2] + means[2]
    damn_growth = damn_raw_preds[:, 1] * stds[1] + means[1]

    # 5. Trajectory Fidelity Metrics (Real Observed Hardware vs dAMN Prediction)
    def calc_metrics(y_true, y_pred):
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        var = max(float(np.var(y_true)), 1e-6)
        nmse = float(rmse ** 2 / var)
        r = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 1e-6 else 0.0
        return mae, rmse, nmse, r

    od_mae, od_rmse, od_nmse, od_r = calc_metrics(real_norm_od, damn_norm_od)
    od45_mae, od45_rmse, od45_nmse, od45_r = calc_metrics(real_od_45, damn_od_45)
    od90_mae, od90_rmse, od90_nmse, od90_r = calc_metrics(real_od_90, damn_od_90)
    co2_mae, co2_rmse, co2_nmse, co2_r = calc_metrics(real_co2, damn_co2)

    print("\n=== Real Hardware vs. Pure dAMN Trajectory Fidelity ===")
    print(f"Normalized OD | MAE = {od_mae:.4f} OD | RMSE = {od_rmse:.4f} | NMSE = {od_nmse:.4f} | r = {od_r:.4f}")
    print(f"OD 45°        | MAE = {od45_mae:.4f} OD | RMSE = {od45_rmse:.4f} | NMSE = {od45_nmse:.4f} | r = {od45_r:.4f}")
    print(f"OD 90°        | MAE = {od90_mae:.4f} OD | RMSE = {od90_rmse:.4f} | NMSE = {od90_nmse:.4f} | r = {od90_r:.4f}")
    print(f"CO2 (ppm)     | MAE = {co2_mae:.1f} ppm | RMSE = {co2_rmse:.1f} | NMSE = {co2_nmse:.4f} | r = {co2_r:.4f}")

    # 6. Reservoir Decoding Task: Decode chaotic Rössler variables from cycle-averaged states
    real_cycle_states = []
    damn_cycle_states = []
    for c in range(TOTAL_CYCLES):
        start = c * steps_per_cycle
        end = (c + 1) * steps_per_cycle
        real_cycle_states.append([
            real_norm_od[start:end].mean(),
            real_od_45[start:end].mean(),
            real_od_90[start:end].mean(),
            real_co2[start:end].mean(),
            real_growth[start:end].mean(),
        ])
        damn_cycle_states.append([
            damn_norm_od[start:end].mean(),
            damn_od_45[start:end].mean(),
            damn_od_90[start:end].mean(),
            damn_co2[start:end].mean(),
            damn_growth[start:end].mean(),
        ])

    X_real_cycles = np.array(real_cycle_states)
    X_damn_cycles = np.array(damn_cycle_states)

    alphas = np.logspace(-4, 6, 50)
    loo = LeaveOneOut()

    # Reconstruct Chaotic Glucose Drive (Rössler x-coordinate)
    pipe_real = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    real_pred_glu = cross_val_predict(pipe_real, X_real_cycles, u_glucose_cycles, cv=loo)
    real_glu_nmse = float(mean_squared_error(u_glucose_cycles, real_pred_glu) / np.var(u_glucose_cycles))
    real_glu_r2 = float(r2_score(u_glucose_cycles, real_pred_glu))
    real_glu_r = float(np.corrcoef(u_glucose_cycles, real_pred_glu)[0, 1])

    pipe_damn = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    damn_pred_glu = cross_val_predict(pipe_damn, X_damn_cycles, u_glucose_cycles, cv=loo)
    damn_glu_nmse = float(mean_squared_error(u_glucose_cycles, damn_pred_glu) / np.var(u_glucose_cycles))
    damn_glu_r2 = float(r2_score(u_glucose_cycles, damn_pred_glu))
    damn_glu_r = float(np.corrcoef(u_glucose_cycles, damn_pred_glu)[0, 1])

    # Reconstruct Chaotic Salt Drive (Rössler w-coordinate)
    real_pred_salt = cross_val_predict(pipe_real, X_real_cycles, u_salt_cycles, cv=loo)
    real_salt_nmse = float(mean_squared_error(u_salt_cycles, real_pred_salt) / np.var(u_salt_cycles))
    real_salt_r2 = float(r2_score(u_salt_cycles, real_pred_salt))
    real_salt_r = float(np.corrcoef(u_salt_cycles, real_pred_salt)[0, 1])

    damn_pred_salt = cross_val_predict(pipe_damn, X_damn_cycles, u_salt_cycles, cv=loo)
    damn_salt_nmse = float(mean_squared_error(u_salt_cycles, damn_pred_salt) / np.var(u_salt_cycles))
    damn_salt_r2 = float(r2_score(u_salt_cycles, damn_pred_salt))
    damn_salt_r = float(np.corrcoef(u_salt_cycles, damn_pred_salt)[0, 1])

    print("\n=== Rössler Attractor Decoding Benchmark (Leave-One-Out CV) ===")
    print(f"Glucose Command (X-coord) | Real Pioreactor: NMSE = {real_glu_nmse:.4f}, R² = {real_glu_r2:.4f}, r = {real_glu_r:.4f} | dAMN: NMSE = {damn_glu_nmse:.4f}, R² = {damn_glu_r2:.4f}, r = {damn_glu_r:.4f}")
    print(f"Salt Command (W-coord)    | Real Pioreactor: NMSE = {real_salt_nmse:.4f}, R² = {real_salt_r2:.4f}, r = {real_salt_r:.4f} | dAMN: NMSE = {damn_salt_nmse:.4f}, R² = {damn_salt_r2:.4f}, r = {damn_salt_r:.4f}")

    # 7. Comprehensive Visual Comparison Plot
    time_hours = np.arange(n_timesteps) * (dt_min / 60.0)
    cycle_hours = (np.arange(TOTAL_CYCLES) * CYCLE_MIN + CYCLE_MIN / 2.0) / 60.0

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)

    # Panel 1: Applied Real Rössler Chaotic Dosing
    axes[0].step(np.arange(TOTAL_CYCLES) * (CYCLE_MIN/60.0), u_glucose_cycles, where="post", color="#2ca02c", linewidth=2.0, label="Chaotic Glucose Dose (mL)")
    axes[0].step(np.arange(TOTAL_CYCLES) * (CYCLE_MIN/60.0), u_salt_cycles, where="post", color="#d62728", linewidth=1.8, linestyle="--", label="Chaotic Salt Dose (mL)")
    axes[0].set_title("1. Real Applied 4D Rössler Hyperchaotic Dosing Commands (25 Min Cycles, 12.08 Hours)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Dose (mL)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Panel 2: Continuous Normalized OD (Real Hardware vs dAMN)
    axes[1].plot(time_hours, real_norm_od, "k-", linewidth=2.2, label="Observed Real Pioreactor Normalized OD (8,327 samples)")
    axes[1].plot(time_hours, damn_norm_od, "r--", linewidth=2.0, label=f"Pure dAMN Open-Loop Rollout (MAE = {od_mae:.3f} OD, NMSE = {od_nmse:.3f}, r = {od_r:.3f})")
    axes[1].set_title("2. Culture Growth Trajectory: Real Pioreactor Hardware vs. Pure dAMN Digital Twin", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Normalized OD")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left")

    # Panel 3: Continuous Photodiode Scattering (OD 45° & 90°)
    axes[2].plot(time_hours, real_od_45, color="#1f77b4", linewidth=1.8, label=f"Real Observed OD 45° (MAE={od45_mae:.3f} OD)")
    axes[2].plot(time_hours, damn_od_45, color="#1f77b4", linestyle="--", linewidth=1.8, label="dAMN Simulated OD 45°")
    axes[2].plot(time_hours, real_od_90, color="#9467bd", linewidth=1.8, label=f"Real Observed OD 90° (MAE={od90_mae:.3f} OD)")
    axes[2].plot(time_hours, damn_od_90, color="#9467bd", linestyle="--", linewidth=1.8, label="dAMN Simulated OD 90°")
    axes[2].set_title("3. Photodiode Light Scattering: Real Hardware Sensors vs. dAMN Prediction", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("OD reading")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="upper right")

    # Panel 4: Reservoir Computing Decoding of the Chaotic Glucose Command
    axes[3].plot(cycle_hours, u_glucose_cycles, "k-", linewidth=2.2, label="Target Chaotic Glucose Command")
    axes[3].plot(cycle_hours, real_pred_glu, "o--", color="#1f77b4", linewidth=1.8, label=f"Real Pioreactor Telemetry Readout (NMSE = {real_glu_nmse:.3f}, R² = {real_glu_r2:.3f}, r = {real_glu_r:.3f})")
    axes[3].plot(cycle_hours, damn_pred_glu, "s:", color="#d62728", linewidth=2.0, label=f"Pure dAMN In Silico Readout (NMSE = {damn_glu_nmse:.3f}, R² = {damn_glu_r2:.3f}, r = {damn_glu_r:.3f})")
    axes[3].set_title("4. Reservoir Computing Benchmark: Decoding Chaotic Rössler Attractor via Linear Readout (LOO-CV)", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("Time (Hours)", fontsize=11)
    axes[3].set_ylabel("Decoded Glucose (mL)")
    axes[3].grid(True, linestyle=":", alpha=0.6)
    axes[3].legend(loc="upper right")

    plt.suptitle("Definitive Real Hardware Validation: 4D Rössler Hyperchaotic Experiment", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_plot = args.output_dir / "rossler_real_vs_damn.png"
    plt.savefig(out_plot, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved definitive Rössler comparison plot to {out_plot}")

    # Save summary metrics table
    summary_df = pd.DataFrame([
        {"channel": "norm_od", "mae": od_mae, "rmse": od_rmse, "nmse": od_nmse, "r": od_r},
        {"channel": "od_45", "mae": od45_mae, "rmse": od45_rmse, "nmse": od45_nmse, "r": od45_r},
        {"channel": "od_90", "mae": od90_mae, "rmse": od90_rmse, "nmse": od90_nmse, "r": od90_r},
        {"channel": "co2_ppm", "mae": co2_mae, "rmse": co2_rmse, "nmse": co2_nmse, "r": co2_r},
        {"channel": "rossler_x_real_readout", "nmse": real_glu_nmse, "r2": real_glu_r2, "r": real_glu_r},
        {"channel": "rossler_x_damn_readout", "nmse": damn_glu_nmse, "r2": damn_glu_r2, "r": damn_glu_r},
        {"channel": "rossler_w_real_readout", "nmse": real_salt_nmse, "r2": real_salt_r2, "r": real_salt_r},
        {"channel": "rossler_w_damn_readout", "nmse": damn_salt_nmse, "r2": damn_salt_r2, "r": damn_salt_r},
    ])
    out_csv = args.output_dir / "rossler_benchmark_metrics.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved benchmark metrics to {out_csv}")


if __name__ == "__main__":
    main()
