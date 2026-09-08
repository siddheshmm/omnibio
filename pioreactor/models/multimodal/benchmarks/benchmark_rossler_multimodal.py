"""Multimodal 4D Rössler Hyperchaotic Benchmark for Multimodal dFBA and dAMN.

Evaluates both models zero-shot on the held-out continuous 12-hour 4D Rössler experiment:
Actuator Channels:
1. Glucose dosing u_glc (Media pump, mL)
2. Salt osmotic dosing u_salt (Alt-media pump, mL)
3. Temperature schedule u_temp (°C, 24°C - 35°C)
4. UV irradiation u_uv (LED PWM, 0% - 45%)

Evaluations:
- Task A: Continuous forward trajectory prediction (norm_od tracking).
- Task B: 4-Coordinate Reservoir Computing reconstruction (LOO-CV R², NMSE, MAE across delay embeddings d=0..5).
"""

from __future__ import annotations

import argparse
from pathlib import Path
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
from dataset import NormalizationScalers, SENSOR_TARGETS, INPUT_COLUMNS

DEFAULT_ROSSLER_DIR = HERE.parents[2] / "data" / "rossler"
DEFAULT_DAMN_CKPT = HERE.parents[0] / "damn" / "artifacts" / "model" / "multimodal_damn_best.pt"
OUTPUT_DIR = HERE / "artifacts"


def delay_embed(X: np.ndarray, y: np.ndarray, delay: int):
    if delay == 0:
        return X.copy(), y.copy()
    rows = []
    for t in range(delay, len(X)):
        rows.append(np.concatenate([X[t - lag] for lag in range(delay + 1)]))
    return np.asarray(rows), y[delay:]


def ridge_loo(X: np.ndarray, y: np.ndarray, alphas: np.ndarray = np.logspace(-4, 4, 50)):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rcv = RidgeCV(alphas=alphas, cv=None)
    rcv.fit(Xs, y)
    model = Ridge(alpha=rcv.alpha_)
    yp = cross_val_predict(model, Xs, y, cv=LeaveOneOut())
    nmse = mean_squared_error(y, yp) / (np.var(y) + 1e-8)
    mae = mean_absolute_error(y, yp)
    r2 = r2_score(y, yp)
    return yp, nmse, mae, r2, rcv.alpha_


def load_rossler_data(rossler_dir: Path) -> dict[str, Any]:
    """Load and synchronize the real 4D Rössler experiment data."""
    TOTAL_CYCLES = 29
    CYCLE_MIN = 25.0
    dt_min = 5.0
    steps_per_cycle = int(CYCLE_MIN / dt_min)
    n_timesteps = TOTAL_CYCLES * steps_per_cycle

    # 4 Applied chaotic control signals
    u_gluc = np.array([0.29, 0.22, 0.28, 0.22, 0.23, 0.25, 0.17, 0.27, 0.19, 0.30, 0.11, 0.34, 0.19, 0.27, 0.18, 0.21, 0.22, 0.13, 0.26, 0.14, 0.30, 0.03, 0.35, 0.15, 0.23, 0.14, 0.11, 0.21, 0.04], dtype=np.float32)
    u_salt = np.array([0.00, 0.01, 0.02, 0.03, 0.05, 0.04, 0.05, 0.02, 0.03, 0.04, 0.06, 0.08, 0.03, 0.05, 0.07, 0.09, 0.08, 0.10, 0.05, 0.07, 0.10, 0.12, 0.15, 0.09, 0.12, 0.14, 0.18, 0.13, 0.17], dtype=np.float32)
    u_temp = np.array([29.7, 31.1, 28.6, 32.5, 27.0, 32.6, 27.4, 31.1, 29.3, 30.6, 30.7, 28.1, 31.2, 28.3, 33.2, 26.0, 33.5, 26.2, 31.4, 29.2, 30.7, 31.2, 27.1, 32.2, 27.1, 35.0, 24.0, 33.3, 27.1], dtype=np.float32)
    u_uv = np.array([5.0, 1.3, 3.3, 1.3, 1.3, 2.6, 0.6, 20.0, 0.8, 12.8, 0.3, 45.0, 0.9, 2.8, 0.7, 1.0, 1.3, 0.3, 4.8, 0.4, 8.4, 0.0, 42.7, 0.5, 1.3, 0.4, 0.2, 1.3, 0.0], dtype=np.float32)

    # 5-minute binned input trajectory
    u_traj = np.zeros((n_timesteps, 6), dtype=np.float32)
    for c in range(TOTAL_CYCLES):
        s_idx = c * steps_per_cycle
        # Pulse enters at start of 25-min cycle
        u_traj[s_idx, 0] = u_gluc[c]
        u_traj[s_idx, 1] = u_salt[c]
        u_traj[s_idx, 2] = u_gluc[c] + u_salt[c]
        u_traj[s_idx, 3] = u_gluc[c] + u_salt[c]

        # Temp and UV are maintained across the 25-min cycle
        for step in range(steps_per_cycle):
            u_traj[s_idx + step, 4] = u_temp[c]
            u_traj[s_idx + step, 5] = u_uv[c]

    # Load real physical yeast optical density
    od_path = next(rossler_dir.glob("od_readings_filtered-*.csv"))
    od_df = pd.read_csv(od_path)
    od_df["timestamp"] = pd.to_datetime(od_df["timestamp"], utc=True)
    od_res = od_df.set_index("timestamp")["normalized_od_reading"].resample("5min").mean()
    real_od = od_res.iloc[:n_timesteps].to_numpy(float)

    # Real cycle-mean observations for reservoir decoding
    real_features_list = []
    # Angled ODs
    angled_path = next(rossler_dir.glob("od_readings-*.csv"))
    ang_df = pd.read_csv(angled_path)
    ang_df["timestamp"] = pd.to_datetime(ang_df["timestamp"], utc=True)
    for ang in [45, 90, 135]:
        sub = ang_df[ang_df["angle"] == ang].set_index("timestamp")["od_reading"].resample("5min").mean()
        real_features_list.append(sub.iloc[:n_timesteps].ffill().bfill().to_numpy(float))

    # Spectral bands
    spec_path = next(rossler_dir.glob("as7341_spectrum_readings-*.csv"))
    spec_df = pd.read_csv(spec_path)
    spec_df["timestamp"] = pd.to_datetime(spec_df["timestamp"], utc=True)
    for band in [415, 445, 480, 515, 555, 590, 630, 680]:
        sub = spec_df[spec_df["band"] == band].set_index("timestamp")["reading"].resample("5min").mean()
        real_features_list.append(sub.iloc[:n_timesteps].ffill().bfill().to_numpy(float))

    real_sensors = np.column_stack(real_features_list)  # (145, 11)

    return {
        "total_cycles": TOTAL_CYCLES,
        "n_timesteps": n_timesteps,
        "u_gluc": u_gluc,
        "u_salt": u_salt,
        "u_temp": u_temp,
        "u_uv": u_uv,
        "u_trajectory": u_traj,
        "real_od": real_od,
        "real_sensors": real_sensors,
    }


def run_benchmark(rossler_dir: Path = DEFAULT_ROSSLER_DIR, damn_ckpt: Path = DEFAULT_DAMN_CKPT):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_rossler_data(rossler_dir)

    steps_per_cycle = 5
    n_cycles = data["total_cycles"]
    timeline_hours = np.arange(n_cycles) * (25.0 / 60.0)

    # 1. Simulate Multimodal dFBA
    print("\n--- Simulating Multimodal dFBA on 4D Rössler ---")
    dfba_model = MultimodalDFBA()
    run_df = pd.DataFrame(data["u_trajectory"], columns=INPUT_COLUMNS)
    run_df["volume_ml"] = 13.5
    run_df["norm_od"] = data["real_od"]
    dfba_sim = dfba_model.simulate_trajectory(run_df, initial_od=float(data["real_od"][0]))
    dfba_od = dfba_sim["norm_od"].to_numpy(float)

    # Downsample dFBA to cycle-level
    dfba_cycle_od = dfba_od.reshape(n_cycles, steps_per_cycle).mean(axis=1)

    # 2. Simulate Multimodal dAMN
    print("--- Simulating Multimodal dAMN on 4D Rössler ---")
    ckpt = torch.load(damn_ckpt, map_location="cpu")
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
        damn_pred_norm, damn_states, _ = damn_model.rollout(u_tensor, od_tensor, vol_tensor)

    damn_sensors = scalers.denormalize_sensors(damn_pred_norm.squeeze(0)).cpu().numpy()
    damn_od = damn_sensors[:, 0]
    damn_cycle_od = damn_od.reshape(n_cycles, steps_per_cycle).mean(axis=1)
    damn_cycle_features = damn_sensors.reshape(n_cycles, steps_per_cycle, 14).mean(axis=1)

    # Real culture cycle features
    real_cycle_od = data["real_od"].reshape(n_cycles, steps_per_cycle).mean(axis=1)
    real_cycle_features = data["real_sensors"].reshape(n_cycles, steps_per_cycle, -1).mean(axis=1)

    # 3. Dynamic Forward Growth Metrics
    dfba_mae = mean_absolute_error(real_cycle_od, dfba_cycle_od)
    dfba_r2 = r2_score(real_cycle_od, dfba_cycle_od)
    damn_mae = mean_absolute_error(real_cycle_od, damn_cycle_od)
    damn_r2 = r2_score(real_cycle_od, damn_cycle_od)

    print(f"\nForward Optical Density Tracking (12.1h Continuous Horizon):")
    print(f"  Multimodal dFBA: MAE = {dfba_mae:.4f} OD | R2 = {dfba_r2:.4f}")
    print(f"  Multimodal dAMN: MAE = {damn_mae:.4f} OD | R2 = {damn_r2:.4f}")

    # 4. Reservoir Computing 4-Coordinate Decoding Sweeps
    targets = {
        "Glucose (X)": data["u_gluc"],
        "Salt (W)": data["u_salt"],
        "Temperature (Y)": data["u_temp"],
        "UV Light (Z)": data["u_uv"],
    }

    results = []

    for name, y_target in targets.items():
        # Sweep delays d=0..4
        for d in range(5):
            # Physical yeast
            Xd_phys, yd_phys = delay_embed(real_cycle_features, y_target, d)
            yp_phys, nmse_phys, mae_phys, r2_phys, _ = ridge_loo(Xd_phys, yd_phys)

            # dFBA (using OD proxy state)
            Xd_dfba, yd_dfba = delay_embed(dfba_cycle_od.reshape(-1, 1), y_target, d)
            yp_dfba, nmse_dfba, mae_dfba, r2_dfba, _ = ridge_loo(Xd_dfba, yd_dfba)

            # dAMN (using full 14 decoded sensor states)
            Xd_damn, yd_damn = delay_embed(damn_cycle_features, y_target, d)
            yp_damn, nmse_damn, mae_damn, r2_damn, _ = ridge_loo(Xd_damn, yd_damn)

            results.append(
                {
                    "coordinate": name,
                    "delay": d,
                    "phys_r2": r2_phys,
                    "phys_mae": mae_phys,
                    "phys_nmse": nmse_phys,
                    "dfba_r2": r2_dfba,
                    "dfba_mae": mae_dfba,
                    "dfba_nmse": nmse_dfba,
                    "damn_r2": r2_damn,
                    "damn_mae": mae_damn,
                    "damn_nmse": nmse_damn,
                }
            )

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "rossler_multimodal_metrics.csv", index=False)

    print("\n--- Best Decoding R² across Delay Embeddings ---")
    for coord in targets:
        sub = res_df[res_df["coordinate"] == coord]
        best_phys = sub.loc[sub["phys_r2"].idxmax()]
        best_dfba = sub.loc[sub["dfba_r2"].idxmax()]
        best_damn = sub.loc[sub["damn_r2"].idxmax()]
        print(f"\nTarget: {coord}")
        print(f"  Living Yeast : R2 = {best_phys['phys_r2']:.4f} (d={int(best_phys['delay'])}) | MAE = {best_phys['phys_mae']:.4f}")
        print(f"  Multimodal dFBA: R2 = {best_dfba['dfba_r2']:.4f} (d={int(best_dfba['delay'])}) | MAE = {best_dfba['dfba_mae']:.4f}")
        print(f"  Multimodal dAMN: R2 = {best_damn['damn_r2']:.4f} (d={int(best_damn['delay'])}) | MAE = {best_damn['damn_mae']:.4f}")

    # 5. Plotting 4-Coordinate Reconstructions
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    coords_list = list(targets.keys())

    for idx, ax in enumerate(axes.flat):
        coord = coords_list[idx]
        y_target = targets[coord]
        sub = res_df[res_df["coordinate"] == coord]

        best_phys = sub.loc[sub["phys_r2"].idxmax()]
        best_dfba = sub.loc[sub["dfba_r2"].idxmax()]
        best_damn = sub.loc[sub["damn_r2"].idxmax()]

        # Predictions at best delay
        d_p = int(best_phys["delay"])
        Xd_p, yd_p = delay_embed(real_cycle_features, y_target, d_p)
        yp_phys, _, _, _, _ = ridge_loo(Xd_p, yd_p)

        d_d = int(best_dfba["delay"])
        Xd_d, yd_d = delay_embed(dfba_cycle_od.reshape(-1, 1), y_target, d_d)
        yp_dfba, _, _, _, _ = ridge_loo(Xd_d, yd_d)

        d_m = int(best_damn["delay"])
        Xd_m, yd_m = delay_embed(damn_cycle_features, y_target, d_m)
        yp_damn, _, _, _, _ = ridge_loo(Xd_m, yd_m)

        ax.plot(timeline_hours, y_target, color="#1f2937", linewidth=2.4, label="True Chaotic Coordinate")
        ax.plot(timeline_hours[d_p:], yp_phys, color="#059669", linestyle="-.", marker="^", markersize=4, label=f"Living Yeast (d={d_p}, R²={best_phys['phys_r2']:.3f})")
        ax.plot(timeline_hours[d_d:], yp_dfba, color="#7c3aed", linestyle="--", marker="s", markersize=4, label=f"Multimodal dFBA (d={d_d}, R²={best_dfba['dfba_r2']:.3f})")
        ax.plot(timeline_hours[d_m:], yp_damn, color="#dc2626", linestyle=":", marker="o", markersize=4, label=f"Multimodal dAMN (d={d_m}, R²={best_damn['damn_r2']:.3f})")

        ax.set_title(f"4D Rössler Decoding: {coord}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Timeline (Hours)", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")

    plt.suptitle("4D Hyperchaotic Rössler Benchmark: Multimodal dFBA vs. Multimodal dAMN vs. Living Yeast", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "rossler_multimodal_reconstructions.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nBenchmark figure saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rossler-dir", type=Path, default=DEFAULT_ROSSLER_DIR)
    parser.add_argument("--damn-checkpoint", type=Path, default=DEFAULT_DAMN_CKPT)
    args = parser.parse_args()
    run_benchmark(args.rossler_dir, args.damn_checkpoint)
