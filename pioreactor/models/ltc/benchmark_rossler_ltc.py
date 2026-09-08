"""4D Rössler Hyperchaotic 4-Coordinate Benchmark for Multimodal LTC Digital Twin.

Evaluates zero-shot on the unseen 12.08-hour continuous 4D Rössler experiment (pioreactor/data/rossler):
Actuator Channels:
  1. Glucose chaotic dosing u_gluc (Media pump, mL)
  2. Salt osmotic chaotic dosing u_salt (Alt-media pump, mL)
  3. Temperature chaotic schedule u_temp (24°C - 35°C)
  4. UV irradiation chaotic schedule u_uv (LED PWM, 0% - 45%)

Evaluates:
  - 4-Coordinate dynamical reservoir reconstruction via Ridge regression with Leave-One-Out CV
  - Side-by-side comparison of Physical Yeast Culture vs. Multimodal LTC Digital Twin
  - 4-Panel visual dashboard saving to artifacts/benchmarks/ltc_rossler_benchmark.png
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

from dataset import NormalizationScalers
from ltc_model import LTCBioreactorTwin

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROSSLER_DIR = SCRIPT_DIR.parents[1] / "data" / "rossler"
CHECKPOINT_PATH = SCRIPT_DIR / "artifacts" / "model" / "ltc_multimodal_best_checkpoint.pt"
SCALERS_PATH = SCRIPT_DIR / "artifacts" / "dataset" / "scalers_multimodal.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "benchmarks"


def delay_embed(X: np.ndarray, y: np.ndarray, delay: int) -> tuple[np.ndarray, np.ndarray]:
    if delay == 0:
        return X.copy(), y.copy()
    rows = []
    for t in range(delay, len(X)):
        rows.append(np.concatenate([X[t - lag] for lag in range(delay + 1)]))
    return np.asarray(rows), y[delay:]


def ridge_loo(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rcv = RidgeCV(alphas=alphas, cv=None)
    rcv.fit(Xs, y)
    model = Ridge(alpha=rcv.alpha_)
    yp = cross_val_predict(model, Xs, y, cv=LeaveOneOut())
    nmse = mean_squared_error(y, yp) / (np.var(y) + 1e-8)
    r2 = r2_score(y, yp)
    mae = mean_absolute_error(y, yp)
    return yp, nmse, r2, mae, float(rcv.alpha_)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROSSLER_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--scalers", type=Path, default=SCALERS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    alphas = np.logspace(-4, 6, 100)

    # 1. Exact 29-cycle 4D Rössler inputs (25-min cycles = 12.08 hours)
    total_cycles = 29
    cycle_min = 25.0
    dt_min = 5.0
    steps_per_cycle = int(cycle_min / dt_min)  # 5 steps per cycle
    n_timesteps = total_cycles * steps_per_cycle

    # 4 Applied chaotic control signals
    u_gluc = np.array([
        0.29, 0.22, 0.28, 0.22, 0.23, 0.25, 0.17, 0.27, 0.19, 0.30,
        0.11, 0.34, 0.19, 0.27, 0.18, 0.21, 0.22, 0.13, 0.26, 0.14,
        0.30, 0.03, 0.35, 0.15, 0.23, 0.14, 0.11, 0.21, 0.04
    ], dtype=np.float32)

    u_salt = np.array([
        0.00, 0.01, 0.02, 0.03, 0.05, 0.04, 0.05, 0.02, 0.03, 0.04,
        0.06, 0.08, 0.03, 0.05, 0.07, 0.09, 0.08, 0.10, 0.05, 0.07,
        0.10, 0.12, 0.15, 0.09, 0.12, 0.14, 0.18, 0.13, 0.17
    ], dtype=np.float32)

    u_temp = np.array([
        29.7, 31.1, 28.6, 32.5, 27.0, 32.6, 27.4, 31.1, 29.3, 30.6,
        30.7, 28.1, 31.2, 28.3, 33.2, 26.0, 33.5, 26.2, 31.4, 29.2,
        30.7, 31.2, 27.1, 32.2, 27.1, 35.0, 24.0, 33.3, 27.1
    ], dtype=np.float32)

    u_uv = np.array([
        5.0, 1.3, 3.3, 1.3, 1.3, 2.6, 0.6, 20.0, 0.8, 12.8,
        0.3, 45.0, 0.9, 2.8, 0.7, 1.0, 1.3, 0.3, 4.8, 0.4,
        8.4, 0.0, 42.7, 0.5, 1.3, 0.4, 0.2, 1.3, 0.0
    ], dtype=np.float32)

    targets = {
        "Glucose (X)": {"y": u_gluc, "unit": "mL", "best_delay": 2, "color": "#e63946"},
        "Salt (W)": {"y": u_salt, "unit": "mL", "best_delay": 1, "color": "#2a9d8f"},
        "Temperature (Y)": {"y": u_temp, "unit": "°C", "best_delay": 1, "color": "#f4a261"},
        "UV Light (Z)": {"y": u_uv, "unit": "% PWM", "best_delay": 1, "color": "#9c27b0"},
    }

    # 2. Ingest Real Physical Yeast Culture Observations
    state_matrix_file = args.data_dir / "results" / "rossler_raw_state_matrix.csv"
    if not state_matrix_file.exists():
        state_matrix_file = SCRIPT_DIR.parents[1] / "data" / "rossler" / "results" / "rossler_raw_state_matrix.csv"

    df_raw = pd.read_csv(state_matrix_file)
    feature_cols = [c for c in df_raw.columns if c not in ["cycle", "norm_od"]]
    X_real = df_raw[feature_cols].to_numpy()
    real_norm_od = df_raw["norm_od"].to_numpy()

    # 3. Simulate Multimodal LTC Digital Twin on Exact 5-min Trajectory
    scalers = NormalizationScalers.load(args.scalers)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
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

    # Build 5-min input trajectory:
    # - Glucose and Salt pulse enter at cycle start (s_idx)
    # - Temperature and UV are maintained across all 5 steps of the cycle
    u_traj_raw = np.zeros((n_timesteps, 6), dtype=np.float32)
    for c in range(total_cycles):
        s_idx = c * steps_per_cycle
        u_traj_raw[s_idx, 0] = u_gluc[c]
        u_traj_raw[s_idx, 1] = u_salt[c]
        u_traj_raw[s_idx, 2] = u_gluc[c] + u_salt[c]
        u_traj_raw[s_idx, 3] = u_gluc[c] + u_salt[c]
        for step in range(steps_per_cycle):
            u_traj_raw[s_idx + step, 4] = u_temp[c]
            u_traj_raw[s_idx + step, 5] = u_uv[c]

    u_norm = np.nan_to_num(scalers.normalize_inputs(u_traj_raw), nan=0.0)
    u_tensor = torch.from_numpy(u_norm).unsqueeze(0)

    init_od_val = float(real_norm_od[0]) if len(real_norm_od) > 0 else 0.5
    with torch.no_grad():
        out_ltc = model(
            u_seq=u_tensor,
            init_od=torch.tensor([[init_od_val]], dtype=torch.float32),
            init_vol=torch.tensor([[13.5]], dtype=torch.float32),
        )

    hidden_all = out_ltc["hidden"].squeeze(0).numpy()
    y_pred_all = scalers.denormalize_sensors(out_ltc["y_pred"].squeeze(0).numpy())

    # Cycle-average LTC state representations
    ltc_cycle_features = []
    for c in range(total_cycles):
        start = c * steps_per_cycle
        end = (c + 1) * steps_per_cycle
        cycle_h = np.mean(hidden_all[start:end], axis=0)
        cycle_y = np.mean(y_pred_all[start:end], axis=0)
        ltc_cycle_features.append(np.concatenate([cycle_h, cycle_y]))

    X_ltc = np.array(ltc_cycle_features)

    # 4. Decode ALL 4 Coordinates on Physical Yeast vs. Multimodal LTC
    metrics_records = []
    plot_data = {}

    print(f"\n--- 4D Rössler 4-Coordinate Reservoir Decoding (LOOCV) ---")
    timeline_h = np.arange(total_cycles) * (cycle_min / 60.0)

    for coord_name, info in targets.items():
        y_val = info["y"]

        # Sweep delays d = 0..3 for both
        best_real_r2, best_real_nmse, best_real_mae, best_real_d, best_real_yp = -float("inf"), 1.0, 1.0, 0, None
        best_ltc_r2, best_ltc_nmse, best_ltc_mae, best_ltc_d, best_ltc_yp = -float("inf"), 1.0, 1.0, 0, None

        for d in range(4):
            # Physical yeast
            Xd_r, yd_r = delay_embed(X_real, y_val, d)
            yp_r, nmse_r, r2_r, mae_r, _ = ridge_loo(Xd_r, yd_r, alphas)
            if r2_r > best_real_r2:
                best_real_r2, best_real_nmse, best_real_mae, best_real_d, best_real_yp = r2_r, nmse_r, mae_r, d, yp_r

            # LTC digital twin
            Xd_l, yd_l = delay_embed(X_ltc, y_val, d)
            yp_l, nmse_l, r2_l, mae_l, _ = ridge_loo(Xd_l, yd_l, alphas)
            if r2_l > best_ltc_r2:
                best_ltc_r2, best_ltc_nmse, best_ltc_mae, best_ltc_d, best_ltc_yp = r2_l, nmse_l, mae_l, d, yp_l

        print(f"\nTarget: {coord_name}")
        print(f"  Physical Yeast : R² = {best_real_r2:.4f} (d={best_real_d}, NMSE={best_real_nmse:.4f}, MAE={best_real_mae:.4f})")
        print(f"  Multimodal LTC : R² = {best_ltc_r2:.4f} (d={best_ltc_d}, NMSE={best_ltc_nmse:.4f}, MAE={best_ltc_mae:.4f})")

        metrics_records.append({
            "coordinate": coord_name,
            "real_r2": best_real_r2,
            "real_nmse": best_real_nmse,
            "real_mae": best_real_mae,
            "real_delay": best_real_d,
            "ltc_r2": best_ltc_r2,
            "ltc_nmse": best_ltc_nmse,
            "ltc_mae": best_ltc_mae,
            "ltc_delay": best_ltc_d,
        })

        plot_data[coord_name] = {
            "y_true": y_val,
            "real_yp": best_real_yp,
            "real_d": best_real_d,
            "real_r2": best_real_r2,
            "ltc_yp": best_ltc_yp,
            "ltc_d": best_ltc_d,
            "ltc_r2": best_ltc_r2,
            "unit": info["unit"],
            "color": info["color"],
        }

    # 5. Plot 4-Panel Reconstruction Dashboard (All 4 Coordinates)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (coord_name, p) in enumerate(plot_data.items()):
        ax = axes[idx]
        y_true = p["y_true"]
        real_yp = p["real_yp"]
        ltc_yp = p["ltc_yp"]
        d_r = p["real_d"]
        d_l = p["ltc_d"]

        t_r = timeline_h[d_r:]
        t_l = timeline_h[d_l:]

        ax.plot(timeline_h, y_true, color="#1d3557", lw=2.4, label=f"True {coord_name}")
        ax.plot(t_r, real_yp, color="#457b9d", lw=1.8, ls="-.", marker="^", ms=4, label=f"Living Yeast (d={d_r}, R²={p['real_r2']:.4f})")
        ax.plot(t_l, ltc_yp, color=p["color"], lw=2.0, ls="--", marker="o", ms=4, label=f"LTC Twin (d={d_l}, R²={p['ltc_r2']:.4f})")

        ax.set_title(f"{coord_name} Chaotic Reconstruction", fontweight="bold", fontsize=12)
        ax.set_xlabel("Timeline [Hours]", fontweight="bold")
        ax.set_ylabel(f"Amplitude [{p['unit']}]", fontweight="bold")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)

    plt.suptitle("4D Rössler Hyperchaotic 4-Coordinate Benchmark: Living Yeast Culture vs. Multimodal LTC Digital Twin", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = args.output_dir / "ltc_rossler_benchmark.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved 4-coordinate Rössler benchmark plot to: {out_file}")

    df_metrics = pd.DataFrame(metrics_records)
    metrics_path = args.output_dir / "rossler_ltc_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Saved 4-coordinate Rössler metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
