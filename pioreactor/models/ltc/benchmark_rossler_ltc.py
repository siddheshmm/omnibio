"""4D Rössler Hyperchaotic Dual-Chemical Benchmark for LTC Digital Twin.

Evaluates on the unseen 12.08-hour Rössler experiment (pioreactor/data/rossler):
- Ingests raw physical state matrix (real culture multi-sensor features)
- Dual inputs: Glucose pulses (chaotic driver x) + Salt pulses (chaotic driver w)
- Simulates continuous LTC digital twin on exact 29-cycle trajectory
- Decodes both chemical driving waveforms via Ridge regression with Leave-One-Out CV
- Compares physical yeast reservoir decoding against the LTC digital twin
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
CHECKPOINT_PATH = SCRIPT_DIR / "artifacts" / "model" / "ltc_best_checkpoint.pt"
SCALERS_PATH = SCRIPT_DIR / "artifacts" / "dataset" / "scalers.json"
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

    # 2. Ingest Real Physical Yeast Culture Observations
    state_matrix_file = args.data_dir / "results" / "rossler_raw_state_matrix.csv"
    if not state_matrix_file.exists():
        state_matrix_file = SCRIPT_DIR.parents[1] / "data" / "rossler" / "results" / "rossler_raw_state_matrix.csv"

    df_raw = pd.read_csv(state_matrix_file)
    feature_cols = [c for c in df_raw.columns if c not in ["cycle", "norm_od"]]
    X_real = df_raw[feature_cols].to_numpy()
    real_norm_od = df_raw["norm_od"].to_numpy()

    # Real Yeast decodings
    Xd_rg, yd_rg = delay_embed(X_real, u_gluc, 2)
    yp_real_gluc, nmse_rg, r2_rg, mae_rg, _ = ridge_loo(Xd_rg, yd_rg, alphas)

    Xd_rs, yd_rs = delay_embed(X_real, u_salt, 0)
    yp_real_salt, nmse_rs, r2_rs, mae_rs, _ = ridge_loo(Xd_rs, yd_rs, alphas)

    print(f"Loaded 4D Rössler Experiment: {total_cycles} cycles ({total_cycles * cycle_min / 60:.2f} h)")
    print(f"  Physical Yeast Glucose Decoding (d=2): R² = {r2_rg:.4f}, NMSE = {nmse_rg:.4f}")
    print(f"  Physical Yeast Salt Decoding    (d=0): R² = {r2_rs:.4f}, NMSE = {nmse_rs:.4f}")

    # 3. Simulate LTC Digital Twin on Exact 5-minute binned Trajectory
    # 25-min cycle = 5 steps of dt=5 min each
    steps_per_cycle = int(cycle_min / 5.0)  # 5 steps
    n_timesteps = total_cycles * steps_per_cycle

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

    # Build 5-min input trajectory: pulse applied at cycle start
    u_traj_raw = np.zeros((n_timesteps, 6), dtype=np.float32)
    u_traj_raw[:, 4] = 30.0  # baseline temp
    u_traj_raw[:, 5] = 0.0   # baseline uv

    for c in range(total_cycles):
        step_idx = c * steps_per_cycle
        u_traj_raw[step_idx, 0] = u_gluc[c]
        u_traj_raw[step_idx, 1] = u_salt[c]
        u_traj_raw[step_idx, 2] = u_gluc[c] + u_salt[c]  # chemostat waste removal
        u_traj_raw[step_idx, 3] = u_gluc[c] + u_salt[c]

    u_norm = scalers.normalize_inputs(u_traj_raw)
    u_tensor = torch.from_numpy(u_norm).unsqueeze(0)  # (1, T, 6)

    init_od_val = float(real_norm_od[0]) if len(real_norm_od) > 0 else 0.5
    with torch.no_grad():
        out_ltc = model(
            u_seq=u_tensor,
            init_od=torch.tensor([[init_od_val]], dtype=torch.float32),
            init_vol=torch.tensor([[13.5]], dtype=torch.float32),
        )

    hidden_all = out_ltc["hidden"].squeeze(0).numpy()
    y_pred_all = scalers.denormalize_sensors(out_ltc["y_pred"].squeeze(0).numpy())

    # Cycle-average LTC state representations to match experimental 25-min cycles
    ltc_cycle_features = []
    pred_od_cycles = []
    for c in range(total_cycles):
        start = c * steps_per_cycle
        end = (c + 1) * steps_per_cycle
        cycle_h = np.mean(hidden_all[start:end], axis=0)
        cycle_y = np.mean(y_pred_all[start:end], axis=0)
        pred_od_cycles.append(cycle_y[0])  # norm_od is index 0
        ltc_cycle_features.append(np.concatenate([cycle_h, cycle_y]))

    X_ltc = np.array(ltc_cycle_features)
    pred_norm_od = np.array(pred_od_cycles)

    # 4. Decode Dual Chemicals on LTC Twin
    # Glucose decoding on LTC (d=2)
    Xd_lg, yd_lg = delay_embed(X_ltc, u_gluc, 2)
    yp_ltc_gluc, nmse_lg, r2_lg, mae_lg, _ = ridge_loo(Xd_lg, yd_lg, alphas)

    # Salt decoding on LTC (d=1)
    Xd_ls, yd_ls = delay_embed(X_ltc, u_salt, 1)
    yp_ltc_salt, nmse_ls, r2_ls, mae_ls, _ = ridge_loo(Xd_ls, yd_ls, alphas)

    print(f"\nLTC Digital Twin Decodings:")
    print(f"  LTC Glucose Decoding (d=2): R² = {r2_lg:.4f}, NMSE = {nmse_lg:.4f}")
    print(f"  LTC Salt Decoding    (d=1): R² = {r2_ls:.4f}, NMSE = {nmse_ls:.4f}")

    # 5. Plot Comparative Benchmark Dashboard (2 Panels)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    timeline_h = np.arange(total_cycles) * (cycle_min / 60.0)

    # Left: Trajectory Growth (OD)
    axes[0].plot(timeline_h, real_norm_od, color="#1f77b4", lw=2.4, label="Physical Yeast (Observed norm_od)")
    axes[0].plot(timeline_h, pred_norm_od, color="#e63946", lw=2.0, ls="--", label="LTC Digital Twin Simulation")
    axes[0].set_title("4D Rössler (12.1h): Physical Growth vs. LTC Trajectory", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Timeline [Hours]", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Normalized OD", fontweight="bold", fontsize=11)
    axes[0].legend(loc="best", frameon=True)
    axes[0].grid(True, alpha=0.3)

    # Right: Dual-Chemical Reconstruction Overlay
    t_d2 = timeline_h[2:]
    t_d1 = timeline_h[1:]

    axes[1].plot(timeline_h, u_gluc, color="#1d3557", lw=2.4, label="Actual Glucose Input")
    axes[1].plot(t_d2, yp_ltc_gluc, color="#e63946", lw=1.8, ls="--", marker="o", ms=4, label=f"LTC Glucose (d=2, R²={r2_lg:.4f})")
    axes[1].plot(timeline_h, u_salt, color="#457b9d", lw=2.4, label="Actual Salt Input")
    axes[1].plot(t_d1, yp_ltc_salt, color="#2a9d8f", lw=1.8, ls="-.", marker="s", ms=4, label=f"LTC Salt (d=1, R²={r2_ls:.4f})")
    axes[1].set_title(f"4D Rössler Dual-Chemical Decoding with LTC (Glucose R²={r2_lg:.4f} | Salt R²={r2_ls:.4f})", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Timeline [Hours]", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Dosing Volume [mL]", fontweight="bold", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Unseen 12.1h 4D Rössler Hyperchaotic Benchmark: Physical Yeast Culture vs. LTC Digital Twin", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = args.output_dir / "ltc_rossler_benchmark.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Rössler benchmark plot to: {out_file}")

    # Save metrics CSV
    df_metrics = pd.DataFrame([
        {"target": "glucose", "model": "Physical Yeast Culture", "delay": 2, "r2": r2_rg, "nmse": nmse_rg, "mae": mae_rg},
        {"target": "glucose", "model": "LTC Digital Twin", "delay": 2, "r2": r2_lg, "nmse": nmse_lg, "mae": mae_lg},
        {"target": "salt", "model": "Physical Yeast Culture", "delay": 0, "r2": r2_rs, "nmse": nmse_rs, "mae": mae_rs},
        {"target": "salt", "model": "LTC Digital Twin", "delay": 1, "r2": r2_ls, "nmse": nmse_ls, "mae": mae_ls},
    ])
    df_metrics.to_csv(args.output_dir / "rossler_ltc_metrics.csv", index=False)
    print(f"Saved Rössler metrics to: {args.output_dir / 'rossler_ltc_metrics.csv'}")


if __name__ == "__main__":
    main()
