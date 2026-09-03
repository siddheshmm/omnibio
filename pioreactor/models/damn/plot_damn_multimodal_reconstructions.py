"""Generate multimodal chaotic trajectory reconstructions for the pure dAMN digital twin.

Matches the exact styling, delay-embedding optimization, and presentation of:
pioreactor/data/rossler/results/multimodal_reconstructions.png
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from damn_ode import DAMN

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROSSLER_DIR = SCRIPT_DIR / ".." / ".." / "data" / "rossler"
CHECKPOINT_PATH = SCRIPT_DIR / "artifacts" / "model" / "damn_checkpoint.pt"
OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "benchmarks"


def delay_embed(X: np.ndarray, y: np.ndarray, delay: int):
    if delay == 0:
        return X.copy(), y.copy()
    rows = []
    for t in range(delay, len(X)):
        rows.append(np.concatenate([X[t - lag] for lag in range(delay + 1)]))
    return np.asarray(rows), y[delay:]


def ridge_loo_reg(X: np.ndarray, y: np.ndarray, alphas: np.ndarray):
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TOTAL_CYCLES = 29
    CYCLE_MIN = 25.0
    dt_min = 5.0
    steps_per_cycle = int(CYCLE_MIN / dt_min)
    n_timesteps = TOTAL_CYCLES * steps_per_cycle

    # 1. Real Applied Rössler Targets
    u_uv = np.array([5.0, 1.3, 3.3, 1.3, 1.3, 2.6, 0.6, 20.0, 0.8, 12.8, 0.3, 45.0, 0.9, 2.8, 0.7, 1.0, 1.3, 0.3, 4.8, 0.4, 8.4, 0.0, 42.7, 0.5, 1.3, 0.4, 0.2, 1.3, 0.0], dtype=np.float32)
    u_temp = np.array([29.7, 31.1, 28.6, 32.5, 27.0, 32.6, 27.4, 31.1, 29.3, 30.6, 30.7, 28.1, 31.2, 28.3, 33.2, 26.0, 33.5, 26.2, 31.4, 29.2, 30.7, 31.2, 27.1, 32.2, 27.1, 35.0, 24.0, 33.3, 27.1], dtype=np.float32)
    u_gluc = np.array([0.29, 0.22, 0.28, 0.22, 0.23, 0.25, 0.17, 0.27, 0.19, 0.30, 0.11, 0.34, 0.19, 0.27, 0.18, 0.21, 0.22, 0.13, 0.26, 0.14, 0.30, 0.03, 0.35, 0.15, 0.23, 0.14, 0.11, 0.21, 0.04], dtype=np.float32)
    u_salt = np.array([0.00, 0.01, 0.02, 0.03, 0.05, 0.04, 0.05, 0.02, 0.03, 0.04, 0.06, 0.08, 0.03, 0.05, 0.07, 0.09, 0.08, 0.10, 0.05, 0.07, 0.10, 0.12, 0.15, 0.09, 0.12, 0.14, 0.18, 0.13, 0.17], dtype=np.float32)

    # 2. Simulate Pure Continuous dAMN Model on Real Applied Dosing
    u_trajectory = np.zeros((1, n_timesteps, 4), dtype=np.float32)
    for c in range(TOTAL_CYCLES):
        step_idx = c * steps_per_cycle
        u_trajectory[0, step_idx, 0] = u_gluc[c]
        u_trajectory[0, step_idx, 1] = u_salt[c]
        u_trajectory[0, step_idx, 2] = u_gluc[c] + u_salt[c]
        u_trajectory[0, step_idx, 3] = u_gluc[c] + u_salt[c]

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = DAMN(dt_min=dt_min)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    init_biomass = torch.tensor([[0.225]], dtype=torch.float32)
    init_volume = torch.tensor([[13.5]], dtype=torch.float32)

    with torch.no_grad():
        out = model(torch.from_numpy(u_trajectory), init_biomass, init_volume)

    damn_y = out["y_pred"][0].numpy()  # (145, 14)
    biomass = out["biomass"][0].numpy()
    mu = out["growth_rate"][0].numpy()
    fluxes = out["fluxes"][0].numpy()
    damn_all = np.concatenate([biomass, mu, fluxes, damn_y], axis=-1)

    # Aggregate into cycle-mean state matrix
    damn_cycles = []
    for c in range(TOTAL_CYCLES):
        start = c * steps_per_cycle
        end = (c + 1) * steps_per_cycle
        damn_cycles.append(np.mean(damn_all[start:end], axis=0))
    X_damn = np.array(damn_cycles)  # (29, 20)

    # Separate feature sets: bio features for glucose, full features for salt
    X_damn_bio = X_damn[:, :2]   # biomass & growth rate
    X_damn_sensors = X_damn[:, 6:]  # 14 sensor channels

    # 3. Optimize Delay Embeddings for dAMN
    ALPHAS = np.logspace(-4, 6, 100)
    MAX_DELAY = 5

    # Run delay sweeps
    def find_best(X, y_target):
        best_r2 = -999.0
        best_pack = None
        for d in range(MAX_DELAY + 1):
            Xd, yd = delay_embed(X, y_target, d)
            yp, nmse, mae, r2, alpha = ridge_loo_reg(Xd, yd, ALPHAS)
            if r2 > best_r2:
                best_r2 = r2
                best_pack = (d, yp, yd, nmse, mae, r2)
        return best_pack

    best_gluc_damn = find_best(X_damn_bio, u_gluc)
    best_salt_damn = find_best(X_damn_sensors, u_salt)

    # Also load the Real Yeast best packs for reference / comparison
    rossler_raw_csv = DEFAULT_ROSSLER_DIR / "results" / "rossler_raw_state_matrix.csv"
    if not rossler_raw_csv.exists():
        rossler_raw_csv = SCRIPT_DIR.parents[1] / "data" / "rossler" / "results" / "rossler_raw_state_matrix.csv"
    df_real = pd.read_csv(rossler_raw_csv)
    X_real = df_real[[c for c in df_real.columns if c != "cycle"]].to_numpy(dtype=np.float32)

    best_gluc_real = find_best(X_real, u_gluc)
    best_salt_real = find_best(X_real, u_salt)
    best_uv_real = find_best(X_real, u_uv)
    best_temp_real = find_best(X_real, u_temp)

    print("=== Pure dAMN Optimized Multimodal Reconstructions ===")
    print(f"Glucose (mL) | d={best_gluc_damn[0]} | R² = {best_gluc_damn[5]:.4f} | NMSE = {best_gluc_damn[3]:.4f} | MAE = {best_gluc_damn[4]:.4f}")
    print(f"Salt (mL)    | d={best_salt_damn[0]} | R² = {best_salt_damn[5]:.4f} | NMSE = {best_salt_damn[3]:.4f} | MAE = {best_salt_damn[4]:.4f}")

    print("\n=== Real Pioreactor Yeast Hardware Reference ===")
    print(f"Glucose (mL) | d={best_gluc_real[0]} | R² = {best_gluc_real[5]:.4f} | NMSE = {best_gluc_real[3]:.4f} | MAE = {best_gluc_real[4]:.4f}")
    print(f"Salt (mL)    | d={best_salt_real[0]} | R² = {best_salt_real[5]:.4f} | NMSE = {best_salt_real[3]:.4f} | MAE = {best_salt_real[4]:.4f}")

    # 4. Generate the Exact 4-Panel Plot Matching multimodal_reconstructions.png
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)

    # --- Panel 1: Glucose Reconstruction on Pure dAMN ---
    d_g, yp_g, yd_g, nmse_g, mae_g, r2_g = best_gluc_damn
    t_g = np.arange(d_g, TOTAL_CYCLES) * (CYCLE_MIN / 60.0)
    axes[0].plot(t_g, yd_g, color="#1d3557", lw=2.2, label="Actual Glucose (mL)")
    axes[0].plot(t_g, yp_g, color="#e63946", lw=1.8, ls="--", marker="o", ms=4, label=f"dAMN Reconstructed (d={d_g}, R²={r2_g:.4f})")
    axes[0].set_title(f"Pure dAMN Twin Reconstruction: Glucose (mL) (MAE = {mae_g:.4f})", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Glucose (mL)", fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Salt Reconstruction on Pure dAMN ---
    d_s, yp_s, yd_s, nmse_s, mae_s, r2_s = best_salt_damn
    t_s = np.arange(d_s, TOTAL_CYCLES) * (CYCLE_MIN / 60.0)
    axes[1].plot(t_s, yd_s, color="#1d3557", lw=2.2, label="Actual Salt (mL)")
    axes[1].plot(t_s, yp_s, color="#e63946", lw=1.8, ls="--", marker="o", ms=4, label=f"dAMN Reconstructed (d={d_s}, R²={r2_s:.4f})")
    axes[1].set_title(f"Pure dAMN Twin Reconstruction: Salt (mL) (MAE = {mae_s:.4f})", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Salt (mL)", fontweight="bold")
    axes[1].legend(loc="upper left", frameon=True)
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Comparison Overlay: Actual vs. Real Yeast vs. dAMN (Glucose) ---
    axes[2].plot(np.arange(TOTAL_CYCLES) * (CYCLE_MIN / 60.0), u_gluc, color="#1d3557", lw=2.4, label="Target Chaotic Glucose Command")
    d_gr, yp_gr, _, _, _, r2_gr = best_gluc_real
    t_gr = np.arange(d_gr, TOTAL_CYCLES) * (CYCLE_MIN / 60.0)
    axes[2].plot(t_gr, yp_gr, color="#2a9d8f", lw=1.8, ls="-.", marker="s", ms=4, label=f"Real Yeast Telemetry (d={d_gr}, R²={r2_gr:.4f})")
    axes[2].plot(t_g, yp_g, color="#e63946", lw=1.8, ls="--", marker="o", ms=4, label=f"Pure dAMN Twin (d={d_g}, R²={r2_g:.4f})")
    axes[2].set_title(f"Comparison: Glucose Decoding (Real Yeast R²={r2_gr:.4f} vs. dAMN R²={r2_g:.4f})", fontweight="bold", fontsize=12)
    axes[2].set_ylabel("Glucose (mL)", fontweight="bold")
    axes[2].legend(loc="upper right", frameon=True)
    axes[2].grid(True, alpha=0.3)

    # --- Panel 4: Comparison Overlay: Actual vs. Real Yeast vs. dAMN (Salt) ---
    axes[3].plot(np.arange(TOTAL_CYCLES) * (CYCLE_MIN / 60.0), u_salt, color="#1d3557", lw=2.4, label="Target Chaotic Salt Command")
    d_sr, yp_sr, _, _, _, r2_sr = best_salt_real
    t_sr = np.arange(d_sr, TOTAL_CYCLES) * (CYCLE_MIN / 60.0)
    axes[3].plot(t_sr, yp_sr, color="#2a9d8f", lw=1.8, ls="-.", marker="s", ms=4, label=f"Real Yeast Telemetry (d={d_sr}, R²={r2_sr:.4f})")
    axes[3].plot(t_s, yp_s, color="#e63946", lw=1.8, ls="--", marker="o", ms=4, label=f"Pure dAMN Twin (d={d_s}, R²={r2_s:.4f})")
    axes[3].set_title(f"Comparison: Salt Decoding (Real Yeast R²={r2_sr:.4f} vs. dAMN R²={r2_s:.4f})", fontweight="bold", fontsize=12)
    axes[3].set_xlabel("Timeline [Hours] (29 steps x 25.0 min = 12.08h)", fontweight="bold", fontsize=11)
    axes[3].set_ylabel("Salt (mL)", fontweight="bold")
    axes[3].legend(loc="upper left", frameon=True)
    axes[3].grid(True, alpha=0.3)

    plt.suptitle("4D Rössler Hyperchaotic Trajectory Reconstructions: Pure dAMN vs. Real Pioreactor", fontsize=14, fontweight="bold", y=0.992)
    plt.tight_layout()

    out_plot = OUTPUT_DIR / "damn_multimodal_reconstructions.png"
    plt.savefig(out_plot, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved dAMN multimodal reconstruction plot to: {out_plot}")


if __name__ == "__main__":
    main()
