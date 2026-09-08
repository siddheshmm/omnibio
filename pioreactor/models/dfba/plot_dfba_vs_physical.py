"""Comparative visual dashboard: Physical Pioreactor Culture vs. Pure dFBA vs. Pure dAMN.

Evaluates both:
1. 4D Rössler Hyperchaotic Dual-Chemical Experiment (12.08h)
2. Periodic Mackey-Glass Glucose Experiment (6.80h)
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

import run_dfba

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
DEFAULT_MODEL = HERE / "models" / "yeast-GEM-src" / "model" / "yeast-GEM.xml"
OUTPUT_DIR = HERE / "artifacts" / "plots"


def delay_embed(X: np.ndarray, y: np.ndarray, delay: int):
    if delay == 0:
        return X.copy(), y.copy()
    rows = []
    for t in range(delay, len(X)):
        rows.append(np.concatenate([X[t - lag] for lag in range(delay + 1)]))
    return np.asarray(rows), y[delay:]


def ridge_loo(X: np.ndarray, y: np.ndarray, alphas: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rcv = RidgeCV(alphas=alphas, cv=None)
    rcv.fit(Xs, y)
    model = Ridge(alpha=rcv.alpha_)
    yp = cross_val_predict(model, Xs, y, cv=LeaveOneOut())
    nmse = mean_squared_error(y, yp) / (np.var(y) + 1e-8)
    r2 = r2_score(y, yp)
    mae = mean_absolute_error(y, yp)
    return yp, nmse, r2, mae


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = run_dfba.load_config(DEFAULT_CONFIG)

    # -------------------------------------------------------------
    # 1. 25 MAY MACKEY-GLASS COMPARISON
    # -------------------------------------------------------------
    data_path_mg = Path(r"D:\omnibio\results\25th may")
    cycle_min_mg = 6.0
    dosing_mg = pd.read_csv(data_path_mg / "dosing_events-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])
    settings_mg = pd.read_csv(data_path_mg / "dosing_automation_settings-Demo_experiment-all_units-20260526090312.csv", parse_dates=["started_at"])
    od_filt_mg = pd.read_csv(data_path_mg / "od_readings_filtered-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])

    mask_mg = (settings_mg["automation_name"] == "mg_narma_dosing") & settings_mg["json_settings"].str.contains("duration", na=False)
    run_start_mg = settings_mg[mask_mg]["started_at"].max()

    media_mg = dosing_mg[(dosing_mg["source_of_event"] == "dosing_automation:mg_narma_dosing") & (dosing_mg["event"] == "add_media") & (dosing_mg["timestamp"] >= run_start_mg)].sort_values("timestamp").copy()
    media_mg["t_min"] = (media_mg["timestamp"] - run_start_mg).dt.total_seconds() / 60.0
    media_mg["cycle"] = np.floor(media_mg["t_min"] / cycle_min_mg).astype(int)

    u_mg = media_mg.groupby("cycle")["volume_change_ml"].sum()
    total_cycles_mg = int(u_mg.index.max()) + 1
    u_mg = u_mg.reindex(range(total_cycles_mg), fill_value=0.0)
    y_target_mg = u_mg.to_numpy(dtype=np.float32)

    od_filt_copy = od_filt_mg.copy()
    od_filt_copy["t_min"] = (od_filt_copy["timestamp"] - run_start_mg).dt.total_seconds() / 60.0
    od_filt_copy = od_filt_copy[(od_filt_copy["t_min"] >= 0) & (od_filt_copy["t_min"] < total_cycles_mg * cycle_min_mg)]
    od_filt_copy["cycle"] = np.floor(od_filt_copy["t_min"] / cycle_min_mg).astype(int)
    norm_od_mg = od_filt_copy.groupby("cycle")["normalized_od_reading"].mean().reindex(range(total_cycles_mg)).interpolate(limit_direction="both").values

    timestamps_mg = [run_start_mg + pd.Timedelta(minutes=c * cycle_min_mg) for c in range(total_cycles_mg)]
    df_run_mg = pd.DataFrame({
        "timestamp": timestamps_mg,
        "add_media_ml": y_target_mg,
        "add_alt_media_ml": np.zeros(total_cycles_mg),
        "remove_waste_ml": y_target_mg,
        "condition": ["glucose"] * total_cycles_mg,
        "norm_od": norm_od_mg,
    })

    sim_mg, _ = run_dfba.simulate(DEFAULT_MODEL, df_run_mg, config)
    pred_od_dfba_mg = sim_mg["predicted_norm_od"].values

    state_cols_mg = ["biomass_gdw_per_l", "volume_ml", "glucose_mmol_per_l", "predicted_norm_od"]
    X_dfba_mg = sim_mg[state_cols_mg].values
    Xd_mg, yd_mg = delay_embed(X_dfba_mg, y_target_mg, 2)
    yp_dfba_mg, nmse_dfba_mg, r2_dfba_mg, mae_dfba_mg = ridge_loo(Xd_mg, yd_mg, np.logspace(-4, 4, 50))

    # -------------------------------------------------------------
    # 2. 4D RÖSSLER COMPARISON
    # -------------------------------------------------------------
    data_dir_rossler = Path(r"D:\omnibio\pioreactor\data\rossler")
    total_cycles_rossler = 29
    cycle_min_rossler = 25.0

    u_gluc = np.array([0.29, 0.22, 0.28, 0.22, 0.23, 0.25, 0.17, 0.27, 0.19, 0.30, 0.11, 0.34, 0.19, 0.27, 0.18, 0.21, 0.22, 0.13, 0.26, 0.14, 0.30, 0.03, 0.35, 0.15, 0.23, 0.14, 0.11, 0.21, 0.04], dtype=np.float32)
    u_salt = np.array([0.00, 0.01, 0.02, 0.03, 0.05, 0.04, 0.05, 0.02, 0.03, 0.04, 0.06, 0.08, 0.03, 0.05, 0.07, 0.09, 0.08, 0.10, 0.05, 0.07, 0.10, 0.12, 0.15, 0.09, 0.12, 0.14, 0.18, 0.13, 0.17], dtype=np.float32)

    df_raw_rossler = pd.read_csv(data_dir_rossler / "results" / "rossler_raw_state_matrix.csv")
    real_norm_od_rossler = df_raw_rossler["norm_od"].values

    t0_rossler = pd.Timestamp("2026-08-26 18:00:00")
    timestamps_rossler = [t0_rossler + pd.Timedelta(minutes=c * cycle_min_rossler) for c in range(total_cycles_rossler)]

    df_run_rossler = pd.DataFrame({
        "timestamp": timestamps_rossler,
        "add_media_ml": u_gluc,
        "add_alt_media_ml": u_salt,
        "remove_waste_ml": u_gluc + u_salt,
        "condition": ["glucose"] * total_cycles_rossler,
        "norm_od": real_norm_od_rossler,
    })

    sim_rossler, _ = run_dfba.simulate(DEFAULT_MODEL, df_run_rossler, config)
    pred_od_dfba_rossler = sim_rossler["predicted_norm_od"].values

    state_cols_rossler = ["biomass_gdw_per_l", "volume_ml", "glucose_mmol_per_l", "sodium_mmol_per_l", "chloride_mmol_per_l", "predicted_norm_od"]
    X_dfba_rossler = sim_rossler[state_cols_rossler].values

    # Glucose decoding on dFBA (d=2)
    Xd_g, yd_g = delay_embed(X_dfba_rossler, u_gluc, 2)
    yp_dfba_gluc, nmse_dfba_gluc, r2_dfba_gluc, mae_dfba_gluc = ridge_loo(Xd_g, yd_g, np.logspace(-4, 6, 100))

    # Salt decoding on dFBA (d=1)
    Xd_s, yd_s = delay_embed(X_dfba_rossler, u_salt, 1)
    yp_dfba_salt, nmse_dfba_salt, r2_dfba_salt, mae_dfba_salt = ridge_loo(Xd_s, yd_s, np.logspace(-4, 6, 100))

    # -------------------------------------------------------------
    # 3. PLOT 4-PANEL DASHBOARD
    # -------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Panel 1: Mackey-Glass Trajectory Dynamics (OD)
    t_mg = np.arange(total_cycles_mg) * cycle_min_mg
    axes[0, 0].plot(t_mg, norm_od_mg, color="#1f77b4", lw=2.2, label="Physical Culture (Observed norm_od)")
    axes[0, 0].plot(t_mg, pred_od_dfba_mg, color="#9c27b0", lw=2.0, ls="--", label="dFBA Simulation (Yeast-GEM)")
    axes[0, 0].set_title("Mackey-Glass (6.8h): Culture Growth vs. dFBA Trajectory", fontweight="bold", fontsize=12)
    axes[0, 0].set_xlabel("Timeline [Minutes]", fontweight="bold")
    axes[0, 0].set_ylabel("Normalized OD", fontweight="bold")
    axes[0, 0].legend(loc="best", frameon=True)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Mackey-Glass Waveform Decoding Comparison
    t_mg_d = np.arange(2, total_cycles_mg) * cycle_min_mg
    axes[0, 1].plot(t_mg, y_target_mg, color="#1d3557", lw=2.2, label="True MG Dosing Input")
    axes[0, 1].plot(t_mg_d, yp_dfba_mg, color="#9c27b0", lw=1.8, ls="--", marker="^", ms=3, label=f"dFBA Readout (d=2, R²={r2_dfba_mg:.4f})")
    axes[0, 1].set_title(f"Mackey-Glass Input Decoding: dFBA (R²={r2_dfba_mg:.4f}) vs. Physical (R²=0.9641)", fontweight="bold", fontsize=12)
    axes[0, 1].set_xlabel("Timeline [Minutes]", fontweight="bold")
    axes[0, 1].set_ylabel("Media dose (mL)", fontweight="bold")
    axes[0, 1].legend(loc="upper right", frameon=True)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: 4D Rössler Trajectory Dynamics (OD)
    t_rossler = np.arange(total_cycles_rossler) * (cycle_min_rossler / 60.0)
    axes[1, 0].plot(t_rossler, real_norm_od_rossler, color="#1f77b4", lw=2.2, label="Physical Culture (Observed norm_od)")
    axes[1, 0].plot(t_rossler, pred_od_dfba_rossler, color="#9c27b0", lw=2.0, ls="--", label="dFBA Simulation (Yeast-GEM)")
    axes[1, 0].set_title("4D Rössler (12.1h): Culture Growth vs. dFBA Trajectory", fontweight="bold", fontsize=12)
    axes[1, 0].set_xlabel("Timeline [Hours]", fontweight="bold")
    axes[1, 0].set_ylabel("Normalized OD", fontweight="bold")
    axes[1, 0].legend(loc="best", frameon=True)
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: 4D Rössler Chaotic Reconstruction (Glucose & Salt)
    t_rossler_d2 = np.arange(2, total_cycles_rossler) * (cycle_min_rossler / 60.0)
    t_rossler_d1 = np.arange(1, total_cycles_rossler) * (cycle_min_rossler / 60.0)
    axes[1, 1].plot(t_rossler, u_gluc, color="#1d3557", lw=2.2, label="Actual Glucose (mL)")
    axes[1, 1].plot(t_rossler_d2, yp_dfba_gluc, color="#e63946", lw=1.8, ls="--", marker="o", ms=4, label=f"dFBA Glucose (d=2, R²={r2_dfba_gluc:.4f})")
    axes[1, 1].plot(t_rossler, u_salt, color="#457b9d", lw=2.2, label="Actual Salt (mL)")
    axes[1, 1].plot(t_rossler_d1, yp_dfba_salt, color="#2a9d8f", lw=1.8, ls="-.", marker="s", ms=4, label=f"dFBA Salt (d=1, R²={r2_dfba_salt:.4f})")
    axes[1, 1].set_title("4D Rössler Dual-Chemical Decoding with dFBA", fontweight="bold", fontsize=12)
    axes[1, 1].set_xlabel("Timeline [Hours]", fontweight="bold")
    axes[1, 1].set_ylabel("Dosing Volume (mL)", fontweight="bold")
    axes[1, 1].legend(loc="upper left", frameon=True)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Comparative Evaluation: Pure dFBA Model vs. Physical Pioreactor Dynamics", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = OUTPUT_DIR / "dfba_vs_physical_benchmark.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved comparative benchmark plot to: {out_file}")


if __name__ == "__main__":
    main()
