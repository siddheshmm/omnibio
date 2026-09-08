"""Generate separate high-resolution benchmark plots for Mackey-Glass and Rössler.

1. dfba_mackey_glass_benchmark.png: 2-panel figure for the periodic Mackey-Glass experiment.
2. dfba_rossler_benchmark.png: 2-panel (or 3-panel) figure for the 4D Rössler experiment.
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
from sklearn.preprocessing import StandardScaler

import run_dfba

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
DEFAULT_MODEL = HERE / "models" / "yeast-GEM-src" / "model" / "yeast-GEM.xml"
OUTPUT_DIR = HERE / "artifacts" / "plots"
DAMN_DIR = HERE / ".." / "damn"


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


def run_mackey_glass() -> None:
    print("Simulating and plotting Mackey-Glass experiment...")
    config = run_dfba.load_config(DEFAULT_CONFIG)

    data_path_mg = Path(r"D:\omnibio\results\25th may")
    cycle_min_mg = 6.0
    dosing_mg = pd.read_csv(data_path_mg / "dosing_events-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])
    settings_mg = pd.read_csv(data_path_mg / "dosing_automation_settings-Demo_experiment-all_units-20260526090312.csv", parse_dates=["started_at"])
    od_filt_mg = pd.read_csv(data_path_mg / "od_readings_filtered-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])
    od_mg = pd.read_csv(data_path_mg / "od_readings-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])
    spec_mg = pd.read_csv(data_path_mg / "as7341_spectrum_readings-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])
    growth_mg = pd.read_csv(data_path_mg / "growth_rates-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])

    mask_mg = (settings_mg["automation_name"] == "mg_narma_dosing") & settings_mg["json_settings"].str.contains("duration", na=False)
    run_start_mg = settings_mg[mask_mg]["started_at"].max()

    media_mg = dosing_mg[(dosing_mg["source_of_event"] == "dosing_automation:mg_narma_dosing") & (dosing_mg["event"] == "add_media") & (dosing_mg["timestamp"] >= run_start_mg)].sort_values("timestamp").copy()
    media_mg["t_min"] = (media_mg["timestamp"] - run_start_mg).dt.total_seconds() / 60.0
    media_mg["cycle"] = np.floor(media_mg["t_min"] / cycle_min_mg).astype(int)

    u_mg = media_mg.groupby("cycle")["volume_change_ml"].sum()
    total_cycles_mg = int(u_mg.index.max()) + 1
    u_mg = u_mg.reindex(range(total_cycles_mg), fill_value=0.0)
    y_target_mg = u_mg.to_numpy(dtype=np.float32)

    def get_cycle_mean(df, val_col, filter_col=None, filter_val=None):
        tmp = df.copy()
        if filter_col is not None:
            tmp = tmp[tmp[filter_col] == filter_val]
        tmp["t_min"] = (tmp["timestamp"] - run_start_mg).dt.total_seconds() / 60.0
        tmp = tmp[(tmp["t_min"] >= 0) & (tmp["t_min"] < total_cycles_mg * cycle_min_mg)]
        tmp["cycle"] = np.floor(tmp["t_min"] / cycle_min_mg).astype(int)
        return tmp.groupby("cycle")[val_col].mean().reindex(range(total_cycles_mg)).interpolate(limit_direction="both").values

    norm_od_mg = get_cycle_mean(od_filt_mg, "normalized_od_reading")

    # Real Yeast 13-feature readout
    real_feats = [
        get_cycle_mean(od_mg, "od_reading", "angle", 45),
        get_cycle_mean(od_mg, "od_reading", "angle", 90),
        get_cycle_mean(od_mg, "od_reading", "angle", 135),
    ]
    for band in [415, 445, 480, 515, 555, 590, 630, 680]:
        real_feats.append(get_cycle_mean(spec_mg, "reading", "band", band))
    real_feats.append(get_cycle_mean(growth_mg, "rate"))
    real_feats.append(norm_od_mg)
    X_real_mg = np.column_stack(real_feats)

    Xd_real, yd_real = delay_embed(X_real_mg, y_target_mg, 3)
    yp_real_mg, nmse_real, r2_real, mae_real = ridge_loo(Xd_real, yd_real, np.logspace(-4, 4, 50))

    # Run dFBA
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
    Xd_dfba, yd_dfba = delay_embed(X_dfba_mg, y_target_mg, 2)
    yp_dfba_mg, nmse_dfba, r2_dfba, mae_dfba = ridge_loo(Xd_dfba, yd_dfba, np.logspace(-4, 4, 50))

    # Run dAMN
    import sys
    sys.path.insert(0, str(DAMN_DIR))
    from damn_ode import DAMN
    u_traj_mg = np.zeros((1, total_cycles_mg, 4), dtype=np.float32)
    u_traj_mg[0, :, 0] = y_target_mg
    u_traj_mg[0, :, 2] = y_target_mg
    u_traj_mg[0, :, 3] = y_target_mg

    checkpoint = torch.load(DAMN_DIR / "artifacts" / "model" / "damn_checkpoint.pt", map_location="cpu")
    model_damn = DAMN(dt_min=cycle_min_mg)
    model_damn.load_state_dict(checkpoint["model_state"])
    model_damn.eval()

    with torch.no_grad():
        out_damn = model_damn(torch.from_numpy(u_traj_mg), torch.tensor([[0.25]]), torch.tensor([[13.5]]))
    damn_biomass = out_damn["biomass"][0].numpy()
    damn_mu = out_damn["growth_rate"][0].numpy()
    damn_fluxes = out_damn["fluxes"][0].numpy()
    X_damn_mg = np.concatenate([damn_biomass, damn_mu, damn_fluxes], axis=-1)
    Xd_damn, yd_damn = delay_embed(X_damn_mg, y_target_mg, 2)
    yp_damn_mg, nmse_damn, r2_damn, mae_damn = ridge_loo(Xd_damn, yd_damn, np.logspace(-4, 4, 50))

    # Plot Separate Mackey-Glass Figure (1x2)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Trajectory Growth
    t_mg = np.arange(total_cycles_mg) * cycle_min_mg
    axes[0].plot(t_mg, norm_od_mg, color="#1f77b4", lw=2.4, label="Physical Culture (Observed norm_od)")
    axes[0].plot(t_mg, pred_od_dfba_mg, color="#9c27b0", lw=2.0, ls="--", label="dFBA Simulation (Yeast-GEM)")
    axes[0].set_title("Mackey-Glass (6.8h): Culture Growth vs. dFBA Trajectory", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Timeline [Minutes]", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Normalized OD", fontweight="bold", fontsize=11)
    axes[0].legend(loc="best", frameon=True)
    axes[0].grid(True, alpha=0.3)

    # Right: Waveform Decoding
    t_mg_d3 = np.arange(3, total_cycles_mg) * cycle_min_mg
    t_mg_d2 = np.arange(2, total_cycles_mg) * cycle_min_mg
    axes[1].plot(t_mg, y_target_mg, color="#1d3557", lw=2.4, label="True MG Dosing Input")
    axes[1].plot(t_mg_d3, yp_real_mg, color="#2a9d8f", lw=1.8, ls="-.", marker="^", ms=4, label=f"Real Yeast (d=3, R²={r2_real:.4f})")
    axes[1].plot(t_mg_d2, yp_dfba_mg, color="#9c27b0", lw=1.8, ls="--", marker="s", ms=3, label=f"dFBA Readout (d=2, R²={r2_dfba:.4f})")
    axes[1].plot(t_mg_d2, yp_damn_mg, color="#e63946", lw=1.8, ls=":", marker="o", ms=3, label=f"dAMN Twin (d=2, R²={r2_damn:.4f})")
    axes[1].set_title("Mackey-Glass Input Decoding: Real Yeast vs. dFBA vs. dAMN", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Timeline [Minutes]", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Media dose (mL)", fontweight="bold", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Periodic Mackey-Glass Experiment: Pure dFBA Model vs. Physical Pioreactor Dynamics", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUT_DIR / "dfba_mackey_glass_benchmark.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved separate Mackey-Glass benchmark plot to: {out_file}")


def run_rossler() -> None:
    print("Simulating and plotting 4D Rössler experiment...")
    config = run_dfba.load_config(DEFAULT_CONFIG)

    data_dir_rossler = Path(r"D:\omnibio\pioreactor\data\rossler")
    total_cycles_rossler = 29
    cycle_min_rossler = 25.0

    u_gluc = np.array([0.29, 0.22, 0.28, 0.22, 0.23, 0.25, 0.17, 0.27, 0.19, 0.30, 0.11, 0.34, 0.19, 0.27, 0.18, 0.21, 0.22, 0.13, 0.26, 0.14, 0.30, 0.03, 0.35, 0.15, 0.23, 0.14, 0.11, 0.21, 0.04], dtype=np.float32)
    u_salt = np.array([0.00, 0.01, 0.02, 0.03, 0.05, 0.04, 0.05, 0.02, 0.03, 0.04, 0.06, 0.08, 0.03, 0.05, 0.07, 0.09, 0.08, 0.10, 0.05, 0.07, 0.10, 0.12, 0.15, 0.09, 0.12, 0.14, 0.18, 0.13, 0.17], dtype=np.float32)

    df_raw_rossler = pd.read_csv(data_dir_rossler / "results" / "rossler_raw_state_matrix.csv")
    real_norm_od_rossler = df_raw_rossler["norm_od"].values
    X_real_rossler = df_raw_rossler[[c for c in df_raw_rossler.columns if c != "cycle"]].values

    # Real Yeast decoding
    Xd_gr, yd_gr = delay_embed(X_real_rossler, u_gluc, 2)
    yp_real_gluc, _, r2_real_gluc, _ = ridge_loo(Xd_gr, yd_gr, np.logspace(-4, 6, 100))

    Xd_sr, yd_sr = delay_embed(X_real_rossler, u_salt, 0)
    yp_real_salt, _, r2_real_salt, _ = ridge_loo(Xd_sr, yd_sr, np.logspace(-4, 6, 100))

    # Run dFBA
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
    yp_dfba_gluc, _, r2_dfba_gluc, _ = ridge_loo(Xd_g, yd_g, np.logspace(-4, 6, 100))

    # Salt decoding on dFBA (d=1)
    Xd_s, yd_s = delay_embed(X_dfba_rossler, u_salt, 1)
    yp_dfba_salt, _, r2_dfba_salt, _ = ridge_loo(Xd_s, yd_s, np.logspace(-4, 6, 100))

    # Run dAMN
    import sys
    sys.path.insert(0, str(DAMN_DIR))
    from damn_ode import DAMN
    dt_min_rossler = 5.0
    steps_per_cycle = int(cycle_min_rossler / dt_min_rossler)
    n_timesteps = total_cycles_rossler * steps_per_cycle

    u_traj_rossler = np.zeros((1, n_timesteps, 4), dtype=np.float32)
    for c in range(total_cycles_rossler):
        step_idx = c * steps_per_cycle
        u_traj_rossler[0, step_idx, 0] = u_gluc[c]
        u_traj_rossler[0, step_idx, 1] = u_salt[c]
        u_traj_rossler[0, step_idx, 2] = u_gluc[c] + u_salt[c]
        u_traj_rossler[0, step_idx, 3] = u_gluc[c] + u_salt[c]

    checkpoint = torch.load(DAMN_DIR / "artifacts" / "model" / "damn_checkpoint.pt", map_location="cpu")
    model_damn = DAMN(dt_min=dt_min_rossler)
    model_damn.load_state_dict(checkpoint["model_state"])
    model_damn.eval()

    with torch.no_grad():
        out_damn = model_damn(torch.from_numpy(u_traj_rossler), torch.tensor([[0.225]]), torch.tensor([[13.5]]))

    damn_biomass = out_damn["biomass"][0].numpy()
    damn_mu = out_damn["growth_rate"][0].numpy()
    damn_sensors = out_damn["y_pred"][0].numpy()

    damn_cycles_bio, damn_cycles_sensors = [], []
    for c in range(total_cycles_rossler):
        start = c * steps_per_cycle
        end = (c + 1) * steps_per_cycle
        damn_cycles_bio.append(np.mean(np.concatenate([damn_biomass[start:end], damn_mu[start:end]], axis=-1), axis=0))
        damn_cycles_sensors.append(np.mean(damn_sensors[start:end], axis=0))
    X_damn_bio = np.array(damn_cycles_bio)
    X_damn_sensors = np.array(damn_cycles_sensors)

    Xd_gd, yd_gd = delay_embed(X_damn_bio, u_gluc, 4)
    yp_damn_gluc, _, r2_damn_gluc, _ = ridge_loo(Xd_gd, yd_gd, np.logspace(-4, 6, 100))

    Xd_sd, yd_sd = delay_embed(X_damn_sensors, u_salt, 3)
    yp_damn_salt, _, r2_damn_salt, _ = ridge_loo(Xd_sd, yd_sd, np.logspace(-4, 6, 100))

    # Plot Separate 4D Rössler Figure (1x2 or 2 panels)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Trajectory Growth
    t_rossler = np.arange(total_cycles_rossler) * (cycle_min_rossler / 60.0)
    axes[0].plot(t_rossler, real_norm_od_rossler, color="#1f77b4", lw=2.4, label="Physical Culture (Observed norm_od)")
    axes[0].plot(t_rossler, pred_od_dfba_rossler, color="#9c27b0", lw=2.0, ls="--", label="dFBA Simulation (Yeast-GEM)")
    axes[0].set_title("4D Rössler (12.1h): Culture Growth vs. dFBA Trajectory", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Timeline [Hours]", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Normalized OD", fontweight="bold", fontsize=11)
    axes[0].legend(loc="best", frameon=True)
    axes[0].grid(True, alpha=0.3)

    # Right: Dual-Chemical Decoding (Glucose & Salt)
    t_rossler_d2 = np.arange(2, total_cycles_rossler) * (cycle_min_rossler / 60.0)
    t_rossler_d1 = np.arange(1, total_cycles_rossler) * (cycle_min_rossler / 60.0)
    axes[1].plot(t_rossler, u_gluc, color="#1d3557", lw=2.2, label="Actual Glucose (mL)")
    axes[1].plot(t_rossler_d2, yp_dfba_gluc, color="#e63946", lw=1.8, ls="--", marker="o", ms=4, label=f"dFBA Glucose (d=2, R²={r2_dfba_gluc:.4f})")
    axes[1].plot(t_rossler, u_salt, color="#457b9d", lw=2.2, label="Actual Salt (mL)")
    axes[1].plot(t_rossler_d1, yp_dfba_salt, color="#2a9d8f", lw=1.8, ls="-.", marker="s", ms=4, label=f"dFBA Salt (d=1, R²={r2_dfba_salt:.4f})")
    axes[1].set_title(f"4D Rössler Decoding: Glucose (R²={r2_dfba_gluc:.4f}) vs. Salt (R²={r2_dfba_salt:.4f})", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Timeline [Hours]", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Dosing Volume (mL)", fontweight="bold", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("4D Rössler Hyperchaotic Experiment: Pure dFBA Model vs. Physical Pioreactor Dynamics", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUT_DIR / "dfba_rossler_benchmark.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved separate 4D Rössler benchmark plot to: {out_file}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_mackey_glass()
    run_rossler()


if __name__ == "__main__":
    main()
