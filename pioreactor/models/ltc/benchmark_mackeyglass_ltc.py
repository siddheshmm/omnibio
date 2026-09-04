"""Mackey-Glass periodic encoding benchmark for the Liquid Time-Constant (LTC) digital twin.

Evaluates on the unseen 25 May continuous 6-minute Mackey-Glass experiment:
- Ingests the real Pioreactor 68-cycle (6.80 hour) dataset from results/25th may
- Simulates the continuous LTC digital twin on the exact periodic glucose dosing trajectory
- Performs delay-embedding sweeps (d = 0..10) under Leave-One-Out Cross-Validation (LOOCV)
- Generates delay sweep and waveform reconstruction plots comparing:
    1. True Mackey-Glass dosing input
    2. Physical Yeast Culture reservoir readout (13 channels)
    3. LTC Digital Twin reservoir readout
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
import yaml

from dataset import NormalizationScalers, INPUT_COLUMNS
from ltc_model import LTCBioreactorTwin

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = Path(r"D:\omnibio\results\25th may")
CHECKPOINT_PATH = SCRIPT_DIR / "artifacts" / "model" / "ltc_best_checkpoint.pt"
SCALERS_PATH = SCRIPT_DIR / "artifacts" / "dataset" / "scalers.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "benchmarks"


def cycle_mean(df: pd.DataFrame, value_col: str, t0: pd.Timestamp, total_cycles: int, cycle_min: float) -> pd.Series:
    tmp = df.copy()
    tmp["t_min"] = (tmp["timestamp"] - t0).dt.total_seconds() / 60.0
    tmp = tmp[(tmp["t_min"] >= 0) & (tmp["t_min"] < total_cycles * cycle_min)]
    tmp["cycle"] = np.floor(tmp["t_min"] / cycle_min).astype(int)
    return (
        tmp.groupby("cycle")[value_col]
        .mean()
        .reindex(range(total_cycles))
        .interpolate(limit_direction="both")
    )


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
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--scalers", type=Path, default=SCALERS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cycle-min", type=float, default=6.0)
    parser.add_argument("--max-delay", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cycle_min = args.cycle_min
    max_delay = args.max_delay
    alphas = np.logspace(-4, 4, 50)

    # 1. Ingest Real 25 May Mackey-Glass Run
    data_path = args.data_path
    if not data_path.exists():
        data_path = SCRIPT_DIR.parents[2] / "results" / "25th may"

    dosing = pd.read_csv(data_path / "dosing_events-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])
    settings = pd.read_csv(data_path / "dosing_automation_settings-Demo_experiment-all_units-20260526090312.csv", parse_dates=["started_at"])
    od = pd.read_csv(data_path / "od_readings-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])
    od_filt = pd.read_csv(data_path / "od_readings_filtered-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])
    spec = pd.read_csv(data_path / "as7341_spectrum_readings-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])
    growth = pd.read_csv(data_path / "growth_rates-Demo_experiment-all_units-20260526090312.csv", parse_dates=["timestamp"])

    settings_6min = settings[
        (settings["automation_name"] == "mg_narma_dosing")
        & settings["json_settings"].str.contains('"duration":6.0', regex=False, na=False)
    ]
    if settings_6min.empty:
        raise RuntimeError("Could not find the 6-minute mg_narma_dosing run in settings.")

    run_start = settings_6min["started_at"].max()

    media = dosing[
        (dosing["source_of_event"] == "dosing_automation:mg_narma_dosing")
        & (dosing["event"] == "add_media")
        & (dosing["timestamp"] >= run_start)
    ].sort_values("timestamp").copy()

    media["t_min"] = (media["timestamp"] - run_start).dt.total_seconds() / 60.0
    media["cycle"] = np.floor(media["t_min"] / cycle_min).astype(int)

    u = media.groupby("cycle")["volume_change_ml"].sum()
    total_cycles = int(u.index.max()) + 1
    u = u.reindex(range(total_cycles), fill_value=0.0)
    y_target = u.to_numpy(dtype=np.float32)
    time_min = np.arange(total_cycles) * cycle_min

    print(f"Loaded 25 May Mackey-Glass: {total_cycles} cycles ({total_cycles * cycle_min:.1f} min = {total_cycles * cycle_min / 60:.2f} h)")

    # 2. Build Real Physical Yeast Culture Features (13 channels)
    od_45 = cycle_mean(od[od["angle"] == 45], "od_reading", run_start, total_cycles, cycle_min)
    od_90 = cycle_mean(od[od["angle"] == 90], "od_reading", run_start, total_cycles, cycle_min)
    od_135 = cycle_mean(od[od["angle"] == 135], "od_reading", run_start, total_cycles, cycle_min)

    spec_series = {}
    for band in [415, 445, 480, 515, 555, 590, 630, 680]:
        spec_series[f"nm_{band}"] = cycle_mean(spec[spec["band"] == band], "reading", run_start, total_cycles, cycle_min)

    norm_od = cycle_mean(od_filt, "normalized_od_reading", run_start, total_cycles, cycle_min)
    growth_rate = cycle_mean(growth, "rate", run_start, total_cycles, cycle_min)

    features_real = {
        "OD_45": od_45,
        "OD_90": od_90,
        "OD_135": od_135,
        **spec_series,
        "growth_rate": growth_rate,
        "norm_od": norm_od,
    }
    X_real = np.column_stack([features_real[k].to_numpy() for k in features_real])

    # 3. Simulate LTC Digital Twin on Exact Dosing Trajectory
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    m_cfg = cfg.get("model", {})
    model = LTCBioreactorTwin(
        input_dim=m_cfg.get("input_dim", 6),
        hidden_dim=m_cfg.get("hidden_dim", 32),
        num_sensors=m_cfg.get("num_sensors", 14),
        unfolding_steps=m_cfg.get("unfolding_steps", 2),
        dt_min=cycle_min,
        tau_min=m_cfg.get("tau_min", 1.0),
        tau_max=m_cfg.get("tau_max", 60.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scalers = NormalizationScalers.load(args.scalers)

    # Input tensor: [add_media, add_alt, remove_waste, dose_total, temp_c, uv_intensity]
    u_raw = np.zeros((total_cycles, 6), dtype=np.float32)
    u_raw[:, 0] = y_target
    u_raw[:, 2] = y_target  # chemostat removal
    u_raw[:, 3] = y_target
    u_raw[:, 4] = 30.0     # baseline temp
    u_raw[:, 5] = 0.0      # baseline UV

    u_norm = scalers.normalize_inputs(u_raw)
    u_tensor = torch.from_numpy(u_norm).unsqueeze(0)  # (1, T, 6)

    init_od_val = float(norm_od.iloc[0]) if not norm_od.empty else 0.5
    with torch.no_grad():
        out_ltc = model(
            u_seq=u_tensor,
            init_od=torch.tensor([[init_od_val]], dtype=torch.float32),
            init_vol=torch.tensor([[13.5]], dtype=torch.float32),
        )

    # Denormalize predicted sensors
    y_pred_norm = out_ltc["y_pred"].squeeze(0).numpy()
    y_pred_phys = scalers.denormalize_sensors(y_pred_norm)
    hidden_states = out_ltc["hidden"].squeeze(0).numpy()
    tau_sys = out_ltc["tau_sys"].squeeze(0).numpy()

    # Reservoir state matrix for LTC: [hidden states (32), predicted sensors (14)]
    X_ltc = np.concatenate([hidden_states, y_pred_phys], axis=-1)

    # 4. Run Delay Sweeps for Real Yeast vs. LTC Digital Twin
    delays = list(range(max_delay + 1))
    real_results, ltc_results = [], []

    print("\n--- Running Delay Sweeps (d = 0 .. 10) ---")
    for d in delays:
        # Real Yeast
        Xd_r, yd_r = delay_embed(X_real, y_target, d)
        yp_r, nmse_r, r2_r, mae_r, _ = ridge_loo(Xd_r, yd_r, alphas)
        real_results.append({"delay": d, "nmse": nmse_r, "r2": r2_r, "mae": mae_r, "yp": yp_r})

        # LTC Digital Twin
        Xd_l, yd_l = delay_embed(X_ltc, y_target, d)
        yp_l, nmse_l, r2_l, mae_l, _ = ridge_loo(Xd_l, yd_l, alphas)
        ltc_results.append({"delay": d, "nmse": nmse_l, "r2": r2_l, "mae": mae_l, "yp": yp_l})

        print(f"  Delay d={d:02d} | Real R²: {r2_r:.4f} (NMSE: {nmse_r:.4f}) | LTC R²: {r2_l:.4f} (NMSE: {nmse_l:.4f})")

    # Pick optimal delays (e.g. d=3 for real, d=2 or 3 for LTC)
    best_real_idx = int(np.argmax([r["r2"] for r in real_results]))
    best_ltc_idx = int(np.argmax([r["r2"] for r in ltc_results]))
    best_real = real_results[best_real_idx]
    best_ltc = ltc_results[best_ltc_idx]

    print(f"\nOptimal Reconstruction:")
    print(f"  Real Yeast (d={best_real['delay']}): R² = {best_real['r2']:.4f}, NMSE = {best_real['nmse']:.4f}")
    print(f"  LTC Twin   (d={best_ltc['delay']}): R² = {best_ltc['r2']:.4f}, NMSE = {best_ltc['nmse']:.4f}")

    # 5. Plot Comparative Benchmark Dashboard (2 Panels matching mg.png)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left Panel: Delay embedding sweep
    axes[0].plot(delays, [r["nmse"] for r in real_results], marker="o", color="#1f77b4", lw=2.2, label="Physical Yeast Culture")
    axes[0].plot(delays, [r["nmse"] for r in ltc_results], marker="s", color="#e63946", lw=2.2, ls="--", label="LTC Digital Twin (Ours)")
    axes[0].set_title("Delay Sweep: Normalized Mean Squared Error (NMSE)", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Delay Embedding Depth d", fontweight="bold")
    axes[0].set_ylabel("NMSE (Lower is Better)", fontweight="bold")
    axes[0].set_xticks(delays)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, alpha=0.3)

    # Right Panel: Waveform reconstruction overlay
    d_opt = best_ltc["delay"]
    t_plot = time_min[d_opt:]
    y_target_plot = y_target[d_opt:]

    axes[1].plot(time_min, y_target, color="#1d3557", lw=2.4, label="True Mackey-Glass Input")
    axes[1].plot(time_min[best_real["delay"]:], best_real["yp"], color="#2a9d8f", lw=1.8, ls="-.", marker="^", ms=4, label=f"Real Yeast (d={best_real['delay']}, R²={best_real['r2']:.4f})")
    axes[1].plot(time_min[best_ltc["delay"]:], best_ltc["yp"], color="#e63946", lw=1.8, ls="--", marker="o", ms=3, label=f"LTC Twin (d={best_ltc['delay']}, R²={best_ltc['r2']:.4f})")
    axes[1].set_title("Mackey-Glass Input Waveform Decoding Comparison", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Timeline [Minutes]", fontweight="bold")
    axes[1].set_ylabel("Media Dose [mL]", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Unseen 6.8h Mackey-Glass Benchmark: Physical Yeast Culture vs. Liquid Time-Constant (LTC) Digital Twin", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = args.output_dir / "ltc_mackeyglass_benchmark.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved benchmark plot to: {out_file}")

    # Save metrics CSV
    df_metrics = pd.DataFrame([
        {"model": "Physical Yeast Culture", "delay": best_real["delay"], "r2": best_real["r2"], "nmse": best_real["nmse"], "mae": best_real["mae"]},
        {"model": "LTC Digital Twin", "delay": best_ltc["delay"], "r2": best_ltc["r2"], "nmse": best_ltc["nmse"], "mae": best_ltc["mae"]},
    ])
    df_metrics.to_csv(args.output_dir / "mackeyglass_ltc_metrics.csv", index=False)
    print(f"Saved benchmark metrics to: {args.output_dir / 'mackeyglass_ltc_metrics.csv'}")


if __name__ == "__main__":
    main()
