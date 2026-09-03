"""Mackey-Glass periodic encoding benchmark for the pure dAMN digital twin.

Evaluates on the 25 May 6-minute continuous Mackey-Glass experiment:
- Ingests the real Pioreactor 68-cycle (6.80 hour) dataset from results/25th may
- Simulates the continuous pure dAMN digital twin on the exact periodic glucose dosing trajectory
- Performs delay-embedding sweeps (d = 0..10) under Leave-One-Out Cross-Validation
- Generates delay sweep and waveform reconstruction plots matching pioreactor/mg.png
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from damn_ode import DAMN

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = Path(r"D:\omnibio\results\25th may")
CHECKPOINT_PATH = SCRIPT_DIR / "artifacts" / "model" / "damn_checkpoint.pt"
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
    nmse = mean_squared_error(y, yp) / np.var(y)
    r2 = r2_score(y, yp)
    mae = mean_absolute_error(y, yp)
    return yp, nmse, r2, mae, rcv.alpha_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
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

    print(f"Loaded 25 May Mackey-Glass Experiment: {total_cycles} cycles ({total_cycles * cycle_min:.1f} min = {total_cycles * cycle_min / 60:.2f} h)")
    print(f"Glucose dosing input range: {y_target.min():.4f} to {y_target.max():.4f} mL (Mean: {y_target.mean():.4f} mL)")

    # 2. Build the 13 Real Hardware Sensor Features (Exact matches to mg.png)
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

    # 3. Simulate Pure Continuous dAMN Digital Twin
    u_traj = np.zeros((1, total_cycles, 4), dtype=np.float32)
    u_traj[0, :, 0] = y_target  # Media feed (glucose)
    u_traj[0, :, 2] = y_target  # Waste removal
    u_traj[0, :, 3] = y_target  # Total dose

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = DAMN(dt_min=cycle_min)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    init_biomass = torch.tensor([[0.25]], dtype=torch.float32)
    init_volume = torch.tensor([[13.5]], dtype=torch.float32)

    with torch.no_grad():
        out = model(torch.from_numpy(u_traj), init_biomass, init_volume)

    damn_biomass = out["biomass"][0].numpy()
    damn_mu = out["growth_rate"][0].numpy()
    damn_fluxes = out["fluxes"][0].numpy()
    damn_sensors = out["y_pred"][0].numpy()

    # Biological internal states & sensor outputs
    X_damn_bio = np.concatenate([damn_biomass, damn_mu, damn_fluxes], axis=-1)

    # 4. Perform Delay Embedding Sweeps (d = 0 .. 10)
    real_results, damn_results = [], []
    real_preds, damn_preds = {}, {}

    for d in range(max_delay + 1):
        # Real Hardware Sweep
        Xd_r, yd_r = delay_embed(X_real, y_target, d)
        yp_r, nmse_r, r2_r, mae_r, alpha_r = ridge_loo(Xd_r, yd_r, alphas)
        real_results.append({"delay": d, "nmse": nmse_r, "r2": r2_r, "mae": mae_r})
        real_preds[d] = (yd_r, yp_r)

        # Pure dAMN Twin Sweep
        Xd_d, yd_d = delay_embed(X_damn_bio, y_target, d)
        yp_d, nmse_d, r2_d, mae_d, alpha_d = ridge_loo(Xd_d, yd_d, alphas)
        damn_results.append({"delay": d, "nmse": nmse_d, "r2": r2_d, "mae": mae_d})
        damn_preds[d] = (yd_d, yp_d)

    df_real_sweep = pd.DataFrame(real_results)
    df_damn_sweep = pd.DataFrame(damn_results)

    best_real_row = df_real_sweep.loc[df_real_sweep["nmse"].idxmin()]
    d_best_real = int(best_real_row["delay"])
    yd_best_real, yp_best_real = real_preds[d_best_real]
    t_best_real = np.arange(d_best_real, d_best_real + len(yd_best_real)) * cycle_min

    best_damn_row = df_damn_sweep.loc[df_damn_sweep["nmse"].idxmin()]
    d_best_damn = int(best_damn_row["delay"])
    yd_best_damn, yp_best_damn = damn_preds[d_best_damn]
    t_best_damn = np.arange(d_best_damn, d_best_damn + len(yd_best_damn)) * cycle_min

    print("\n=== Real Pioreactor Yeast Hardware Sweep (Matches mg.png) ===")
    print(df_real_sweep.to_string(index=False))
    print(f"Best Real Readout: d={d_best_real} ({d_best_real * cycle_min:.0f} min) | NMSE = {best_real_row['nmse']:.4f} | R² = {best_real_row['r2']:.4f} | MAE = {best_real_row['mae']:.4f} mL")

    print("\n=== Pure dAMN In Silico Digital Twin Sweep ===")
    print(df_damn_sweep.to_string(index=False))
    print(f"Best dAMN Readout: d={d_best_damn} ({d_best_damn * cycle_min:.0f} min) | NMSE = {best_damn_row['nmse']:.4f} | R² = {best_damn_row['r2']:.4f} | MAE = {best_damn_row['mae']:.4f} mL")

    # 5. Plot Figures Matching pioreactor/mg.png Style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left Panel: Delay Sweep Curves
    axes[0].plot(df_real_sweep["delay"], df_real_sweep["nmse"], "o-", color="#1f77b4", linewidth=2.0, label="Real Yeast Hardware Readout")
    axes[0].plot(df_damn_sweep["delay"], df_damn_sweep["nmse"], "s--", color="#e63946", linewidth=2.0, label="Pure dAMN Digital Twin Readout")
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="baseline")
    axes[0].axhline(0.3, color="orange", linestyle="--", linewidth=1.2, label="good")
    axes[0].axhline(0.1, color="green", linestyle="--", linewidth=1.2, label="excellent")
    axes[0].set_xlabel("Delay embedding depth", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("LOO NMSE", fontsize=11, fontweight="bold")
    axes[0].set_title("Mackey-Glass Encoding: Delay Embedding Sweep", fontsize=12, fontweight="bold")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9, frameon=True)

    # Right Panel: Best Delayed Readout Waveforms
    axes[1].plot(time_min, y_target, "o-", color="#1f77b4", linewidth=2.2, label="True MG dosing input")
    axes[1].plot(
        t_best_real,
        yp_best_real,
        "^:",
        color="#2a9d8f",
        linewidth=1.8,
        label=f"Real Yeast (d={d_best_real}, NMSE={best_real_row['nmse']:.4f}, R²={best_real_row['r2']:.4f})",
    )
    axes[1].plot(
        t_best_damn,
        yp_best_damn,
        "s--",
        color="#e63946",
        linewidth=1.8,
        label=f"Pure dAMN Twin (d={d_best_damn}, NMSE={best_damn_row['nmse']:.4f}, R²={best_damn_row['r2']:.4f})",
    )
    axes[1].set_xlabel("Minutes since 6-minute automation start", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Media dose (mL)", fontsize=11, fontweight="bold")
    axes[1].set_title("Best Delayed Readout Waveform Reconstruction", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9, frameon=True)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Periodic Mackey-Glass Encoding Benchmark: Pure dAMN Digital Twin vs. Real Pioreactor", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()

    out_plot = args.output_dir / "mackeyglass_encoding_damn.png"
    plt.savefig(out_plot, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Mackey-Glass benchmark plot to: {out_plot}")


if __name__ == "__main__":
    main()
