"""
ERC Chemical-Pulse Response Analysis Template
==============================================
For Pioreactor exports from the MgSO4 / (NH4)2SO4 / uracil pulse experiments.

WHAT THIS DOES
--------------
1. Loads the 6 standard Pioreactor CSV exports for one experiment (chemical).
2. Collapses the raw dosing log (which records every individual pump tick,
   not one row per pulse) into discrete pulse events, separately for
   chemical pulses ("add_media") and glucose pulses ("add_alt_media").
3. For every sensor stream, computes a per-pulse response profile:
   baseline level/variability just before the pulse, and the rate of
   change + net shift in the window just after the pulse.
4. Plots one overview figure per chemical (growth rate / nOD / OD /
   CO2 [optional] / relative humidity / spectrometer), x-axis = wall-clock
   time, with pulses marked.
5. Once you point EXPERIMENTS at all 3 chemical folders, run
   compare_chemicals() to get a tidy summary table + comparison bar plots
   across chemicals.

DIRECTORY LAYOUT
-----------------
Point each entry in EXPERIMENTS at a folder containing that chemical's
6 CSV exports (any timestamp suffix is fine, matched by prefix):

    growth_rates-*.csv
    od_readings-*.csv
    od_readings_filtered-*.csv
    as7341_spectrum_readings-*.csv
    co2_readings-*.csv        <- optional, some exports don't have this
    dosing_events-*.csv

USAGE
-----
    EXPERIMENTS = {
        "MgSO4":  "/path/to/mgso4_export",
        "NH4SO4": "/path/to/nh4so4_export",
        "Uracil": "/path/to/uracil_export",
    }
    if __name__ == "__main__":
        run_all(EXPERIMENTS, outdir="./erc_pulse_analysis_out")
"""

import glob
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ======================================================================
# CONFIG - tweak these
# ======================================================================

GAP_HOURS = 0.05          # micro-pulse clustering gap threshold (see note below)
BASELINE_MIN = 10         # minutes before pulse start used as the "baseline" window
RESPONSE_MIN = 20         # minutes after pulse start used to measure the response
RECOVERY_SEARCH_MIN = 55  # how far past the pulse to look for recovery (keep < inter-pulse
                           # gap, e.g. < 60 min for hourly chemostat pulses, so you don't
                           # accidentally search into the next pulse's response)
RECOVERY_TOL_Z = 0.5      # "recovered" = within RECOVERY_TOL_Z * baseline_std of baseline_mean
CHEM_EVENT = "add_media"        # chemical pulses (Mg/N/uracil)
GLUCOSE_EVENT = "add_alt_media"  # glucose pulses
CHEM_SOURCES = None             # e.g. ["UI"] to keep only manually-triggered chemical
                                 # doses and drop automated chemostat top-offs -- see the
                                 # note in get_pulses() before trusting the default (None = all sources)

SPECTRUM_BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
BAND_COLORS = plt.cm.plasma(np.linspace(0.05, 0.9, len(SPECTRUM_BANDS)))

PULSE_COLORS = {CHEM_EVENT: "tab:orange", GLUCOSE_EVENT: "tab:blue"}


# ======================================================================
# LOADING
# ======================================================================

def _find(folder, prefix):
    matches = sorted(glob.glob(os.path.join(folder, f"{prefix}*.csv")))
    if len(matches) > 1:
        print(f"warning: multiple files match '{prefix}*.csv' in {folder} -- "
              f"using the most recent by filename timestamp: {os.path.basename(matches[-1])}. "
              f"(Others found: {[os.path.basename(m) for m in matches[:-1]]}. "
              f"Put each experiment export in its own folder to avoid this.)")
    return matches[-1] if matches else None


def load_experiment(folder, name):
    """Load all available CSVs for one chemical experiment folder."""
    file_map = {
        "growth_rate": "growth_rates",
        "od": "od_readings-",              # trailing '-' avoids matching od_readings_filtered
        "nod": "od_readings_filtered",
        "spectrum": "as7341_spectrum_readings",
        "co2": "co2_readings",
        "dosing": "dosing_events",
    }
    data = {"name": name}
    for key, prefix in file_map.items():
        path = _find(folder, prefix)
        if path is None:
            data[key] = None
            continue
        df = pd.read_csv(path)
        df["timestamp_localtime"] = pd.to_datetime(df["timestamp_localtime"])
        df = df.sort_values("timestamp_localtime").reset_index(drop=True)
        data[key] = df

    if data["co2"] is None:
        print(f"[{name}] no CO2 export found -- CO2/humidity subplot & metrics will be skipped")
    if data["dosing"] is None:
        raise FileNotFoundError(f"[{name}] dosing_events file is required but missing")

    return data


# ======================================================================
# PULSE DETECTION (collapse micro-tick clusters into single pulse events)
# ======================================================================

def cluster_events(dosing_df, event_type, gap_hours=GAP_HOURS, sources=None):
    """Collapse a burst of individual pump ticks into one pulse event.

    sources: optional list to restrict to specific `source_of_event` values,
    e.g. sources=["UI"] to keep only manually-triggered doses and drop
    automated chemostat/turbidostat top-offs (see note in get_pulses()).

    Returns a DataFrame: pulse_start, pulse_end, n_ticks, total_volume_ml
    """
    sub = dosing_df[dosing_df["event"] == event_type]
    if sources is not None:
        sub = sub[sub["source_of_event"].isin(sources)]
    sub = sub.sort_values("timestamp_localtime")
    if sub.empty:
        return pd.DataFrame(columns=["pulse_start", "pulse_end", "n_ticks", "total_volume_ml"])

    gap = pd.Timedelta(hours=gap_hours)
    cluster_id = (sub["timestamp_localtime"].diff() > gap).cumsum()
    pulses = (
        sub.groupby(cluster_id)
        .agg(
            pulse_start=("timestamp_localtime", "first"),
            pulse_end=("timestamp_localtime", "last"),
            n_ticks=("timestamp_localtime", "size"),
            total_volume_ml=("volume_change_ml", "sum"),
        )
        .reset_index(drop=True)
    )
    return pulses


def get_pulses(data, chem_sources=CHEM_SOURCES):
    """Return (chemical_pulses, glucose_pulses) DataFrames for an experiment.

    NOTE: in at least one export, 'add_media' events come from two very
    different sources: a one-off 'UI' dose (the manual pulse you triggered)
    and repeated 'dosing_automation:chemostat' doses (automatic top-offs
    the turbidostat fires whenever OD rises, unrelated to your deliberate
    chemical-pulse schedule). Check data["dosing"]["source_of_event"]
    .value_counts() for your own exports and set CHEM_SOURCES accordingly
    -- otherwise you may be treating routine turbidostat dosing as if it
    were your intentional chemical pulses.
    """
    chem = cluster_events(data["dosing"], CHEM_EVENT, sources=chem_sources)
    gluc = cluster_events(data["dosing"], GLUCOSE_EVENT)
    return chem, gluc


# ======================================================================
# PER-PULSE RESPONSE METRICS
# ======================================================================

def pulse_response(sensor_df, time_col, value_col, pulses,
                    baseline_min=BASELINE_MIN, response_min=RESPONSE_MIN,
                    recovery_search_min=RECOVERY_SEARCH_MIN, recovery_tol_z=RECOVERY_TOL_Z):
    """For each pulse, compute:
      - baseline_mean / baseline_std  (window before the pulse)
      - response_mean                 (window after the pulse)
      - delta = response_mean - baseline_mean
      - slope_per_min = linear-fit rate of change during the response window
      - resp_range = max-min swing during the response window
      - effect_size = delta / baseline_std : a noise-normalized "per-pulse
        effect" -- lets you compare magnitude across sensors/chemicals with
        very different absolute scales and noise floors (a NaN/inf shows up
        when baseline_std ~ 0, e.g. a very flat CO2 baseline)
      - recovery_min = minutes after the pulse until the signal first comes
        back within recovery_tol_z * baseline_std of baseline_mean. NaN if
        it never recovers within recovery_search_min (keep
        recovery_search_min shorter than your inter-pulse gap, e.g. < 60 min
        for hourly pulses, so this doesn't bleed into the next pulse)
    """
    rows = []
    if sensor_df is None or pulses.empty:
        return pd.DataFrame(rows)

    for i, p in pulses.iterrows():
        t0 = p["pulse_start"]
        base = sensor_df[
            (sensor_df[time_col] >= t0 - pd.Timedelta(minutes=baseline_min))
            & (sensor_df[time_col] < t0)
        ]
        resp = sensor_df[
            (sensor_df[time_col] >= t0)
            & (sensor_df[time_col] <= t0 + pd.Timedelta(minutes=response_min))
        ]
        if base.empty or resp.empty:
            continue

        baseline_mean = base[value_col].mean()
        baseline_std = base[value_col].std()
        resp_mean = resp[value_col].mean()
        delta = resp_mean - baseline_mean

        x = (resp[time_col] - t0).dt.total_seconds() / 60.0
        y = resp[value_col].values
        slope = np.polyfit(x, y, 1)[0] if len(x) >= 2 else np.nan

        effect_size = delta / baseline_std if baseline_std and not np.isnan(baseline_std) and baseline_std > 0 else np.nan

        # recovery time: first timestamp after the pulse where the signal
        # is back within tolerance of baseline_mean
        search = sensor_df[
            (sensor_df[time_col] >= t0)
            & (sensor_df[time_col] <= t0 + pd.Timedelta(minutes=recovery_search_min))
        ]
        recovery_min = np.nan
        if not search.empty and baseline_std and not np.isnan(baseline_std) and baseline_std > 0:
            tol = recovery_tol_z * baseline_std
            within = search[(search[value_col] - baseline_mean).abs() <= tol]
            if not within.empty:
                recovery_min = (within[time_col].iloc[0] - t0).total_seconds() / 60.0

        rows.append({
            "pulse_index": i,
            "pulse_start": t0,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "response_mean": resp_mean,
            "delta": delta,
            "slope_per_min": slope,
            "resp_min": resp[value_col].min(),
            "resp_max": resp[value_col].max(),
            "resp_range": resp[value_col].max() - resp[value_col].min(),
            "effect_size": effect_size,
            "recovery_min": recovery_min,
        })
    return pd.DataFrame(rows)


def summarize_experiment(data, pulses_label="chemical"):
    """Run pulse_response() across every sensor stream for one experiment.
    Returns one tidy DataFrame with a 'sensor' column identifying the stream
    (growth_rate, nod, od_ch{channel}_a{angle}, co2_ppm, relative_humidity,
    band_{nm})."""
    chem_pulses, gluc_pulses = get_pulses(data)
    pulses = chem_pulses if pulses_label == "chemical" else gluc_pulses

    results = []

    def add(sensor_name, df, time_col, value_col):
        r = pulse_response(df, time_col, value_col, pulses)
        if not r.empty:
            r["sensor"] = sensor_name
            results.append(r)

    add("growth_rate", data["growth_rate"], "timestamp_localtime", "rate")
    add("nOD", data["nod"], "timestamp_localtime", "normalized_od_reading")

    if data["od"] is not None:
        for (angle, channel), grp in data["od"].groupby(["angle", "channel"]):
            add(f"OD_a{angle}_c{channel}", grp, "timestamp_localtime", "od_reading")

    if data["co2"] is not None:
        add("co2_ppm", data["co2"], "timestamp_localtime", "co2_reading_ppm")
        add("relative_humidity", data["co2"], "timestamp_localtime", "relative_humidity")

    if data["spectrum"] is not None:
        for band, grp in data["spectrum"].groupby("band"):
            add(f"band_{band}nm", grp, "timestamp_localtime", "reading")

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    out["chemical"] = data["name"]
    out["pulse_type"] = pulses_label
    return out


# ======================================================================
# PLOTTING - one overview figure per chemical
# ======================================================================

def _mark_pulses(ax, chem_pulses, gluc_pulses):
    for _, p in chem_pulses.iterrows():
        ax.axvline(p["pulse_start"], color=PULSE_COLORS[CHEM_EVENT], ls="--", lw=1, alpha=0.8)
    for _, p in gluc_pulses.iterrows():
        ax.axvline(p["pulse_start"], color=PULSE_COLORS[GLUCOSE_EVENT], ls="--", lw=1, alpha=0.8)


def plot_overview(data, savepath=None):
    chem_pulses, gluc_pulses = get_pulses(data)

    panels = ["growth_rate", "nod", "od", "spectrum"]
    if data["co2"] is not None:
        panels += ["co2", "humidity"]

    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 2.4 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    ax_map = dict(zip(panels, axes))

    ax = ax_map["growth_rate"]
    if data["growth_rate"] is not None:
        ax.plot(data["growth_rate"]["timestamp_localtime"], data["growth_rate"]["rate"],
                color="black", lw=0.8)
    ax.set_ylabel("Growth rate")

    ax = ax_map["nod"]
    if data["nod"] is not None:
        ax.plot(data["nod"]["timestamp_localtime"], data["nod"]["normalized_od_reading"],
                color="darkgreen", lw=0.8)
    ax.set_ylabel("nOD")

    ax = ax_map["od"]
    if data["od"] is not None:
        for (angle, channel), grp in data["od"].groupby(["angle", "channel"]):
            ax.plot(grp["timestamp_localtime"], grp["od_reading"], lw=0.7,
                    label=f"{angle}° (ch{channel})")
        ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.set_ylabel("OD (raw)")

    ax = ax_map["spectrum"]
    if data["spectrum"] is not None:
        for band, color in zip(SPECTRUM_BANDS, BAND_COLORS):
            grp = data["spectrum"][data["spectrum"]["band"] == band]
            ax.plot(grp["timestamp_localtime"], grp["reading"], lw=0.6, color=color, label=f"{band}nm")
        ax.legend(fontsize=6, ncol=4, loc="upper left")
    ax.set_ylabel("Spectral reading")

    if data["co2"] is not None:
        ax = ax_map["co2"]
        ax.plot(data["co2"]["timestamp_localtime"], data["co2"]["co2_reading_ppm"],
                color="firebrick", lw=0.8)
        ax.set_ylabel("CO2 (ppm)")

        ax = ax_map["humidity"]
        ax.plot(data["co2"]["timestamp_localtime"], data["co2"]["relative_humidity"],
                color="steelblue", lw=0.8)
        ax.set_ylabel("Rel. humidity (%)")

    for ax in axes:
        _mark_pulses(ax, chem_pulses, gluc_pulses)

    # legend for pulse lines, once, on the top axis
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=PULSE_COLORS[CHEM_EVENT], ls="--", label="chemical pulse"),
        Line2D([0], [0], color=PULSE_COLORS[GLUCOSE_EVENT], ls="--", label="glucose pulse"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%b %d"))
    fig.suptitle(f"{data['name']} pulse experiment - sensor overview", fontsize=13)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"saved {savepath}")
    return fig


# ======================================================================
# CROSS-CHEMICAL COMPARISON
# ======================================================================

def compare_chemicals(experiments, pulses_label="chemical"):
    """experiments: dict {chemical_name: folder_path}
    Returns (per_pulse_summary, agg_summary) DataFrames."""
    all_summaries = []
    loaded = {}
    for name, folder in experiments.items():
        data = load_experiment(folder, name)
        loaded[name] = data
        s = summarize_experiment(data, pulses_label=pulses_label)
        if not s.empty:
            all_summaries.append(s)

    if not all_summaries:
        raise ValueError("no pulse-response data computed for any experiment")

    per_pulse = pd.concat(all_summaries, ignore_index=True)

    agg = (
        per_pulse.groupby(["chemical", "sensor"])
        .agg(
            mean_delta=("delta", "mean"),
            mean_abs_delta=("delta", lambda s: s.abs().mean()),
            mean_slope=("slope_per_min", "mean"),
            mean_resp_range=("resp_range", "mean"),
            mean_effect_size=("effect_size", "mean"),
            mean_abs_effect_size=("effect_size", lambda s: s.abs().mean()),
            mean_recovery_min=("recovery_min", "mean"),
            pct_recovered=("recovery_min", lambda s: s.notna().mean() * 100),
            n_pulses=("delta", "count"),
        )
        .reset_index()
    )
    return per_pulse, agg, loaded


def plot_comparison(agg, sensors=("growth_rate", "nOD", "co2_ppm"), savepath=None):
    """Bar plot comparing mean_abs_delta across chemicals for chosen sensors."""
    sensors = [s for s in sensors if s in agg["sensor"].unique()]
    fig, axes = plt.subplots(1, len(sensors), figsize=(4.5 * len(sensors), 4))
    if len(sensors) == 1:
        axes = [axes]
    for ax, sensor in zip(axes, sensors):
        sub = agg[agg["sensor"] == sensor]
        ax.bar(sub["chemical"], sub["mean_abs_delta"], color="slategray")
        ax.set_title(sensor)
        ax.set_ylabel("mean |Δ| per pulse")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"saved {savepath}")
    return fig


def plot_recovery_and_effect_size(agg, sensors=("growth_rate", "nOD", "co2_ppm"), savepath=None):
    """Two-panel comparison: mean |effect size| (noise-normalized) and mean
    recovery time, across chemicals, for the chosen sensors."""
    sensors = [s for s in sensors if s in agg["sensor"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    width = 0.8 / max(len(sensors), 1)
    chemicals = sorted(agg["chemical"].unique())
    x = np.arange(len(chemicals))

    for j, sensor in enumerate(sensors):
        sub = agg[agg["sensor"] == sensor].set_index("chemical").reindex(chemicals)
        axes[0].bar(x + j * width, sub["mean_abs_effect_size"], width, label=sensor)
        axes[1].bar(x + j * width, sub["mean_recovery_min"], width, label=sensor)

    axes[0].set_title("Mean |effect size| per pulse (Δ / baseline std)")
    axes[0].set_ylabel("|effect size| (std units)")
    axes[1].set_title("Mean recovery time per pulse")
    axes[1].set_ylabel("minutes to return to baseline")

    for ax in axes:
        ax.set_xticks(x + width * (len(sensors) - 1) / 2)
        ax.set_xticklabels(chemicals, rotation=20)
        ax.legend(fontsize=8)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"saved {savepath}")
    return fig


# ======================================================================
# CROSS-CORRELATION (lead/lag between two signals, per pulse)
# ======================================================================

def cross_correlate_pulses(df_a, col_a, df_b, col_b, pulses,
                            window_min=RESPONSE_MIN, resample_sec=15, max_lag_min=10):
    """For each pulse, resample both signals onto a common time grid within
    the post-pulse response window and cross-correlate to find the lag at
    which they align best.

    Returns one row per pulse with:
      - best_lag_min : shift (minutes) applied to signal B to line it up
        with signal A. Positive => B changes AFTER A (B lags A).
        Negative => B changes BEFORE A (B leads A).
      - peak_corr : normalized correlation coefficient at that lag
        (close to +1/-1 = strong coupling, close to 0 = no linear relation
        in this window -- consistent with your coherence-analysis findings
        that ERC nonlinear structure doesn't have to show up as a strong
        *linear* lag relationship)
    """
    rows = []
    if df_a is None or df_b is None or pulses.empty:
        return pd.DataFrame(rows)

    for i, p in pulses.iterrows():
        t0, t1 = p["pulse_start"], p["pulse_start"] + pd.Timedelta(minutes=window_min)
        a = df_a[(df_a["timestamp_localtime"] >= t0) & (df_a["timestamp_localtime"] <= t1)][
            ["timestamp_localtime", col_a]
        ].dropna()
        b = df_b[(df_b["timestamp_localtime"] >= t0) & (df_b["timestamp_localtime"] <= t1)][
            ["timestamp_localtime", col_b]
        ].dropna()
        if len(a) < 3 or len(b) < 3:
            continue

        grid = pd.date_range(t0, t1, freq=f"{resample_sec}s")
        grid_num = grid.astype(np.int64)
        a_i = np.interp(grid_num, a["timestamp_localtime"].astype(np.int64), a[col_a])
        b_i = np.interp(grid_num, b["timestamp_localtime"].astype(np.int64), b[col_b])
        a_i = a_i - a_i.mean()
        b_i = b_i - b_i.mean()

        if np.allclose(a_i, 0) or np.allclose(b_i, 0):
            continue

        max_lag_steps = int(max_lag_min * 60 / resample_sec)
        corr = np.correlate(a_i, b_i, mode="full")
        lags = np.arange(-len(a_i) + 1, len(a_i))
        mask = np.abs(lags) <= max_lag_steps
        corr_r, lags_r = corr[mask], lags[mask]
        if len(corr_r) == 0:
            continue

        best_idx = np.argmax(np.abs(corr_r))
        denom = np.linalg.norm(a_i) * np.linalg.norm(b_i)
        rows.append({
            "pulse_index": i,
            "pulse_start": t0,
            "best_lag_min": lags_r[best_idx] * resample_sec / 60.0,
            "peak_corr": corr_r[best_idx] / denom if denom > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_cross_correlation(data, reference_sensors=("growth_rate", "nOD"),
                                 target_bands=SPECTRUM_BANDS, pulses_label="chemical"):
    """Cross-correlate each spectral band against each reference sensor
    (growth_rate, nOD by default), per pulse. Returns a tidy DataFrame:
    chemical, reference, band, pulse_index, best_lag_min, peak_corr."""
    chem_pulses, gluc_pulses = get_pulses(data)
    pulses = chem_pulses if pulses_label == "chemical" else gluc_pulses

    ref_map = {
        "growth_rate": (data["growth_rate"], "rate"),
        "nOD": (data["nod"], "normalized_od_reading"),
    }

    results = []
    for ref_name in reference_sensors:
        ref_df, ref_col = ref_map.get(ref_name, (None, None))
        if ref_df is None or data["spectrum"] is None:
            continue
        for band in target_bands:
            band_df = data["spectrum"][data["spectrum"]["band"] == band]
            r = cross_correlate_pulses(ref_df, ref_col, band_df, "reading", pulses)
            if not r.empty:
                r["reference"] = ref_name
                r["band"] = band
                results.append(r)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    out["chemical"] = data["name"]
    return out


def plot_cross_correlation_heatmap(cc_summary, savepath=None):
    """Heatmap of mean best_lag_min (band x chemical) faceted by reference
    sensor. Positive lag = band changes after the reference sensor."""
    references = sorted(cc_summary["reference"].unique())
    fig, axes = plt.subplots(1, len(references), figsize=(6 * len(references), 4.5))
    if len(references) == 1:
        axes = [axes]

    for ax, ref in zip(axes, references):
        sub = cc_summary[cc_summary["reference"] == ref]
        pivot = sub.pivot_table(index="band", columns="chemical", values="best_lag_min", aggfunc="mean")
        im = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto",
                        vmin=-np.nanmax(np.abs(pivot.values)), vmax=np.nanmax(np.abs(pivot.values)))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=20)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{b}nm" for b in pivot.index])
        ax.set_title(f"lag vs {ref} (min)")
        for yi in range(pivot.shape[0]):
            for xi in range(pivot.shape[1]):
                val = pivot.values[yi, xi]
                if not np.isnan(val):
                    ax.text(xi, yi, f"{val:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8, label="lag (min)")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"saved {savepath}")
    return fig


# ======================================================================
# STATISTICAL TESTING - are the chemicals actually differentiable?
# ======================================================================
#
# With only 6-10 pulses per chemical, don't reach for a t-test (assumes
# normality you can't verify with n this small). This section runs:
#   1. Kruskal-Wallis: per sensor+metric, omnibus test of "do all 3
#      chemicals differ at all" (nonparametric, no normality assumption)
#   2. Pairwise exact/Monte-Carlo permutation tests (difference in medians)
#      for every chemical pair, for sensors where the omnibus test (or
#      just curiosity) says it's worth a closer look
#   3. Benjamini-Hochberg FDR correction, since you're running this across
#      many sensors x metrics x pairs simultaneously -- uncorrected p-values
#      here will absolutely give false positives.
#
# Metrics tested: delta, effect_size, slope_per_min, recovery_min (edit
# METRICS_TO_TEST below to add/remove).

METRICS_TO_TEST = ["delta", "effect_size", "slope_per_min", "recovery_min"]
MIN_N_PER_GROUP = 4  # skip a sensor/metric/pair if either group has fewer valid pulses than this


def _benjamini_hochberg(pvals):
    """Standard BH FDR correction. Returns corrected p-values, same order as input."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    out = np.full(n, np.nan)
    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return out
    p = pvals[valid]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    bh = ranked * m / (np.arange(m) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)
    corrected = np.empty(m)
    corrected[order] = bh
    out[valid] = corrected
    return out


def kruskal_by_sensor(per_pulse, metrics=METRICS_TO_TEST, min_n=MIN_N_PER_GROUP):
    """Omnibus test per (sensor, metric): does chemical identity affect this
    metric at all, across all 3(+) groups? Nonparametric (Kruskal-Wallis),
    no normality assumption -- appropriate for small n."""
    rows = []
    for metric in metrics:
        if metric not in per_pulse.columns:
            continue
        for sensor, sgrp in per_pulse.groupby("sensor"):
            groups = [
                g[metric].dropna().values
                for _, g in sgrp.groupby("chemical")
                if g[metric].dropna().shape[0] >= min_n
            ]
            chem_names = [
                name for name, g in sgrp.groupby("chemical")
                if g[metric].dropna().shape[0] >= min_n
            ]
            if len(groups) < 2 or any(len(g) == 0 for g in groups):
                continue
            try:
                stat, p = stats.kruskal(*groups)
            except ValueError:
                continue
            rows.append({
                "metric": metric,
                "sensor": sensor,
                "n_groups": len(groups),
                "chemicals": ", ".join(chem_names),
                "group_sizes": ", ".join(str(len(g)) for g in groups),
                "kruskal_stat": stat,
                "kruskal_p": p,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        for metric in out["metric"].unique():
            mask = out["metric"] == metric
            out.loc[mask, "kruskal_p_bh"] = _benjamini_hochberg(out.loc[mask, "kruskal_p"].values)
    return out


def pairwise_tests_by_sensor(per_pulse, metrics=METRICS_TO_TEST, min_n=MIN_N_PER_GROUP,
                              n_resamples=9999, random_state=0):
    """For every (sensor, metric, chemical pair): Mann-Whitney U (rank-based)
    AND an exact/Monte-Carlo permutation test on the difference in medians.
    Report both -- if they disagree it's usually a sign the effect is
    borderline and you shouldn't over-trust either one at this sample size.
    """
    rows = []
    rng = np.random.default_rng(random_state)

    def median_diff(x, y):
        return np.median(x) - np.median(y)

    for metric in metrics:
        if metric not in per_pulse.columns:
            continue
        for sensor, sgrp in per_pulse.groupby("sensor"):
            chem_groups = {
                name: g[metric].dropna().values
                for name, g in sgrp.groupby("chemical")
                if g[metric].dropna().shape[0] >= min_n
            }
            chems = sorted(chem_groups.keys())
            for i in range(len(chems)):
                for j in range(i + 1, len(chems)):
                    a_name, b_name = chems[i], chems[j]
                    a, b = chem_groups[a_name], chem_groups[b_name]

                    try:
                        mw_stat, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")
                    except ValueError:
                        mw_p = np.nan

                    try:
                        perm = stats.permutation_test(
                            (a, b), median_diff, vectorized=False,
                            n_resamples=n_resamples, alternative="two-sided",
                            random_state=rng,
                        )
                        perm_p = perm.pvalue
                    except Exception:
                        perm_p = np.nan

                    rows.append({
                        "metric": metric,
                        "sensor": sensor,
                        "chemical_a": a_name,
                        "chemical_b": b_name,
                        "n_a": len(a),
                        "n_b": len(b),
                        "median_a": np.median(a),
                        "median_b": np.median(b),
                        "median_diff": median_diff(a, b),
                        "mannwhitney_p": mw_p,
                        "permutation_p": perm_p,
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        for metric in out["metric"].unique():
            mask = out["metric"] == metric
            out.loc[mask, "permutation_p_bh"] = _benjamini_hochberg(out.loc[mask, "permutation_p"].values)
            out.loc[mask, "mannwhitney_p_bh"] = _benjamini_hochberg(out.loc[mask, "mannwhitney_p"].values)
    return out


def run_stats(per_pulse, metrics=METRICS_TO_TEST, min_n=MIN_N_PER_GROUP, alpha=0.05):
    """Convenience wrapper: runs both tests, prints a short significant-results
    summary to console, returns (kruskal_table, pairwise_table)."""
    kw = kruskal_by_sensor(per_pulse, metrics=metrics, min_n=min_n)
    pw = pairwise_tests_by_sensor(per_pulse, metrics=metrics, min_n=min_n)

    if not kw.empty:
        sig_kw = kw[kw["kruskal_p_bh"] < alpha].sort_values("kruskal_p_bh")
        print(f"\nKruskal-Wallis omnibus: {len(sig_kw)}/{len(kw)} sensor-metric combos significant at BH-corrected alpha={alpha}")
        if not sig_kw.empty:
            print(sig_kw[["metric", "sensor", "chemicals", "kruskal_p", "kruskal_p_bh"]].to_string(index=False))
    else:
        print("\nKruskal-Wallis: no sensor/metric had enough pulses per chemical (check MIN_N_PER_GROUP / your n_pulses)")

    if not pw.empty:
        sig_pw = pw[pw["permutation_p_bh"] < alpha].sort_values("permutation_p_bh")
        print(f"\nPairwise permutation tests: {len(sig_pw)}/{len(pw)} pairs significant at BH-corrected alpha={alpha}")
        if not sig_pw.empty:
            print(sig_pw[["metric", "sensor", "chemical_a", "chemical_b", "median_diff",
                           "permutation_p", "permutation_p_bh"]].to_string(index=False))
    else:
        print("\nPairwise tests: no valid comparisons (check MIN_N_PER_GROUP / your n_pulses)")

    return kw, pw


# ======================================================================
# CONTROL / VEHICLE-CORRECTED ANALYSIS
# ======================================================================
#
# If you add a control (vehicle-only, or plain diluent) experiment to
# EXPERIMENTS, set CONTROL_LABEL to its dict key below. Two things then
# become available for free / with one extra call each:
#
#   1. The existing kruskal_by_sensor() / pairwise_tests_by_sensor() /
#      run_stats() functions already include the control as just another
#      group -- so "MgSO4 vs Control", "NH4SO4 vs Control", etc. show up
#      automatically in the pairwise table. This tells you whether a
#      chemical's pulse response is even distinguishable from the
#      mechanical/dilution artifact of dosing liquid at all.
#
#   2. add_control_corrected_columns() baseline-subtracts the control's
#      per-sensor median from every chemical's per-pulse metric, giving you
#      a "{metric}_vs_control" column -- the chemical-attributable effect
#      with the shared dosing-mechanics artifact removed. Then
#      one_sample_vs_control_tests() runs a Wilcoxon signed-rank test
#      (nonparametric one-sample test, appropriate for small n) asking
#      "is this chemical's control-corrected effect actually different
#      from zero?"

CONTROL_LABEL = None  # e.g. "Control" or "Vehicle" -- set once you add a control folder to EXPERIMENTS


def add_control_corrected_columns(per_pulse, control_label=CONTROL_LABEL, metrics=METRICS_TO_TEST):
    """Subtract the control group's per-sensor median from every other
    chemical's per-pulse values. Adds "{metric}_vs_control" columns.
    No-op (returns per_pulse unchanged) if control_label isn't set or
    isn't present in the data."""
    if control_label is None:
        print("CONTROL_LABEL not set -- skipping control-corrected columns (see run script comments)")
        return per_pulse
    if control_label not in per_pulse["chemical"].unique():
        print(f"CONTROL_LABEL='{control_label}' not found in per_pulse['chemical'] -- check EXPERIMENTS key matches exactly")
        return per_pulse

    out = per_pulse.copy()
    control = out[out["chemical"] == control_label]
    for metric in metrics:
        if metric not in out.columns:
            continue
        control_medians = control.groupby("sensor")[metric].median()
        col = f"{metric}_vs_control"
        out[col] = np.nan
        not_control = out["chemical"] != control_label
        out.loc[not_control, col] = out.loc[not_control].apply(
            lambda row: row[metric] - control_medians.get(row["sensor"], np.nan), axis=1
        )
    return out


def one_sample_vs_control_tests(per_pulse_corrected, metrics=METRICS_TO_TEST,
                                 min_n=MIN_N_PER_GROUP, control_label=CONTROL_LABEL, alpha=0.05):
    """For each (chemical, sensor, metric): Wilcoxon signed-rank test of
    whether the control-corrected effect differs from zero -- i.e. is there
    a chemical-attributable perturbation beyond the shared dosing-mechanics
    artifact captured by the control? Requires add_control_corrected_columns()
    to have been run first."""
    rows = []
    if control_label is None:
        return pd.DataFrame(rows)

    for metric in metrics:
        col = f"{metric}_vs_control"
        if col not in per_pulse_corrected.columns:
            continue
        treated = per_pulse_corrected[per_pulse_corrected["chemical"] != control_label]
        for chem, cgrp in treated.groupby("chemical"):
            for sensor, sgrp in cgrp.groupby("sensor"):
                vals = sgrp[col].dropna().values
                if len(vals) < min_n or np.allclose(vals, 0):
                    continue
                try:
                    stat, p = stats.wilcoxon(vals)
                except ValueError:
                    continue
                rows.append({
                    "metric": metric,
                    "chemical": chem,
                    "sensor": sensor,
                    "n": len(vals),
                    "median_vs_control": np.median(vals),
                    "wilcoxon_p": p,
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        for metric in out["metric"].unique():
            mask = out["metric"] == metric
            out.loc[mask, "wilcoxon_p_bh"] = _benjamini_hochberg(out.loc[mask, "wilcoxon_p"].values)
        sig = out[out["wilcoxon_p_bh"] < alpha].sort_values("wilcoxon_p_bh")
        print(f"\nControl-corrected (vs {control_label}) one-sample tests: "
              f"{len(sig)}/{len(out)} chemical-sensor-metric combos significant at BH-corrected alpha={alpha}")
        if not sig.empty:
            print(sig[["metric", "chemical", "sensor", "median_vs_control", "wilcoxon_p", "wilcoxon_p_bh"]].to_string(index=False))
    return out




def run_all(experiments, outdir="./erc_pulse_analysis_out"):
    os.makedirs(outdir, exist_ok=True)

    per_pulse, agg, loaded = compare_chemicals(experiments, pulses_label="chemical")
    per_pulse.to_csv(os.path.join(outdir, "per_pulse_response_summary.csv"), index=False)
    agg.to_csv(os.path.join(outdir, "aggregate_response_summary.csv"), index=False)

    for name, data in loaded.items():
        plot_overview(data, savepath=os.path.join(outdir, f"overview_{name}.png"))

    plot_comparison(agg, savepath=os.path.join(outdir, "chemical_comparison.png"))
    plot_recovery_and_effect_size(agg, savepath=os.path.join(outdir, "recovery_and_effect_size.png"))

    cc_all = []
    for name, data in loaded.items():
        cc = summarize_cross_correlation(data)
        if not cc.empty:
            cc_all.append(cc)
    if cc_all:
        cc_summary = pd.concat(cc_all, ignore_index=True)
        cc_summary.to_csv(os.path.join(outdir, "cross_correlation_summary.csv"), index=False)
        plot_cross_correlation_heatmap(cc_summary, savepath=os.path.join(outdir, "cross_correlation_heatmap.png"))

    kw, pw = run_stats(per_pulse)
    kw.to_csv(os.path.join(outdir, "stats_kruskal_omnibus.csv"), index=False)
    pw.to_csv(os.path.join(outdir, "stats_pairwise_permutation.csv"), index=False)

    if CONTROL_LABEL is not None:
        if CONTROL_LABEL not in experiments:
            print(f"CONTROL_LABEL='{CONTROL_LABEL}' is set but not a key in EXPERIMENTS -- skipping control-corrected analysis")
        else:
            corrected = add_control_corrected_columns(per_pulse, control_label=CONTROL_LABEL)
            corrected.to_csv(os.path.join(outdir, "per_pulse_control_corrected.csv"), index=False)
            osc = one_sample_vs_control_tests(corrected, control_label=CONTROL_LABEL)
            osc.to_csv(os.path.join(outdir, "stats_vs_control_wilcoxon.csv"), index=False)

    print(f"\nDone. Outputs in {outdir}/")
    return per_pulse, agg

# ======================================================================
if __name__ == "__main__":
    
    CONTROL_LABEL = 'Vehicle'

    EXPERIMENTS = {
        "MgSO4": "./data/magnesium sulphate pulses",
        "NH4SO4": "./data/ammonium sulphate",
        "Uracil": "./data/uracil",
        "Glucose": "./data/glucose",
        "Salt": "./data/salt",
        "Vehicle": "./data/control",
    }
    run_all(EXPERIMENTS)