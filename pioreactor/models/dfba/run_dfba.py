"""Static-optimization dFBA for Pioreactor yeast dosing runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
os.environ["LOCALAPPDATA"] = str(HERE / "artifacts" / "localappdata")
import appdirs

appdirs.user_cache_dir = lambda **_kwargs: str(HERE / "artifacts" / "cobra_cache")
import cobra

DEFAULT_CONFIG = HERE / "config.yaml"
DEFAULT_MODEL = HERE / "models" / "yeast-GEM-src" / "model" / "yeast-GEM.xml"
DEFAULT_OUTPUT = HERE / "artifacts"
METABOLITE_ORDER = ("glucose", "ammonium", "sulfate", "uracil", "sodium", "chloride")


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def g_per_l_to_mmol_per_l(concentration_g_per_l: float, molecular_weight: float) -> float:
    return float(concentration_g_per_l) / float(molecular_weight) * 1000.0


def resolve_condition(config: dict[str, Any], condition: str) -> dict[str, Any]:
    aliases = config.get("condition_aliases", {})
    key = aliases.get(condition, condition)
    if key not in config["conditions"]:
        raise KeyError(f"Unknown condition '{condition}' (resolved '{key}')")
    return config["conditions"][key]


def stock_mmol_per_l(config: dict[str, Any], stock_key: str, metabolite: str) -> float:
    stock_g_per_l = float(config["stock_concentrations_g_per_l"].get(stock_key, 0.0))
    if stock_g_per_l <= 0.0:
        return 0.0
    if stock_key in config["molecular_weights_g_per_mol"]:
        return g_per_l_to_mmol_per_l(stock_g_per_l, config["molecular_weights_g_per_mol"][stock_key])
    return g_per_l_to_mmol_per_l(stock_g_per_l, config["molecular_weights_g_per_mol"][metabolite])


def find_reaction(model: cobra.Model, candidates: list[str], patterns: list[str], label: str, required: bool = False):
    for reaction_id in candidates:
        if reaction_id in model.reactions:
            return model.reactions.get_by_id(reaction_id)
    lowered_patterns = [pattern.lower() for pattern in patterns]
    for reaction in model.reactions:
        name = (reaction.name or "").lower()
        if any(pattern in name for pattern in lowered_patterns):
            return reaction
    if required:
        raise KeyError(f"Could not find required {label} exchange. Candidates={candidates}, patterns={patterns}")
    return None


def resolve_exchanges(model: cobra.Model, config: dict[str, Any]) -> dict[str, Any]:
    exchanges: dict[str, Any] = {}
    for label in config["exchange_candidates"]:
        exchanges[label] = find_reaction(
            model,
            config["exchange_candidates"][label],
            config.get("exchange_name_patterns", {}).get(label, []),
            label,
            required=(label in {"glucose", "growth"}),
        )
    growth = exchanges["growth"]
    if growth is not None:
        model.objective = growth
    return exchanges


def monod_uptake_bound(concentration_mmol_per_l: float, vmax: float, km: float) -> float:
    concentration = max(0.0, float(concentration_mmol_per_l))
    return -min(vmax, vmax * concentration / max(km + concentration, 1e-9))


def set_exchange_bounds(
    model: cobra.Model,
    exchanges: dict[str, Any],
    concentrations: dict[str, float],
    config: dict[str, Any],
) -> None:
    kinetics = config["uptake_kinetics"]
    vmax = float(kinetics["vmax_mmol_per_gdw_per_h"])
    km = kinetics["monod_K_mmol_per_L"]
    for metabolite in METABOLITE_ORDER:
        reaction = exchanges.get(metabolite)
        if reaction is None:
            continue
        reaction.lower_bound = monod_uptake_bound(concentrations.get(metabolite, 0.0), vmax, float(km.get(metabolite, 1.0)))
        reaction.upper_bound = 1000.0
    oxygen = exchanges.get("oxygen")
    if oxygen is not None:
        oxygen.lower_bound = float(config.get("aerobic", {}).get("oxygen_lower_bound", -20.0))
        oxygen.upper_bound = 1000.0


def load_run(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    required = {"timestamp", "condition", "add_media_ml", "remove_waste_ml"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Run table is missing {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if "add_alt_media_ml" not in frame:
        frame["add_alt_media_ml"] = 0.0
    if "dose_total_ml" not in frame:
        frame["dose_total_ml"] = frame["add_media_ml"] + frame["add_alt_media_ml"]
    return frame


def initial_concentrations(config: dict[str, Any]) -> dict[str, float]:
    baseline = config.get("baseline_medium_g_per_l", {})
    weights = config["molecular_weights_g_per_mol"]
    return {
        metabolite: g_per_l_to_mmol_per_l(float(baseline.get(metabolite, 0.0)), float(weights[metabolite]))
        for metabolite in METABOLITE_ORDER
        if metabolite in weights
    }


def initial_biomass(run: pd.DataFrame, config: dict[str, Any]) -> float:
    scale = float(config["norm_od_to_biomass_g_per_l"])
    if "norm_od" in run.columns and run["norm_od"].notna().any():
        return float(run["norm_od"].dropna().iloc[0]) * scale
    return 0.05


def dose_metabolite_rates(
    media_ml: float,
    alt_media_ml: float,
    volume_ml: float,
    dt_min: float,
    condition_cfg: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float]:
    stock_key = condition_cfg["stock_key"]
    rates = {metabolite: 0.0 for metabolite in METABOLITE_ORDER}
    dose_ml = float(media_ml + alt_media_ml)
    if dose_ml <= 0.0 or not condition_cfg.get("metabolites"):
        return rates
    for metabolite, stoich in condition_cfg["metabolites"].items():
        if metabolite not in rates:
            continue
        stock_mmol = stock_mmol_per_l(config, stock_key, metabolite)
        rates[metabolite] = dose_ml * stock_mmol * float(stoich) / max(volume_ml, 1e-6) / max(dt_min, 1e-6)
    return rates


def simulate(model_path: Path, run: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    model = cobra.io.read_sbml_model(str(model_path))
    exchanges = resolve_exchanges(model, config)
    resolved = {key: getattr(value, "id", None) for key, value in exchanges.items()}
    print("Resolved reactions:", resolved)

    condition = str(run["condition"].iloc[0]).lower()
    condition_cfg = resolve_condition(config, condition)
    start = run["timestamp"].min()
    times = (run["timestamp"] - start).dt.total_seconds().to_numpy() / 60.0
    media_ml = run["add_media_ml"].to_numpy(float)
    alt_media_ml = run["add_alt_media_ml"].to_numpy(float)
    waste_ml = run["remove_waste_ml"].to_numpy(float)

    conc0 = initial_concentrations(config)
    biomass0 = initial_biomass(run, config)
    volume0 = float(config["working_volume_ml"])
    state_names = ["biomass_gdw_per_l", "volume_ml", *[f"{met}_mmol_per_l" for met in METABOLITE_ORDER]]
    y0 = np.array(
        [biomass0, volume0, *[conc0.get(met, 0.0) for met in METABOLITE_ORDER]],
        dtype=float,
    )

    vmax = float(config["uptake_kinetics"]["vmax_mmol_per_gdw_per_h"])
    growth_id = exchanges["growth"].id
    exchange_ids = {met: exchanges[met].id for met in METABOLITE_ORDER if exchanges.get(met) is not None}

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        biomass, volume, *met_vals = np.maximum(y, [1e-9, 1e-6] + [0.0] * len(METABOLITE_ORDER))
        idx = min(max(int(np.searchsorted(times, t, side="right") - 1), 0), len(run) - 1)
        dt_min = max(float(times[idx] - times[idx - 1]) if idx > 0 else float(config["bin_minutes"]), 1e-6)
        concentrations = {met: float(met_vals[i]) for i, met in enumerate(METABOLITE_ORDER)}
        inflow_ml = float(media_ml[idx] + alt_media_ml[idx])
        dilution_per_min = inflow_ml / max(volume, 1e-6) / dt_min
        dose_rates = dose_metabolite_rates(
            media_ml[idx],
            alt_media_ml[idx],
            volume,
            dt_min,
            condition_cfg,
            config,
        )
        volume_rate = (float(media_ml[idx] + alt_media_ml[idx]) - float(waste_ml[idx])) / dt_min

        with model:
            set_exchange_bounds(model, exchanges, concentrations, config)
            solution = model.optimize()
        if solution.status != "optimal":
            return np.zeros_like(y)

        mu_per_min = max(0.0, float(solution.objective_value)) / 60.0
        derivatives = np.zeros_like(y)
        derivatives[0] = (mu_per_min - dilution_per_min) * biomass
        derivatives[1] = volume_rate
        for i, metabolite in enumerate(METABOLITE_ORDER):
            reaction_id = exchange_ids.get(metabolite)
            if reaction_id is None:
                derivatives[2 + i] = dose_rates[metabolite] - dilution_per_min * concentrations[metabolite]
                continue
            flux = float(solution.fluxes[reaction_id])
            uptake_per_min = max(0.0, -flux) * biomass / 60.0
            derivatives[2 + i] = dose_rates[metabolite] - uptake_per_min - dilution_per_min * concentrations[metabolite]
        return derivatives

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        y0,
        t_eval=times,
        method="BDF",
        rtol=1e-5,
        atol=1e-8,
    )
    output = pd.DataFrame(
        {
            "timestamp": run["timestamp"],
            "time_min": times,
            "condition": condition,
            "solver_success": bool(solution.success),
            "biomass_gdw_per_l": solution.y[0],
            "volume_ml": solution.y[1],
        }
    )
    for i, metabolite in enumerate(METABOLITE_ORDER):
        output[f"{metabolite}_mmol_per_l"] = solution.y[2 + i]
    if "norm_od" in run.columns:
        scale = float(config["norm_od_to_biomass_g_per_l"])
        output["predicted_norm_od"] = output["biomass_gdw_per_l"] / max(scale, 1e-9)
        output["observed_norm_od"] = run["norm_od"].to_numpy()
    if "growth_rate" in run.columns:
        output["observed_growth_rate_per_h"] = run["growth_rate"].to_numpy()
    diagnostics = {
        "resolved_exchanges": resolved,
        "condition": condition,
        "condition_config": condition_cfg,
        "initial_biomass_gdw_per_l": biomass0,
        "initial_concentrations_mmol_per_l": conc0,
        "solver_message": solution.message,
        "assumptions": config.get("assumptions", {}),
    }
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-table", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Yeast SBML model not found at {args.model}. "
            "Clone yeast-GEM into models/yeast-GEM-src or pass --model."
        )
    config = load_config(args.config)
    result, diagnostics = simulate(args.model, load_run(args.run_table), config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run_table.stem
    result.to_csv(args.output_dir / f"{stem}_dfba.csv", index=False)
    with (args.output_dir / f"{stem}_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    with (args.output_dir / f"{stem}_config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"wrote {len(result)} dFBA points to {args.output_dir}")


if __name__ == "__main__":
    main()
