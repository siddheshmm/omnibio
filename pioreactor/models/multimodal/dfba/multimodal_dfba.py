"""Multimodal Dynamic Flux Balance Analysis (dFBA) for Saccharomyces cerevisiae.

Couples continuous chemostat CSTR mass balances with:
1. Yeast-GEM stoichiometric linear programming (optimizing biomass yield under substrate bounds).
2. Cardinal Temperature Model (CTMI) scaling enzyme kinetics and uptake capacity with temperature T(t).
3. UV photo-inhibition and DNA damage mortality rate k_uv * I_uv(t).
4. Dual fluid pumping (media + alt-media) with dilution washout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import cobra

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parents[1] / "dfba" / "models" / "yeast-GEM-src" / "model" / "yeast-GEM.xml"

# Biological parameters
T_MIN = 5.0
T_OPT = 30.0
T_MAX = 42.0
V_MAX_DEFAULT = 0.5  # mmol / gDW / h at 30°C
K_UV = 0.05          # 1/h per 100% UV intensity
K_THERMAL = 0.005    # 1/h per (°C above 34)^2
BIOMASS_TO_OD = 1.0 / 0.15  # gDW/L to OD600

METABOLITE_ORDER = ("glucose", "ammonium", "sulfate", "uracil", "sodium", "chloride")

MONOD_KM = {
    "glucose": 0.05,
    "ammonium": 0.1,
    "sulfate": 0.05,
    "uracil": 0.01,
    "sodium": 1.0,
    "chloride": 1.0,
}

BASELINE_CONCENTRATIONS = {
    "glucose": 35.0 / 180.156 * 1000.0,
    "ammonium": 5.0 / 18.039 * 1000.0,
    "sulfate": 2.0 / 96.06 * 1000.0,
    "uracil": 0.05 / 112.08 * 1000.0,
    "sodium": 0.5 / 22.99 * 1000.0,
    "chloride": 0.75 / 35.45 * 1000.0,
}

EXCHANGE_CANDIDATES = {
    "glucose": ["r_1714", "EX_glc__D_e"],
    "ammonium": ["r_1654", "EX_nh4_e"],
    "sulfate": ["r_2060", "EX_so4_e"],
    "uracil": ["r_2090", "EX_ura_e"],
    "sodium": ["r_2049", "EX_na1_e"],
    "chloride": ["r_4593", "EX_cl_e"],
    "oxygen": ["r_1992", "EX_o2_e"],
    "growth": ["r_2111", "BIOMASS_SC5_notrace"],
}


def ctmi_temperature_multiplier(temp_c: float) -> float:
    """Cardinal Temperature Model with Inflection (CTMI) for yeast metabolic rate."""
    t = float(temp_c)
    if t <= T_MIN or t >= T_MAX:
        return 0.0
    num = (t - T_MAX) * ((t - T_MIN) ** 2)
    den = (T_OPT - T_MIN) * (
        (T_OPT - T_MIN) * (t - T_OPT) - (T_OPT - T_MAX) * (T_OPT + T_MIN - 2.0 * t)
    )
    if abs(den) < 1e-9:
        return 1.0
    return max(0.0, min(1.3, float(num / den)))


def monod_bound(conc: float, vmax: float, km: float) -> float:
    c = max(0.0, float(conc))
    return -min(vmax, vmax * c / max(km + c, 1e-9))


class MultimodalDFBA:
    """Multimodal dFBA solver combining Yeast-GEM with thermal and UV kinetics."""

    def __init__(self, xml_path: Path = MODEL_PATH):
        if not xml_path.exists():
            raise FileNotFoundError(f"yeast-GEM model not found at {xml_path}")
        self.model = cobra.io.read_sbml_model(str(xml_path))
        self.model.solver = "glpk"

        self.exchanges = {}
        for label, candidates in EXCHANGE_CANDIDATES.items():
            rxn = None
            for cid in candidates:
                if cid in self.model.reactions:
                    rxn = self.model.reactions.get_by_id(cid)
                    break
            self.exchanges[label] = rxn

        if self.exchanges["growth"] is not None:
            self.model.objective = self.exchanges["growth"]

    def step(
        self,
        dt_hours: float,
        volume_ml: float,
        biomass_gdw_l: float,
        concentrations: dict[str, float],
        inflow_ml: float,
        outflow_ml: float,
        temp_c: float,
        uv_intensity_pct: float,
    ) -> dict[str, Any]:
        dt = max(dt_hours, 1e-6)
        vol_l = max(volume_ml, 1.0) / 1000.0
        x = max(biomass_gdw_l, 1e-5)

        # 1. Temperature multiplier
        k_t = ctmi_temperature_multiplier(temp_c)
        vmax = V_MAX_DEFAULT * k_t

        # 2. Set bounds in Yeast-GEM
        with self.model:
            for met in METABOLITE_ORDER:
                rxn = self.exchanges.get(met)
                if rxn is not None:
                    c = concentrations.get(met, 0.0)
                    km = MONOD_KM.get(met, 0.1)
                    rxn.lower_bound = monod_bound(c, vmax, km)
                    rxn.upper_bound = 1000.0

            ox = self.exchanges.get("oxygen")
            if ox is not None:
                ox.lower_bound = -20.0
                ox.upper_bound = 1000.0

            solution = self.model.optimize()

        if solution.status == "optimal":
            mu_fba = max(0.0, float(solution.objective_value))
        else:
            mu_fba = 0.0

        # 3. Stress and environmental death rates
        d_uv = K_UV * (max(0.0, uv_intensity_pct) / 100.0)
        d_thermal = K_THERMAL * (max(0.0, temp_c - 34.0) ** 2)
        mu_net = max(-0.5, mu_fba - d_uv - d_thermal)

        # 4. Fluid dynamics and continuous mass balance
        dilution = (inflow_ml / 1000.0) / vol_l / dt  # 1/h
        new_vol_ml = max(1.0, volume_ml + inflow_ml - outflow_ml)
        new_x = max(1e-5, x + dt * (mu_net * x - dilution * x))

        new_concs = {}
        for met in METABOLITE_ORDER:
            c = concentrations.get(met, 0.0)
            c_in = BASELINE_CONCENTRATIONS.get(met, 0.0)
            uptake = abs(monod_bound(c, vmax, MONOD_KM.get(met, 0.1)))
            new_c = max(0.0, c + dt * (dilution * (c_in - c) - uptake * x))
            new_concs[met] = new_c

        return {
            "volume_ml": new_vol_ml,
            "biomass_gdw_l": new_x,
            "norm_od": new_x * 0.15,
            "mu_fba": mu_fba,
            "mu_net": mu_net,
            "d_uv": d_uv,
            "d_thermal": d_thermal,
            "concentrations": new_concs,
        }

    def simulate_trajectory(self, run_df: pd.DataFrame, initial_od: float | None = None) -> pd.DataFrame:
        dt_hours = 5.0 / 60.0
        v_init = float(run_df["volume_ml"].iloc[0]) if "volume_ml" in run_df else 13.5

        if initial_od is None:
            if "norm_od" in run_df and run_df["norm_od"].dropna().shape[0] > 0:
                od_0 = float(run_df["norm_od"].dropna().iloc[0])
            else:
                od_0 = 0.5
        else:
            od_0 = initial_od

        x_0 = od_0 / 0.15
        concs = dict(BASELINE_CONCENTRATIONS)

        records = []
        vol = v_init
        x = x_0

        for idx, row in run_df.iterrows():
            inflow = float(row.get("add_media_ml", 0.0)) + float(row.get("add_alt_media_ml", 0.0))
            outflow = float(row.get("remove_waste_ml", 0.0))
            temp_c = float(row.get("temp_c", 30.0))
            uv_pct = float(row.get("uv_intensity", 0.0))

            res = self.step(
                dt_hours=dt_hours,
                volume_ml=vol,
                biomass_gdw_l=x,
                concentrations=concs,
                inflow_ml=inflow,
                outflow_ml=outflow,
                temp_c=temp_c,
                uv_intensity_pct=uv_pct,
            )

            records.append(
                {
                    "timestamp": row.get("timestamp", idx),
                    "volume_ml": res["volume_ml"],
                    "biomass_gdw_l": res["biomass_gdw_l"],
                    "norm_od": res["norm_od"],
                    "mu_fba": res["mu_fba"],
                    "mu_net": res["mu_net"],
                    "d_uv": res["d_uv"],
                    "d_thermal": res["d_thermal"],
                }
            )

            vol = res["volume_ml"]
            x = res["biomass_gdw_l"]
            concs = res["concentrations"]

        return pd.DataFrame(records)
