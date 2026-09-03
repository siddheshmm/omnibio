"""Extract core metabolic stoichiometry and yields from yeast-GEM SBML model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cobra
import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"


def extract_stoichiometric_data(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    gem_path = (SCRIPT_DIR / config["gem_model_path"]).resolve()
    if not gem_path.exists():
        raise FileNotFoundError(f"yeast-GEM SBML model not found at {gem_path}")

    print(f"Loading yeast-GEM model from {gem_path}...")
    model = cobra.io.read_sbml_model(str(gem_path))

    exchanges = {
        "glucose": "r_1714",
        "ammonium": "r_1654",
        "sulfate": "r_2060",
        "uracil": "r_2090",
        "sodium": "r_2049",
        "chloride": "r_4593",
        "oxygen": "r_1992",
        "growth": "r_2111",
    }

    # Verify all reaction IDs exist in model
    for label, rxn_id in exchanges.items():
        if rxn_id not in model.reactions:
            raise KeyError(f"Reaction ID {rxn_id} for {label} not in yeast-GEM")

    # Sample optimal FBA solutions across nutrient ranges to learn stoichiometric yield bounds
    yields = {}
    with model:
        # Default baseline medium with trace minerals
        model.reactions.get_by_id("r_2049").lower_bound = -100.0  # Na
        model.reactions.get_by_id("r_4593").lower_bound = -100.0  # Cl
        model.reactions.get_by_id("r_1992").lower_bound = -20.0   # O2
        model.reactions.get_by_id("r_1714").lower_bound = -1.0    # 1 mmol/gDW/h glucose

        sol = model.optimize()
        if sol.status == "optimal":
            mu = float(sol.objective_value)
            glucose_flux = abs(float(sol.fluxes["r_1714"]))
            nh4_flux = abs(float(sol.fluxes["r_1654"]))
            so4_flux = abs(float(sol.fluxes["r_2060"]))
            o2_flux = abs(float(sol.fluxes["r_1992"]))

            yields["yield_biomass_on_glucose"] = mu / max(glucose_flux, 1e-6)
            yields["yield_biomass_on_ammonium"] = mu / max(nh4_flux, 1e-6)
            yields["yield_biomass_on_sulfate"] = mu / max(so4_flux, 1e-6)
            yields["yield_biomass_on_oxygen"] = mu / max(o2_flux, 1e-6)

    # Extract sub-matrix S_core for tracked metabolites
    # Metabolites involved in core carbon/nitrogen/energy exchange
    metabolite_ids = [
        "s_0565[e]",  # D-glucose [extracellular]
        "s_0399[e]",  # ammonium [extracellular]
        "s_1479[e]",  # sulphate [extracellular]
        "s_1545[e]",  # uracil [extracellular]
        "s_1447[e]",  # sodium [extracellular]
        "s_0513[e]",  # chloride [extracellular]
        "s_1275[e]",  # oxygen [extracellular]
    ]

    valid_mets = [mid for mid in metabolite_ids if mid in model.metabolites]
    rxn_ids = list(exchanges.values())

    S_core = np.zeros((len(valid_mets), len(rxn_ids)), dtype=np.float32)
    for i, mid in enumerate(valid_mets):
        met = model.metabolites.get_by_id(mid)
        for j, rid in enumerate(rxn_ids):
            rxn = model.reactions.get_by_id(rid)
            S_core[i, j] = rxn.metabolites.get(met, 0.0)

    extracted_data = {
        "exchanges": exchanges,
        "yields": yields,
        "metabolites": valid_mets,
        "reactions": rxn_ids,
        "S_core": S_core.tolist(),
    }

    out_file = SCRIPT_DIR / "core_stoichiometry.json"
    with out_file.open("w") as handle:
        json.dump(extracted_data, handle, indent=2)

    # Save torch tensor version for direct PyTorch loading
    torch_file = SCRIPT_DIR / "core_stoichiometry.pt"
    torch.save(
        {
            "S_core": torch.tensor(S_core, dtype=torch.float32),
            "yields": yields,
            "reactions": rxn_ids,
            "metabolites": valid_mets,
        },
        torch_file,
    )

    print(f"Extracted stoichiometric core saved to {out_file} and {torch_file}")
    return extracted_data


if __name__ == "__main__":
    extract_stoichiometric_data()
