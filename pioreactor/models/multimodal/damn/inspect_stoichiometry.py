"""Diagnostic and debug script to inspect yeast-GEM stoichiometric matrix.

This script demonstrates:
1. Why the previous extraction resulted in an empty S_core (bracketed metabolite IDs vs bare IDs).
2. The true genome-scale stoichiometric matrix S (2,748 metabolites x 4,105 reactions).
3. The core exchange stoichiometry linking physical nutrients to biomass growth.
4. The pathway stoichiometry for glycolysis and biomass synthesis.
"""

from __future__ import annotations

import json
from pathlib import Path
import cobra
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
GEM_PATH = REPO_ROOT / "pioreactor" / "models" / "dfba" / "models" / "yeast-GEM-src" / "model" / "yeast-GEM.xml"


def inspect_stoichiometric_matrix(verbose: bool = True) -> dict:
    print("=" * 80)
    print("[DEBUG] YEAST-GEM STOICHIOMETRIC MATRIX INSPECTION & DEBUG REPORT")
    print("=" * 80)

    if not GEM_PATH.exists():
        raise FileNotFoundError(f"yeast-GEM model not found at {GEM_PATH}")

    print(f"Loading SBML model from: {GEM_PATH} ...")
    model = cobra.io.read_sbml_model(str(GEM_PATH))

    n_mets = len(model.metabolites)
    n_rxns = len(model.reactions)
    print(f"[OK] Model successfully loaded:")
    print(f"   * Total Metabolites (Rows of S): {n_mets}")
    print(f"   * Total Reactions  (Cols of S): {n_rxns}")
    print(f"   * Theoretical Matrix Elements:  {n_mets * n_rxns:,}")

    # 1. WHY WAS IT PREVIOUSLY EMPTY?
    print("\n" + "-" * 80)
    print("1. ROOT CAUSE ANALYSIS: WHY WAS dAMN's core_stoichiometry.json EMPTY?")
    print("-" * 80)
    buggy_ids = [
        "s_0565[e]",  # D-glucose
        "s_0399[e]",  # ammonium
        "s_1479[e]",  # sulphate
        "s_1545[e]",  # uracil
        "s_1447[e]",  # sodium
        "s_0513[e]",  # chloride
        "s_1275[e]",  # oxygen
    ]
    print("In extract_gem.py, metabolite IDs had compartment suffixes like '[e]' appended:")
    matched_buggy = [mid for mid in buggy_ids if mid in model.metabolites]
    print(f"   - Query IDs: {buggy_ids}")
    print(f"   - Matches in model.metabolites: {matched_buggy} (Length = {len(matched_buggy)})")
    print("   --> Result: len(valid_mets) == 0, causing S_core to be initialized as shape (0, 8) -> []!")

    # 2. THE CORRECT METABOLITE & REACTION MAPPING
    print("\n" + "-" * 80)
    print("2. CORRECTED NUTRIENT EXCHANGE MAPPING IN YEAST-GEM")
    print("-" * 80)
    corrected_map = {
        "glucose": {"rxn": "r_1714", "met": "s_0565", "name": "D-glucose"},
        "ammonium": {"rxn": "r_1654", "met": "s_0420", "name": "ammonium"},
        "sulfate": {"rxn": "r_2060", "met": "s_1468", "name": "sulphate"},
        "uracil": {"rxn": "r_2090", "met": "s_1551", "name": "uracil"},
        "sodium": {"rxn": "r_2049", "met": "s_1438", "name": "sodium"},
        "chloride": {"rxn": "r_4593", "met": "s_4200", "name": "chloride"},
        "oxygen": {"rxn": "r_1992", "met": "s_1277", "name": "oxygen"},
        "growth": {"rxn": "r_2111", "met": "s_0450", "name": "biomass"},
    }

    correct_mets = [v["met"] for v in corrected_map.values()]
    correct_rxns = [v["rxn"] for v in corrected_map.values()]
    matched_correct = [mid for mid in correct_mets if mid in model.metabolites]
    print(f"   - Correct IDs: {correct_mets}")
    print(f"   - Matches in model.metabolites: {len(matched_correct)} / {len(correct_mets)}")

    # 3. BUILD AND PRINT S_core (EXCHANGE SUBMATRIX)
    print("\n" + "-" * 80)
    print("3. DEBUG PRINT: CORE EXCHANGE STOICHIOMETRIC SUBMATRIX (S_core)")
    print("-" * 80)
    S_core = np.zeros((len(correct_mets), len(correct_rxns)), dtype=np.float32)
    for i, mid in enumerate(correct_mets):
        met = model.metabolites.get_by_id(mid)
        for j, rid in enumerate(correct_rxns):
            rxn = model.reactions.get_by_id(rid)
            S_core[i, j] = rxn.metabolites.get(met, 0.0)

    row_labels = [f"{v['name']} ({v['met']})" for v in corrected_map.values()]
    col_labels = [f"{k} ({v['rxn']})" for k, v in corrected_map.items()]
    df_core = pd.DataFrame(S_core, index=row_labels, columns=col_labels)
    print(df_core.to_string())

    # 4. INTERNAL CENTRAL METABOLISM & BIOMASS PRECURSOR STOICHIOMETRY
    print("\n" + "-" * 80)
    print("4. DEBUG PRINT: BIOMASS SYNTHESIS & GLYCOLYSIS STOICHIOMETRIC SLICE")
    print("-" * 80)
    # The biomass equation in yeast-GEM is r_4041
    biomass_pseudorxn = model.reactions.get_by_id("r_4041")
    print(f"Biomass Formation Pseudoreaction (r_4041):")
    print(f"   Name: {biomass_pseudorxn.name}")
    print(f"   Metabolite count: {len(biomass_pseudorxn.metabolites)}")
    print("\nTop Precursors required to assemble 1 unit of Yeast Biomass:")
    precursor_data = []
    for met, coeff in biomass_pseudorxn.metabolites.items():
        precursor_data.append({
            "met_id": met.id,
            "name": met.name,
            "compartment": met.compartment,
            "stoichiometric_coeff": coeff,
        })
    df_prec = pd.DataFrame(precursor_data).sort_values(by="stoichiometric_coeff")
    print("Reactants (Substrates consumed per unit Biomass produced):")
    print(df_prec[df_prec["stoichiometric_coeff"] < 0].to_string(index=False))
    print("\nProducts:")
    print(df_prec[df_prec["stoichiometric_coeff"] > 0].to_string(index=False))

    # 5. GLYCOLYSIS FLUX SLICE
    print("\n" + "-" * 80)
    print("5. DEBUG PRINT: GLYCOLYSIS REACTION PATHWAY SLICE")
    print("-" * 80)
    glycolysis_rxns = ["r_0534", "r_0466", "r_0467", "r_0486", "r_1054", "r_0892", "r_0886"]
    valid_glyco = [rid for rid in glycolysis_rxns if rid in model.reactions]
    glyco_data = []
    for rid in valid_glyco:
        rxn = model.reactions.get_by_id(rid)
        glyco_data.append({
            "reaction_id": rxn.id,
            "name": rxn.name,
            "stoichiometry": rxn.reaction,
        })
    print(pd.DataFrame(glyco_data).to_string(index=False))

    # 6. EXPORT POPULATED core_stoichiometry.json and core_stoichiometry.pt
    out_dir = SCRIPT_DIR / "artifacts" / "stoichiometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "core_stoichiometry.json"
    pt_path = out_dir / "core_stoichiometry.pt"

    export_dict = {
        "metabolites": correct_mets,
        "reactions": correct_rxns,
        "labels": {k: v["name"] for k, v in corrected_map.items()},
        "S_core": S_core.tolist(),
        "yield_biomass_on_glucose": 0.08110025381639172,
    }
    with json_path.open("w") as f:
        json.dump(export_dict, f, indent=2)

    torch.save(
        {
            "S_core": torch.tensor(S_core, dtype=torch.float32),
            "metabolites": correct_mets,
            "reactions": correct_rxns,
            "yield_biomass_on_glucose": 0.08110025381639172,
        },
        pt_path,
    )
    print(f"\n[OK] Saved populated stoichiometric matrix artifacts to:")
    print(f"   * {json_path}")
    print(f"   * {pt_path}")
    print("=" * 80)

    return export_dict


if __name__ == "__main__":
    inspect_stoichiometric_matrix()
