# Yeast dFBA digital twin

This folder couples a *Saccharomyces cerevisiae* genome-scale model to a variable-volume Pioreactor chemostat. It is intentionally separate from the NARX model: dFBA supplies interpretable biomass, nutrient, and exchange-flux trajectories; the NARX model can later learn residual sensor/stress dynamics.

## Setup

Install COBRApy in the environment used for this project:

```powershell
cd pioreactor/models/dfba
python -m pip install -r requirements.txt
```

Obtain a pinned yeast-GEM SBML release. The repo already expects a clone at `models/yeast-GEM-src/model/yeast-GEM.xml`. You can also pass `--model` explicitly.

Review `config.yaml` before trusting results. Stock concentrations, baseline medium, and the `norm_od_to_biomass_g_per_l` scale factor are provisional assumptions for INS 491 / INS 300 cultures.

## Run one canonical NARX table

```powershell
python run_dfba.py --run-table ..\narx\artifacts\dataset\runs\pulse__glucose__20260709215429.csv
```

Outputs go to `artifacts/`:

- `<run>_dfba.csv` simulated biomass, volume, and extracellular pools
- `<run>_diagnostics.json` resolved exchange IDs and assumptions used
- `<run>_config_snapshot.json` config copy for provenance

## Run all dosing-only NARX tables

```powershell
python run_batch.py
python compare_observations.py
python plot_results.py
```

`artifacts/batch/batch_summary.csv` reports per-run status. `artifacts/batch/observation_metrics.csv` compares the provisional biomass proxy to observed `norm_od`. Generated visual dashboards are written to `artifacts/plots/`.

## Model scope

The present model uses static-optimization dFBA:

1. At each 5-minute bin, extracellular concentrations set Monod-limited exchange bounds.
2. FBA maximizes yeast-GEM growth (`r_2111`).
3. ODEs update biomass, volume, and extracellular pools from dosing, waste removal, uptake, and dilution.

This is a baseline scaffold, not a validated yeast physiology model. Temperature, UV, magnesium, osmotic shock, and regulation are not yet mechanistically represented. Salt is mapped to sodium/chloride exchanges only as a coarse proxy.

## Supported dosing conditions

Mapped from the NARX dataset:

- `glucose`
- `salt`
- `nitrogen` / `nitro`
- `sulfur`
- `uracil`
- `control`

Lorenz, Rössler, temperature, and UV multimodal runs are intentionally out of scope until the dosing-only baseline is calibrated.
