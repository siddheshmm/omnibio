# Dosing NARX Digital Twin

Ridge-regularized NARX baseline for Pioreactor dosing-only experiments. Predicts the next sensor state from lagged dosing/reactor inputs and lagged sensor history.

## Quick start

```powershell
cd pioreactor/models/narx
python -m pip install -r requirements.txt
python build_dataset.py
python train_narx.py
python evaluate_rollouts.py
python plot_results.py
```

## Design

- **Model**: Ridge regression on lagged inputs/states (linear NARX readout).
- **CV**: `GroupKFold` by `run_key` for alpha selection; `GroupShuffleSplit` for held-out runs.
- **Masking**: training rows require both current and future sensor observations.
- **Reactor state**: cumulative dose and estimated `volume_ml` are included as exogenous inputs.

Lorenz, Rössler, temperature, and UV multimodal runs remain held out.

## Interpretation

In `artifacts/model/metrics.csv`, compare `mae` vs `persistence_mae` (or the plot ratio). Values below 1.0 mean the model beats persistence on held-out runs.
