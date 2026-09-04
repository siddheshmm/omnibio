# Liquid Time-Constant (LTC) Yeast Bioreactor Digital Twin

Continuous-time neural-mechanistic digital twin for *Saccharomyces cerevisiae* bioreactor cultures powered by **Liquid Time-Constant (LTC) networks** (Hasani et al., AAAI 2021).

Couples continuous-stirred tank reactor (CSTR) mass balances with input-dependent varying time-constants ($\tau_{\text{sys}}$) to model multi-scale biological dynamics across chemical dosing, temperature, and UV perturbations.

---

## Architecture Overview

1. **Continuous Fluid & Physical Mass Balance**:
   - Continuous volume ODE: $V(t + \Delta t) = V(t) + \Delta V_{\text{in}} - \Delta V_{\text{out}}$
   - Dilution rate: $D(t) = \frac{\Delta V_{\text{in}}}{V(t) \cdot \Delta t}$
2. **LTC Recurrent Core (`ltc_cell.py`)**:
   - Closed-form **Fused Semi-Implicit Euler Solver**:
     $$\frac{dx}{dt} = -\left[ \frac{1}{\tau} + f(x, I) \right] x(t) + f(x, I) A$$
   - Effective liquid time-constant: $\tau_{\text{sys}}(x, I) = \frac{\tau}{1 + \tau f(x, I)}$
3. **Pure State Decoder (`ltc_model.py`)**:
   - Decodes $[x_{\text{LTC}}(t), V(t)]$ into all 14 physical sensor channels without direct algebraic feedthrough shortcuts.

---

## Datasets Ingested (`dataset.py`)

- **Pulse Experiments (20 runs)**:
  - Control / water ($1\times$)
  - Glucose ($3\times$: runs 1, 2, and 3)
  - Nitrogen / ammonium sulfate ($5\times$)
  - Salt / NaCl ($5\times$)
  - Sulfur / magnesium sulfate ($5\times$)
  - Uracil ($1\times$)
- **Sine Wave Encoding (7 runs)**:
  - Chemical dosing ($5\times$: Glucose, Nitro, Salt $\times 2$, Sulfur)
  - Temperature modulation ($1\times$)
  - UV irradiation ($1\times$)
- **Mackey-Glass Encoding (6 runs)**:
  - Chemical dosing ($4\times$: Glucose, Nitro, Salt, Sulfur)
  - Temperature ($1\times$)
  - UV ($1\times$)

---

## Quick Start

```powershell
cd pioreactor/models/ltc

# 1. Ingest and synchronize raw data into 5-minute binned tables
python dataset.py

# 2. Train the LTC digital twin via BPTT
python train_ltc.py

# 3. Evaluate continuous rollouts across all runs
python evaluate_rollouts.py

# 4. Run biocomputing benchmarks on unseen continuous chaotic data
python benchmark_mackeyglass_ltc.py
python benchmark_rossler_ltc.py
```

---

## Benchmark Results on Unseen Continuous Test Runs

### 1. 25 May Continuous Periodic Mackey-Glass (6.80h)

| System | Delay Embedding ($d$) | Reconstruction $R^2$ | NMSE |
| :--- | :---: | :---: | :---: |
| **Physical Yeast Culture (Hardware)** | $d=3$ | **0.9641** | 0.0359 |
| **Pure dFBA Simulation (yeast-GEM)** | $d=2$ | **0.9618** | 0.0382 |
| **LTC Digital Twin (Ours)** | $d=3$ | **0.9996** | **0.0004** |
| **LTC Digital Twin (Ours)** | $d=10$ | **0.9998** | **0.0002** |

*Dashboard: `artifacts/benchmarks/ltc_mackeyglass_benchmark.png`*

---

### 2. 4D Rössler Hyperchaotic Dual-Chemical Experiment (12.08h)

| Chemical Driver | System | Delay ($d$) | Reconstruction $R^2$ | NMSE |
| :--- | :--- | :---: | :---: | :---: |
| **Glucose** | Physical Yeast Culture | $d=2$ | 0.4639 | 0.5361 |
| **Glucose** | **LTC Digital Twin (Ours)** | $d=2$ | **0.9542** | **0.0458** |
| **Salt** | Physical Yeast Culture | $d=0$ | 0.7616 | 0.2384 |
| **Salt** | **LTC Digital Twin (Ours)** | $d=1$ | **0.8107** | **0.1893** |

*Dashboard: `artifacts/benchmarks/ltc_rossler_benchmark.png`*
