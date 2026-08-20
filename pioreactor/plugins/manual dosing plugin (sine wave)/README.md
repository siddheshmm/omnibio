# Manual Dosing Control (Sine Wave & Sequence) Plugin

**Plugin Name:** `manual_dosing_control`  
**Version:** `v0.7.0`  
**Location:** `pioreactor/plugins/manual dosing plugin (sine wave)/`

---

## 1. Overview & Purpose

`manual_dosing_control` is a custom Pioreactor dosing automation plugin designed for scheduled, volume-controlled liquid delivery with support for both:
1. **Explicit Volume Sequences**: Directly passing a list of volumes (e.g. `0.05, 0.10, 0.20, 0.25, 0.20, 0.10`) to execute clean, peak-aligned sinusoidal trajectories.
2. **Mathematical Sine Waves**: Calculating continuous sine waves dynamically from mean, amplitude, and period.

In Biophysical / Biological Reservoir Computing (BRC) and time-series perturbation experiments, driving a continuous yeast culture (*Saccharomyces cerevisiae*) with a periodic nutrient signal requires precise fluid handling without causing dilution shock.

---

## 2. Recommended Optimal Sinusoidal Sequence

To avoid flat plateaus and ensure the culture receives a continuously accelerating/decelerating nutrient gradient, use the **symmetrical 6-step sine trajectory**:

$$\mathbf{[0.05\text{ mL} \longrightarrow 0.10\text{ mL} \longrightarrow 0.20\text{ mL} \longrightarrow 0.25\text{ mL (Peak)} \longrightarrow 0.20\text{ mL} \longrightarrow 0.10\text{ mL}]}$$

* **Trough**: Exactly `0.05 mL` at cycle start.
* **Peak**: Exactly `0.25 mL` at cycle midpoint.
* **Non-Zero Gradient**: Every 40-minute step has a unique volume and non-zero slope ($\frac{du}{dt} \neq 0$), maximizing the dynamic richness for linear readout decoding ($R^2$).

---

## 3. Configuration Parameters

| Parameter | Type | Unit | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `volume_sequence` | `string` | `mL` | `""` | Comma-separated volume list (e.g. `"0.05, 0.10, 0.20, 0.25, 0.20, 0.10"`). |
| `duration` | `float` | `min` | *inherited* | Time interval between dosing cycles (e.g., `40` minutes). |
| `media_ml_mean` | `float` | `mL` | `1.0` | Baseline volume (used when `volume_sequence` is empty). |
| `media_ml_amplitude` | `float` | `mL` | `0.0` | Amplitude (used when `volume_sequence` is empty). |
| `sine_period_cycles` | `float` | `cycles` | `0.0` | Cycles per period (used when `volume_sequence` is empty). |
| `alt_media_ml` | `float` | `mL` | `0.0` | Fixed volume for secondary pump. |
| `waste_ml` | `float` | `mL` | `0.15` | Volume removed by waste pump per cycle. |
| `pump_sequence` | `string` | — | `"waste_first"` | `"waste_first"` or `"media_first"`. |
| `max_cycles` | `integer` | `cycles` | `0` | Auto-stop limit (`18` for 3 full periods of 4 hours). |

---

## 4. Sample CLI Commands

### 🌟 Recommended: 3 Periods $\times$ 4 Hours (12 Hours Total, 40 Min Apart)
Executes the optimal `0.05 -> 0.10 -> 0.20 -> 0.25 -> 0.20 -> 0.10 mL` sine wave every 40 minutes for 18 cycles (12 hours total) and automatically stops:

```bash
pio run dosing_automation \
    --automation-name manual_dosing_control \
    --duration 40 \
    --volume-sequence "0.05, 0.10, 0.20, 0.25, 0.20, 0.10" \
    --waste-ml 0.15 \
    --pump-sequence waste_first \
    --max-cycles 18
```

### Negative Solvent (Sterile Water) Control
Load **sterile water** into the `media` pump bottle and execute the exact same command to benchmark dilution and refractive index shifts:

```bash
pio run dosing_automation \
    --automation-name manual_dosing_control \
    --duration 40 \
    --volume-sequence "0.05, 0.10, 0.20, 0.25, 0.20, 0.10" \
    --waste-ml 0.15 \
    --pump-sequence waste_first \
    --max-cycles 18
```
