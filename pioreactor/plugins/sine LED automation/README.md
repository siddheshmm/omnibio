# Sine LED Automation Plugin

**Plugin Name:** `sine_led_automation`  
**Version:** `v1.1.0`  
**Location:** `pioreactor/plugins/sine LED automation/`

---

## 1. Overview

`sine_led_automation` is a custom Pioreactor LED automation plugin that steps through a user-defined sequence of LED intensity percentages at fixed time intervals. 

It is designed to encode smooth **optical sine waves** and **photostimulation / UV-A perturbations** into the bioreactor culture for Biophysical Reservoir Computing (BRC) and dynamic light-response experiments.

### Key Features:
- **Web UI & CLI Compatibility**: Seamlessly launchable from both the Pioreactor Web UI modal and the terminal.
- **Flexible Channel Mapping**: Supports numbers (`1=A, 2=B, 3=C, 4=D`) and letters (`'A', 'B', 'C', 'D'`).
- **Default 3-Hour Sine Wave**: Automatically runs `0% -> 10% -> 25% -> 35% -> 25% -> 10% -> 0%` (30 minutes per step) when started from UI defaults.
- **Dedicated 0% (OFF) Dark Stage**: Enables observation of baseline dark recovery and unperturbed autofluorescence.
- **SQLite Logging**: Automatically logs all intensity transitions to a dedicated table (`sine_led_intensities`).
- **Live MQTT Telemetry**: Publishes `current_intensity` and `current_step` to MQTT for live Web UI visualization.
- **Fail-Safe Cleanup**: Automatically forces the LED to `0.0%` upon pause, sleep, or disconnection.

---

## 2. Web UI Configuration Guide

When starting from the Pioreactor Web UI (`pioreactor01` $\to$ **LED Automation** $\to$ **Sine LED Automation**):

| Field | Default Value | Notes |
| :--- | :---: | :--- |
| **LED Channel** | `2` | Enter `2` for Channel B (standard UV LED), `1` for A, `3` for C, or `4` for D. |
| **Hold duration per step** | `30` min | Duration in minutes to hold each intensity level. |
| **Intensity sequence** | *(default)* | Leave default to automatically execute the 3-hour sine: `0, 10, 25, 35, 25, 10, 0`. |
| **Repeat full cycle count** | `0` | Set `0` to loop indefinitely, or specify an integer count. |

> **Note:** Simply clicking **START** with the default UI values will immediately run the 3-hour UV sine wave on Channel B.

---

## 3. CLI Usage & Advanced Customization

For full personalization of custom sequence strings or channels:

### Example 1: Standard 3-Hour UV-A Sine Wave on Channel B
* **Sequence**: `0% -> 10% -> 25% -> 35% -> 25% -> 10% -> 0%`
* **Hold time**: 30 minutes per step (Total 6-step cycle = **3.0 hours**).

```bash
pio run led_automation \
    --automation-name sine_led_automation \
    --led-channel B \
    --intensity-sequence "0, 10, 25, 35, 25, 10, 0" \
    --minutes-per-step 30
```

### Example 2: 4-Hour Slower Sine Wave on Channel C (4 Repetitions)
* **Hold time**: 40 minutes per step (Total 6-step cycle = **4.0 hours**).
* Stops automatically after **4 full repetitions (16 hours total)**.

```bash
pio run led_automation \
    --automation-name sine_led_automation \
    --led-channel C \
    --intensity-sequence "0, 10, 25, 35, 25, 10, 0" \
    --minutes-per-step 40 \
    --repeat-cycles 4
```

---

## 4. SQLite Database Schema

On every step transition, the active intensity is recorded in the Pioreactor SQLite database:

```sql
CREATE TABLE IF NOT EXISTS sine_led_intensities (
    experiment       TEXT NOT NULL,
    pioreactor_unit  TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    led_channel      TEXT NOT NULL,
    intensity_pct    REAL NOT NULL
);
```
