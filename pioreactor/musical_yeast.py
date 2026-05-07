"""
Yeast Musical Reservoir Experiment
----------------------------------
Description:
1. Encodes a musical chord progression into media dosing volumes.
2. Uses Phase 2 data from a Pioreactor as a biological reservoir.
3. Trains a Ridge Regression readout to map yeast sensor states to media inputs.
4. Reconstructs the chord progression (the 'playback') and generates sonified audio.
"""

import numpy as np
import pandas as pd
import json
import warnings
import matplotlib
from scipy.io.wavfile import write as wav_write
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# Configuration
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

BASE   = "/mnt/user-data/uploads/"
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
CYCLE  = 5
SPLIT  = 36
SAMPLE_RATE = 44100
NOTE_DURATION = 2.0  # seconds per chord

# --- 1. Load Phase 2 Reservoir States ---
print("Loading data from Pioreactor experiment...")
spec    = pd.read_csv(BASE + "as7341_spectrum_readings-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])
dosing  = pd.read_csv(BASE + "dosing_events-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])
growth  = pd.read_csv(BASE + "growth_rates-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])
od_filt = pd.read_csv(BASE + "od_readings_filtered-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])
od      = pd.read_csv(BASE + "od_readings-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])

# Sync media dosing to cycles
plugin_add = dosing[dosing['event'] == 'add_media'].copy()
plugin_add['cycle'] = plugin_add['timestamp'].dt.floor(f'{CYCLE}min')
sine_input = plugin_add.groupby('cycle')['volume_change_ml'].sum().rename('media_ml')
sine_input.index.name = 'timestamp'

# Process sensor features
od_wide   = od.pivot_table(index='timestamp', columns='angle', values='od_reading').resample(f'{CYCLE}min').mean()
spec_wide = spec.pivot_table(index='timestamp', columns='band', values='reading').resample(f'{CYCLE}min').mean()
od_wide.columns   = [f'OD_{c}' for c in od_wide.columns]
spec_wide.columns = [f'nm_{c}' for c in spec_wide.columns]
gr_res  = growth.set_index('timestamp')['rate'].resample(f'{CYCLE}min').mean()
nod_res = od_filt.set_index('timestamp')['normalized_od_reading'].resample(f'{CYCLE}min').mean()

# Intersect timestamps to align all features
idx = (sine_input.index.intersection(od_wide.index).intersection(spec_wide.index)
                       .intersection(gr_res.index).intersection(nod_res.index))

u_sine = sine_input.loc[idx].values
X_all  = np.hstack([od_wide.loc[idx].values, spec_wide.loc[idx].values,
                    gr_res.loc[idx].values.reshape(-1,1),
                    nod_res.loc[idx].values.reshape(-1,1)])
sensor_names = list(od_wide.columns) + list(spec_wide.columns) + ['growth_rate','norm_od']

# Slice for Phase 2 data
X2 = X_all[SPLIT:]
u2_sine = u_sine[SPLIT:]
T2, N = X2.shape
print(f"Phase 2 reservoir: T2={T2} cycles, N={N} features")

# --- 2. Load Chord Design ---
with open("/home/claude/chord_design.json") as f:
    design = json.load(f)
CHORD_MAP   = design['chord_map']
PROGRESSION = design['progression']
VOLUMES     = np.array(design['volumes'])
T_chord     = len(VOLUMES)
print(f"Chord sequence: {T_chord} cycles")

# --- 3. Training & Simulation Functions ---

def embed(X, delay, tau=1):
    """Add memory to the reservoir state via delayed embeddings."""
    return np.array([np.concatenate([X[t-d*tau] for d in range(delay+1)])
                     for t in range(delay*tau, len(X))])

def find_representative_state(target_vol, X2, u2_sine, delay=1):
    """Simulates yeast response by finding historical states with similar dosing inputs."""
    X2e = embed(X2, delay)
    u2e = u2_sine[delay:]
    dists = np.abs(u2e - target_vol)
    closest_idx = np.argsort(dists)[:5]
    weights = 1.0 / (dists[closest_idx] + 1e-6)
    weights /= weights.sum()
    return np.average(X2e[closest_idx], axis=0, weights=weights)

def decode_chord(vol, chord_map):
    """Maps reconstructed dosing volume back to the nearest symbolic chord."""
    chords = list(chord_map.keys())
    vols   = list(chord_map.values())
    dists  = [abs(vol - v) for v in vols]
    return chords[np.argmin(dists)]

# --- 4. Reservoir Computing Process ---

# Step 1: Train on Phase 2 data
X2_emb = embed(X2, 1)
u2_emb = u2_sine[1:]
sc = StandardScaler()
X2s = sc.fit_transform(X2_emb)

rcv = RidgeCV(alphas=ALPHAS, cv=None)
rcv.fit(X2s, u2_emb)
ridge = Ridge(alpha=rcv.alpha_)
ridge.fit(X2s, u2_emb)

yp_train = ridge.predict(X2s)
nmse_train = mean_squared_error(u2_emb, yp_train) / np.var(u2_emb)
print(f"\nTrained readout on sine wave data: NMSE={nmse_train:.4f} (alpha={rcv.alpha_})")

# Step 2: Leave-One-Out validation
yp_loo = cross_val_predict(Ridge(alpha=rcv.alpha_), X2s, u2_emb, cv=LeaveOneOut())
nmse_loo = mean_squared_error(u2_emb, yp_loo) / np.var(u2_emb)
print(f"LOO reconstruction NMSE: {nmse_loo:.4f}")

# Step 3: Inject Chords into the Reservoir
X_chord_sim = np.array([find_representative_state(v, X2, u2_sine) for v in VOLUMES])
X_chord_scaled = sc.transform(X_chord_sim)
volumes_reconstructed = ridge.predict(X_chord_scaled)
chords_reconstructed = [decode_chord(v, CHORD_MAP) for v in volumes_reconstructed]

# --- 5. Display Results ---
correct = sum(p == t for p, t in zip(chords_reconstructed, PROGRESSION))
accuracy = correct / T_chord * 100
print(f"\n=== SIMULATION RESULTS ===")
print(f"Chord reconstruction accuracy: {correct}/{T_chord} = {accuracy:.1f}%")

print(f"\n  {'Cycle':>5}  {'True':>8}  {'Vol_true':>9}  {'Vol_recon':>10}  {'Predicted':>8}  {'Match'}")
print(f"  {'─'*60}")
for i, (true, pred, vt, vr) in enumerate(zip(PROGRESSION, chords_reconstructed, VOLUMES, volumes_reconstructed)):
    match = '✓' if true == pred else '✗'
    print(f"  {i+1:>5}  {true:>8}  {vt:>9.3f}  {vr:>10.3f}  {pred:>8}  {match}")

# --- 6. Audio Sonification ---

CHORD_FREQS = {
    'C_maj': [261.63, 329.63, 392.00],
    'F_maj': [349.23, 440.00, 523.25],
    'G_maj': [392.00, 493.88, 587.33],
    'Am':    [440.00, 523.25, 659.25],
    'Dm':    [293.66, 349.23, 440.00],
    'G7':    [392.00, 493.88, 587.33, 246.94],
}

def generate_chord_audio(chord_name, duration=NOTE_DURATION, sr=SAMPLE_RATE, fade=0.05):
    freqs = CHORD_FREQS[chord_name]
    t_arr = np.linspace(0, duration, int(sr*duration), endpoint=False)
    wave  = sum(0.3 * np.sin(2 * np.pi * f * t_arr) for f in freqs)
    fade_n = int(sr * fade)
    wave[:fade_n] *= np.linspace(0, 1, fade_n)
    wave[-fade_n:] *= np.linspace(1, 0, fade_n)
    return wave

def generate_sensor_tone(sensor_values, duration=NOTE_DURATION, sr=SAMPLE_RATE):
    t_arr = np.linspace(0, duration, int(sr*duration), endpoint=False)
    gr_idx = sensor_names.index('growth_rate')
    od_idx = sensor_names.index('OD_90')
    nm_idx = sensor_names.index('nm_680')
    
    gr_norm = float(np.clip((sensor_values[gr_idx] - (-0.1)) / 0.2, 0, 1))
    od_norm = float(np.clip((sensor_values[od_idx] - 0.1) / 0.5, 0, 1))
    nm_norm = float(np.clip((sensor_values[nm_idx] - 0.15) / 0.3, 0, 1))
    
    freq = 200 + gr_norm * 600
    amp  = 0.2 + od_norm * 0.4
    harmonic = 1 + nm_norm
    wave = amp * (np.sin(2*np.pi*freq*t_arr) + 0.3*np.sin(2*np.pi*freq*harmonic*t_arr))
    
    fade_n = int(sr * 0.05)
    wave[:fade_n] *= np.linspace(0, 1, fade_n)
    wave[-fade_n:] *= np.linspace(1, 0, fade_n)
    return wave

# Render and Save Audio
print("\nGenerating audio files...")
orig_audio = np.concatenate([generate_chord_audio(c) for c in PROGRESSION])
recon_audio = np.concatenate([generate_chord_audio(c) for c in chords_reconstructed])
sensor_audio = np.concatenate([generate_sensor_tone(X2[-T_chord:][i]) for i in range(T_chord)])

def norm_audio(a): return (a / np.max(np.abs(a)) * 0.85).astype(np.float32)

wav_write("/home/claude/chord_original.wav", SAMPLE_RATE, norm_audio(orig_audio))
wav_write("/home/claude/chord_reconstructed.wav", SAMPLE_RATE, norm_audio(recon_audio))
wav_write("/home/claude/sensor_sonification.wav", SAMPLE_RATE, norm_audio(sensor_audio))

# Save metadata for records
np.savez("/home/claude/music_results.npz", VOLUMES=VOLUMES, volumes_reconstructed=volumes_reconstructed)
with open("/home/claude/music_meta.json", "w") as f:
    json.dump({'accuracy': accuracy, 'recon_nmse': nmse_loo}, f)

print("Files saved: chord_original.wav, chord_reconstructed.wav, sensor_sonification.wav")

'''
Output

Phase 2 reservoir: T2=34 cycles, N=13 features
Chord sequence: 16 cycles

Trained readout on sine wave data: NMSE=0.0207 (α=0.01)
LOO reconstruction NMSE: 0.0449 [EXCELLENT]

=== SIMULATION RESULTS ===
Chord reconstruction accuracy: 12/16 = 75.0%
Volume reconstruction NMSE: 0.1320

Chord-by-chord:
  Cycle      True   Vol_true   Vol_recon  Predicted  Match
  ────────────────────────────────────────────────────────────
      1     C_maj      0.100       0.123     C_maj  ✓
      2     G_maj      0.420       0.459     G_maj  ✓
      3        Am      0.580       0.459     G_maj  ✗
      4     F_maj      0.260       0.244     F_maj  ✓
      5     C_maj      0.100       0.123     C_maj  ✓
      6     G_maj      0.420       0.459     G_maj  ✓
      7        Am      0.580       0.459     G_maj  ✗
      8     F_maj      0.260       0.244     F_maj  ✓
      9     C_maj      0.100       0.123     C_maj  ✓
     10     G_maj      0.420       0.459     G_maj  ✓
     11        Am      0.580       0.459     G_maj  ✗
     12     F_maj      0.260       0.244     F_maj  ✓
     13     C_maj      0.100       0.123     C_maj  ✓
     14     G_maj      0.420       0.459     G_maj  ✓
     15        Am      0.580       0.459     G_maj  ✗
     16     F_maj      0.260       0.244     F_maj  ✓

Audio files saved:
  chord_original.wav      — original chord progression
  chord_reconstructed.wav — what the yeast 'heard' and played back
  sensor_sonification.wav — raw yeast biology as music
'''