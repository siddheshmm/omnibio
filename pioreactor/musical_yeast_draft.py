import numpy as np, pandas as pd, json
from scipy.io.wavfile import write as wav_write
from scipy.signal import butter, lfilter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings('ignore')

BASE  = "24th april/"
CYCLE = 5  # minutes per cycle
SR    = 44100  # audio sample rate
BEAT  = 1.5    # seconds per cycle in audio (tempo)

# ── Load Phase 2 data (the good part of the 72-cycle experiment) ──────────────
spec    = pd.read_csv(BASE+"as7341_spectrum_readings-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])
dosing  = pd.read_csv(BASE+"dosing_events-Demo_experiment-all_units-20260424214253.csv",            parse_dates=["timestamp"])
growth  = pd.read_csv(BASE+"growth_rates-Demo_experiment-all_units-20260424214253.csv",             parse_dates=["timestamp"])
od_filt = pd.read_csv(BASE+"od_readings_filtered-Demo_experiment-all_units-20260424214253.csv",     parse_dates=["timestamp"])
od      = pd.read_csv(BASE+"od_readings-Demo_experiment-all_units-20260424214253.csv",              parse_dates=["timestamp"])

plugin_add = dosing[dosing['event']=='add_media'].copy()
plugin_add['cycle'] = plugin_add['timestamp'].dt.floor(f'{CYCLE}min')
sine_input = plugin_add.groupby('cycle')['volume_change_ml'].sum().rename('media_ml')
sine_input.index.name = 'timestamp'

od_wide   = od.pivot_table(index='timestamp',columns='angle',values='od_reading').resample(f'{CYCLE}min').mean()
spec_wide = spec.pivot_table(index='timestamp',columns='band',values='reading').resample(f'{CYCLE}min').mean()
od_wide.columns   = [f'OD_{c}' for c in od_wide.columns]
spec_wide.columns = [f'nm_{c}' for c in spec_wide.columns]
gr_res  = growth.set_index('timestamp')['rate'].resample(f'{CYCLE}min').mean()
nod_res = od_filt.set_index('timestamp')['normalized_od_reading'].resample(f'{CYCLE}min').mean()

idx = (sine_input.index.intersection(od_wide.index).intersection(spec_wide.index)
                       .intersection(gr_res.index).intersection(nod_res.index))
u  = sine_input.loc[idx].values
X  = np.hstack([od_wide.loc[idx].values, spec_wide.loc[idx].values,
                gr_res.loc[idx].values.reshape(-1,1),
                nod_res.loc[idx].values.reshape(-1,1)])
sensor_names = list(od_wide.columns)+list(spec_wide.columns)+['growth_rate','norm_od']
T, N = X.shape

# Phase 2 only (adapted culture)
SPLIT = 36
X2   = X[SPLIT:];  u2 = u[SPLIT:]
T2   = len(u2)
t2   = np.arange(T2)*CYCLE

print(f"Phase 2 reservoir: {T2} cycles × {CYCLE} min")
print(f"Sensors: {sensor_names}")

# ── Normalise each sensor to [0,1] ────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X2_norm = scaler.fit_transform(X2)   # shape: (T2, 13)

# ── Musical mapping design ────────────────────────────────────────────────────
# We assign each sensor a musical role based on its biological meaning:
#
#  RHYTHM voices (fast-responding sensors → trigger drum-like hits)
#    OD_90     → kick drum (low thud — cell density pulse)
#    OD_45     → snare (mid attack — forward scatter)
#    nm_680    → hi-hat (cytochrome oxidase — aerobic rhythm marker)
#
#  MELODY voices (slow biological signals → melodic lines)
#    growth_rate → lead melody (most important, drives pitch)
#    norm_od     → bass line (normalised culture density)
#    nm_445      → mid melody (NADH cycling — metabolic melody)
#
#  HARMONY voices (spectrometer bands → chord pads)
#    nm_480, nm_515, nm_555 → chord tones (flavin harmonics)
#    nm_415                  → high harmony (UV edge)
#    nm_590, nm_630          → warm harmony (longer wavelengths)
#
#  TEXTURE voice
#    OD_135     → reverb/ambience (backscatter — diffuse environment)

# Pentatonic scale in C minor (sounds good regardless of what the yeast does)
# Frequencies (Hz) for C minor pentatonic: C3-Eb3-F3-G3-Bb3-C4-Eb4-F4-G4-Bb4-C5
PENTATONIC = [
    130.81, 155.56, 174.61, 196.00, 233.08,
    261.63, 311.13, 349.23, 392.00, 466.16,
    523.25, 622.25, 698.46
]

def sensor_to_note_freq(val_norm, scale=PENTATONIC):
    """Map normalised sensor value [0,1] to a note frequency."""
    idx = int(val_norm * (len(scale)-1))
    idx = max(0, min(idx, len(scale)-1))
    return scale[idx]

def generate_sine_tone(freq, duration, amp=0.3, sr=SR):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    return amp * np.sin(2*np.pi*freq*t)

def generate_rich_tone(freq, duration, amp=0.3, sr=SR, harmonics=3):
    """Organ-like tone with harmonics."""
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    wave = np.zeros_like(t)
    for h in range(1, harmonics+1):
        wave += (amp/h) * np.sin(2*np.pi*freq*h*t)
    return wave

def generate_pad_tone(freq, duration, amp=0.2, sr=SR):
    """Soft pad with slight detuning for warmth."""
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    wave = (amp * np.sin(2*np.pi*freq*t) +
            amp*0.7 * np.sin(2*np.pi*freq*1.003*t) +   # slight detune
            amp*0.4 * np.sin(2*np.pi*freq*2*t))         # octave above
    return wave

def generate_pluck(freq, duration, amp=0.5, sr=SR):
    """Karplus-Strong-like pluck for OD rhythmic hits."""
    samples = int(sr*duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    # Start with a burst, decay quickly
    env = np.exp(-t * 8)
    return amp * env * np.sin(2*np.pi*freq*t)

def apply_fade(wave, fade_sec=0.02, sr=SR):
    fn = int(sr*fade_sec)
    if len(wave) > 2*fn:
        wave[:fn]  *= np.linspace(0,1,fn)
        wave[-fn:] *= np.linspace(1,0,fn)
    return wave

def lowpass(wave, cutoff=2000, sr=SR):
    b, a = butter(2, cutoff/(sr/2), btype='low')
    return lfilter(b, a, wave)

# ── Sonification engine ───────────────────────────────────────────────────────
print("\nGenerating yeast music...")
print(f"  {T2} cycles × {BEAT}s/cycle = {T2*BEAT:.1f}s total audio")

# Sensor index lookup
si = {name: i for i, name in enumerate(sensor_names)}

# Build audio track — mix all voices
track_len = int(SR * T2 * BEAT)
track     = np.zeros(track_len)

# Also build individual stems for analysis
stems = {name: np.zeros(track_len) for name in sensor_names}

for cycle_idx in range(T2):
    start = int(cycle_idx * BEAT * SR)
    end   = int((cycle_idx+1) * BEAT * SR)
    dur   = BEAT

    vals = X2_norm[cycle_idx]  # all 13 normalised sensor values

    # ── MELODY: growth_rate → lead melody (rich organ tone) ──────────────────
    gr_val  = vals[si['growth_rate']]
    gr_freq = sensor_to_note_freq(gr_val)
    melody  = generate_rich_tone(gr_freq, dur, amp=0.35, harmonics=4)
    melody  = apply_fade(melody)
    track[start:end]             += melody
    stems['growth_rate'][start:end] += melody

    # ── BASS: norm_od → bass line (low sine tone) ─────────────────────────────
    nod_val  = vals[si['norm_od']]
    nod_freq = sensor_to_note_freq(nod_val, PENTATONIC[:6]) / 2  # octave down
    bass     = generate_sine_tone(nod_freq, dur, amp=0.40)
    bass     = lowpass(bass, 400)
    bass     = apply_fade(bass)
    track[start:end]          += bass
    stems['norm_od'][start:end] += bass

    # ── MID MELODY: nm_445 → NADH melody (soft pad) ──────────────────────────
    nm445_freq = sensor_to_note_freq(vals[si['nm_445']])
    nadh_pad   = generate_pad_tone(nm445_freq, dur, amp=0.22)
    nadh_pad   = apply_fade(nadh_pad)
    track[start:end]          += nadh_pad
    stems['nm_445'][start:end] += nadh_pad

    # ── HARMONY TRIAD: nm_480, nm_515, nm_555 → flavin chord pad ─────────────
    for nm_key, vol_scale in [('nm_480',0.15),('nm_515',0.13),('nm_555',0.12)]:
        f = sensor_to_note_freq(vals[si[nm_key]])
        pad = generate_pad_tone(f, dur, amp=vol_scale)
        pad = apply_fade(pad)
        track[start:end]        += pad
        stems[nm_key][start:end] += pad

    # ── HIGH HARMONY: nm_415 → UV sparkle (high, soft) ───────────────────────
    uv_freq = sensor_to_note_freq(vals[si['nm_415']], PENTATONIC[6:])
    uv_tone = generate_sine_tone(uv_freq, dur, amp=0.10)
    uv_tone = apply_fade(uv_tone)
    track[start:end]          += uv_tone
    stems['nm_415'][start:end] += uv_tone

    # ── WARM HARMONY: nm_590, nm_630 → carotenoid warmth ─────────────────────
    for nm_key, vol_scale in [('nm_590',0.14),('nm_630',0.13)]:
        f = sensor_to_note_freq(vals[si[nm_key]], PENTATONIC[2:9])
        pad = generate_pad_tone(f, dur, amp=vol_scale)
        pad = apply_fade(pad)
        track[start:end]        += pad
        stems[nm_key][start:end] += pad

    # ── RHYTHM: OD_90 → kick (low pluck on high values) ─────────────────────
    if vals[si['OD_90']] > 0.5:  # only trigger when density high
        kick = generate_pluck(65.41, min(dur, 0.4), amp=vals[si['OD_90']]*0.6)
        kick_len = len(kick)
        if start + kick_len <= track_len:
            track[start:start+kick_len]            += kick
            stems['OD_90'][start:start+kick_len]   += kick

    # ── RHYTHM: nm_680 → hi-hat on cytochrome pulse ──────────────────────────
    if vals[si['nm_680']] > 0.4:
        hh = generate_pluck(880, min(dur, 0.15), amp=vals[si['nm_680']]*0.3)
        hh_len = len(hh)
        if start + hh_len <= track_len:
            track[start:start+hh_len]            += hh
            stems['nm_680'][start:start+hh_len]  += hh

    # ── TEXTURE: OD_135 → ambient shimmer ────────────────────────────────────
    tex_freq = sensor_to_note_freq(vals[si['OD_135']], PENTATONIC[8:])
    shimmer  = generate_sine_tone(tex_freq*2, dur, amp=0.08)
    shimmer  = apply_fade(shimmer)
    track[start:end]           += shimmer
    stems['OD_135'][start:end]  += shimmer

# ── Normalise and export ──────────────────────────────────────────────────────
def norm_audio(a, peak=0.88):
    mx = np.max(np.abs(a))
    return (a/mx*peak).astype(np.float32) if mx > 0 else a.astype(np.float32)

track_norm = norm_audio(track)

# Full mix
wav_write("output musical yeast/yeast_music_full.wav", SR, track_norm)

# Melody stem only (growth rate)
wav_write("output musical yeast/yeast_melody_stem.wav", SR, norm_audio(stems['growth_rate']))

# Harmonic stem (spectrometer bands)
spec_stem = sum(stems[f'nm_{b}'] for b in [415,445,480,515,555,590,630,680])
wav_write("output musical yeast/yeast_harmony_stem.wav", SR, norm_audio(spec_stem))

print(f"  Full mix:     yeast_music_full.wav     ({T2*BEAT:.1f}s)")
print(f"  Melody stem:  yeast_melody_stem.wav")
print(f"  Harmony stem: yeast_harmony_stem.wav")

# ── Save metadata for plotting ────────────────────────────────────────────────
np.savez("output musical yeast/yeast_music_data.npz",
    X2_norm=X2_norm, u2=u2, t2=t2, sensor_names=np.array(sensor_names))
print("Data saved.")