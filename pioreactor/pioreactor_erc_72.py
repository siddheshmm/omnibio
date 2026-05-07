import pandas as pd, numpy as np, pickle
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

BASE   = "24th april/"
ALPHAS = [0.001,0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0,100.0]
CYCLE  = 5
SPLIT  = 36

# ── LOAD DATA ─────────────────────────────────────────────
spec    = pd.read_csv(BASE+"as7341_spectrum_readings-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])
dosing  = pd.read_csv(BASE+"dosing_events-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])
growth  = pd.read_csv(BASE+"growth_rates-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])
od_filt = pd.read_csv(BASE+"od_readings_filtered-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])
od      = pd.read_csv(BASE+"od_readings-Demo_experiment-all_units-20260424214253.csv", parse_dates=["timestamp"])

# ── BUILD INPUT SIGNAL ─────────────────────────────────────
plugin_add = dosing[dosing['event']=='add_media'].copy()
plugin_add['cycle'] = plugin_add['timestamp'].dt.floor(f'{CYCLE}min')
sine_input = plugin_add.groupby('cycle')['volume_change_ml'].sum().rename('media_ml')
sine_input.index.name = 'timestamp'

# ── BUILD RESERVOIR STATE ─────────────────────────────────
od_wide   = od.pivot_table(index='timestamp', columns='angle', values='od_reading').resample(f'{CYCLE}min').mean()
spec_wide = spec.pivot_table(index='timestamp', columns='band', values='reading').resample(f'{CYCLE}min').mean()

od_wide.columns   = [f'OD_{c}' for c in od_wide.columns]
spec_wide.columns = [f'nm_{c}' for c in spec_wide.columns]

gr_res  = growth.set_index('timestamp')['rate'].resample(f'{CYCLE}min').mean()
nod_res = od_filt.set_index('timestamp')['normalized_od_reading'].resample(f'{CYCLE}min').mean()

idx = (sine_input.index
       .intersection(od_wide.index)
       .intersection(spec_wide.index)
       .intersection(gr_res.index)
       .intersection(nod_res.index))

u = sine_input.loc[idx].values

X = np.hstack([
    od_wide.loc[idx].values,
    spec_wide.loc[idx].values,
    gr_res.loc[idx].values.reshape(-1,1),
    nod_res.loc[idx].values.reshape(-1,1)
])

sensor_names = list(od_wide.columns) + list(spec_wide.columns) + ['growth_rate','norm_od']

print(f"Loaded: {X.shape[0]} cycles × {X.shape[1]} features")

# ── RIDGE FUNCTION ─────────────────────────────────────────
def ridge_loo(X, y):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    rcv = RidgeCV(alphas=ALPHAS, cv=None)
    rcv.fit(Xs, y)

    model = Ridge(alpha=rcv.alpha_)
    yp = cross_val_predict(model, Xs, y, cv=LeaveOneOut())

    nmse = mean_squared_error(y, yp) / np.var(y)

    model.fit(Xs, y)

    return yp, nmse, model, sc

# ── PHASE 2 DATA ──────────────────────────────────────────
X2 = X[SPLIT:]
u2 = u[SPLIT:]

print(f"Phase 2: {len(u2)} cycles")

# ── TRAIN MODEL (NO EMBEDDING FOR NOW) ─────────────────────
yp, nmse, model, scaler = ridge_loo(X2, u2)

print(f"\nPhase 2 NMSE: {nmse:.4f}")

# ── SAVE MODEL ─────────────────────────────────────────────
model_package = {
    "model": model,
    "scaler": scaler,
    "sensor_names": sensor_names,
    "note": "delay=0 model (no embedding)"
}

with open("ridge_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("\n✅ Model saved: ridge_model.pkl")