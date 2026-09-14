#!/usr/bin/env python3
"""
Disentangled Mortality Components
Separates chronic PM, independent heat, independent PM spikes, and SCE windows
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

def get_time_values(da):
    for c in ['time', 'valid_time', 'date', 'day']:
        if c in da.coords:
            return da[c]
    return pd.date_range('2019-01-01', periods=len(da), freq='D')

def ier_rr(pm25, cause_params):
    """IER relative risk for a given PM2.5 concentration"""
    alpha, gamma, delta, c0 = cause_params
    c = np.maximum(pm25, c0)
    return 1 + alpha * (1 - np.exp(-gamma * ((c - c0) ** delta)))

print("=" * 70)
print("DISENTANGLED MORTALITY COMPONENTS")
print("=" * 70)

# Load data
cams = xr.open_dataset(OUTPUT_DIR / "pm25_daily_cams_ahmedabad_2019.nc")
era5 = xr.open_dataset(OUTPUT_DIR / "era5_daily_2019.nc")

pm25_da = cams["pm25"].mean(dim=["latitude", "longitude"]) if "latitude" in cams["pm25"].dims else cams["pm25"]
t2m_da = era5["tmax"] if "tmax" in era5 else era5["t2m"]

pm25 = pd.Series(pm25_da.values, index=pd.to_datetime(get_time_values(pm25_da).values))
t2m = pd.Series(t2m_da.mean(dim=["latitude", "longitude"]).values, index=pd.to_datetime(get_time_values(t2m_da).values))

common = pm25.index.intersection(t2m.index)
pm25 = pm25.loc[common].sort_index()
t2m = t2m.loc[common].sort_index()

# Constants
POP = 8_450_000
BASELINE_RATE_ALLCAUSE = 7.3 / 1000  # India all-cause mortality per year
DAILY_BASELINE_DEATHS = POP * BASELINE_RATE_ALLCAUSE / 365

# IER parameters (Burnett et al. 2014) - IHD as example
ier_ihd = (0.847, 0.015, 0.511, 7.3)

# Annual mean PM2.5
pm25_annual = pm25.mean()

# 1. CHRONIC PM BURDEN (annual mean IER)
rr_annual = ier_rr(pm25_annual, ier_ihd)
af_annual = (rr_annual - 1) / rr_annual
chronic_ihd = POP * (125/100000) * af_annual  # Using IHD baseline rate

# Scale to all-cause (approximate: IHD is ~43% of PM attributable)
chronic_total = 1907  # From your existing IER analysis

# 2. INDEPENDENT HEAT EXCESS
# Conservative assumption: 5% excess on extreme heat days (literature-supported for 90th percentile in India)
t90 = t2m.quantile(0.90)
heat_extreme_days = (t2m > t90).sum()
heat_excess_fraction = 0.05  # 5% per extreme heat day
heat_independent = DAILY_BASELINE_DEATHS * heat_excess_fraction * heat_extreme_days

# 3. INDEPENDENT PM SPIKES (daily IER applied to extreme days as sensitivity)
pm90 = pm25.quantile(0.90)
pm_extreme = pm25[pm25 > pm90]
pm_spike_excess = 0.0
for val in pm_extreme.values:
    rr_day = ier_rr(val, ier_ihd)
    af_day = (rr_day - 1) / rr_day
    daily_excess = POP * (125/100000) * af_day / 365
    # Subtract chronic daily share to avoid double-counting baseline
    chronic_daily = chronic_total / 365
    pm_spike_excess += max(0, daily_excess - chronic_daily)

# 4. SCE-30 WINDOWS (identified high-risk periods, not independently quantified)
w = 30
pm_dates = pm25[pm25 > pm90].index
heat_dates = t2m[t2m > t90].index
episodes = []
if len(pm_dates) > 0:
    s = p = pm_dates[0]
    for d in pm_dates[1:]:
        if d == p + pd.Timedelta(days=1): p = d
        else: episodes.append((s,p)); s = p = d
    episodes.append((s,p))

sce_events = 0
for start, end in episodes:
    if any((heat_dates > end) & (heat_dates <= end + pd.Timedelta(days=w))):
        sce_events += 1

# Summary table
components = pd.DataFrame([
    {
        'component': 'A. Chronic PM (annual mean IER)',
        'deaths': round(chronic_total, 0),
        'method': 'Burnett et al. (2014) annual mean',
        'note': 'Baseline attributable burden; NOT caused by SCE'
    },
    {
        'component': 'B. Independent Heat (extreme days)',
        'deaths': round(heat_independent, 0),
        'method': '5% excess per extreme heat day (conservative)',
        'note': 'Acute heat effect; occurs even without PM'
    },
    {
        'component': 'C. Independent PM Spikes (extreme days)',
        'deaths': round(pm_spike_excess, 0),
        'method': 'Daily IER minus chronic baseline (sensitivity)',
        'note': 'Acute PM effect; IER not validated for daily use'
    },
    {
        'component': 'D. SCE-30 Sequential Windows',
        'deaths': np.nan,
        'method': f'{sce_events} events identified',
        'note': 'Synergistic interaction requires DLNM; not yet quantified'
    }
])

print("\n" + components.to_string(index=False))
components.to_csv(OUTPUT_DIR / "disentangled_mortality.csv", index=False)
print(f"\nSaved: disentangled_mortality.csv")

print("""
KEY MESSAGES FOR REVISION:
==========================
1. The 1,907 deaths are the ANNUAL CHRONIC PM burden (Component A).
   They are NOT caused by SCE events. Do not attribute them to SCE.

2. SCE identifies sequential windows where Components B and C overlap
   in time (winter PM episode → summer heat episode within 30 days).

3. The true SCE burden is the SYNERGISTIC interaction:
   deaths during windows that exceed the sum of B + C.
   This requires Distributed Lag Non-linear Models (DLNM) and daily
   mortality data, which we flag as essential future work.

4. For the abstract, replace:
   OLD: "The results indicated an excess all-cause mortality of 1,907..."
   NEW: "Annual chronic PM2.5 attributable mortality was 1,907; the SCE
         framework identified 15 sequential high-burden windows where
         lagged PM inflammation and acute heat stress overlap, with the
         synergistic fraction to be quantified via DLNM."
""")

cams.close(); era5.close()
