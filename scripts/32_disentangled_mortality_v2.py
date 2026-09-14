#!/usr/bin/env python3
"""
Disentangled Mortality Components v2
Uses SCE-Seasonal definition
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

print("=" * 70)
print("DISENTANGLED MORTALITY COMPONENTS v2")
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
BASELINE_RATE_ALLCAUSE = 7.3 / 1000
DAILY_BASELINE_DEATHS = POP * BASELINE_RATE_ALLCAUSE / 365

pm90 = pm25.quantile(0.90)
t90 = t2m.quantile(0.90)
pm_extreme = pm25 > pm90
t_extreme = t2m > t90

pm_dates = pm25[pm_extreme].index
heat_dates = t2m[t_extreme].index

# SCE-Seasonal count
winter_pm = pm_dates[pm_dates.month.isin([11, 12, 1, 2])]
summer_heat = heat_dates[heat_dates.month.isin([3, 4, 5, 6])]
sce_seasonal = 0
for d in winter_pm:
    if any(summer_heat > d):
        sce_seasonal += 1

# Mortality components
heat_extreme_days = int(t_extreme.sum())
heat_independent = DAILY_BASELINE_DEATHS * 0.05 * heat_extreme_days

components = pd.DataFrame([
    {
        'component': 'A. Chronic PM (annual mean IER)',
        'deaths': 1907,
        'method': 'Burnett et al. (2014) annual mean',
        'note': 'Baseline burden; NOT caused by SCE'
    },
    {
        'component': 'B. Independent Heat (extreme days)',
        'deaths': round(heat_independent, 0),
        'method': '5% excess per extreme heat day',
        'note': f'{heat_extreme_days} extreme heat days; acute effect'
    },
    {
        'component': 'C. Independent PM Spikes',
        'deaths': 'N/A',
        'method': 'Requires daily mortality data + DLNM',
        'note': 'IER not validated for daily acute attribution'
    },
    {
        'component': 'D. SCE-Seasonal Sequential',
        'deaths': 'N/A',
        'method': f'{sce_seasonal} events identified',
        'note': 'Synergistic fraction requires DLNM; flagged as future work'
    }
])

print("\n" + components.to_string(index=False))
components.to_csv(OUTPUT_DIR / "disentangled_mortality_v2.csv", index=False)
print(f"\nSaved: disentangled_mortality_v2.csv")

print(f"""
REVISED ABSTRACT LANGUAGE:
==========================
"Annual chronic PM2.5 attributable mortality was 1,907. Conventional HPE
captured zero simultaneous extreme days. A fixed 30-day sequential window
also yielded zero events due to the pronounced seasonal separation in
Ahmedabad. A seasonal-transition definition (winter PM extremes followed by
summer heat extremes) identified {sce_seasonal} SCE events, revealing a
complete blind spot in traditional compound-extreme metrics for
inverse-seasonal cities. The synergistic mortality fraction attributable
to sequential exposure requires distributed lag non-linear models (DLNM)
and daily mortality records, which we identify as essential future work."
""")

cams.close(); era5.close()
