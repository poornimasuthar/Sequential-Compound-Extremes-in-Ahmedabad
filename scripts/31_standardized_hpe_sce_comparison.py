#!/usr/bin/env python3
"""
Standardized Comparison of HPE vs SCE-30
Compares exposure days, burden, and mortality density on equal footing
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
print("STANDARDIZED HPE vs SCE COMPARISON")
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

pm90 = pm25.quantile(0.90)
t90 = t2m.quantile(0.90)

pm_extreme = pm25 > pm90
t_extreme = t2m > t90
hpe_mask = pm_extreme & t_extreme

# HPE metrics
hpe_days = hpe_mask.sum()
hpe_pm_burden = (pm25[hpe_mask] - pm90).sum()
hpe_t_burden = (t2m[hpe_mask] - t90).sum()

# SCE-30 metrics (window = 30)
w = 30
pm_dates = pm25[pm_extreme].index
heat_dates = t2m[t_extreme].index

# PM episodes
episodes = []
if len(pm_dates) > 0:
    start = prev = pm_dates[0]
    for d in pm_dates[1:]:
        if d == prev + pd.Timedelta(days=1):
            prev = d
        else:
            episodes.append((start, prev)); start = prev = d
    episodes.append((start, prev))

sce_events = 0
sce_pm_days = 0
sce_heat_day_set = set()
sce_pm_burden = 0.0
sce_t_burden = 0.0

for start, end in episodes:
    window_end = end + pd.Timedelta(days=w)
    heat_in_window = heat_dates[(heat_dates > end) & (heat_dates <= window_end)]
    if len(heat_in_window) > 0:
        sce_events += 1
        n_days = (end - start).days + 1
        sce_pm_days += n_days
        sce_pm_burden += (pm25.loc[start:end] - pm90).sum()
        for hd in heat_in_window:
            if hd not in sce_heat_day_set:
                sce_heat_day_set.add(hd)
                sce_t_burden += (t2m.loc[hd] - t90)

sce_heat_days = len(sce_heat_day_set)
sce_total_days = sce_pm_days + sce_heat_days

# Population
POP = 8_450_000

# Standardized table
comparison = pd.DataFrame([
    {
        'framework': 'HPE (Simultaneous)',
        'exposure_days': int(hpe_days),
        'person_exposure_days': int(hpe_days) * POP,
        'pm_burden_ug_m3_days': float(hpe_pm_burden),
        'heat_burden_K_days': float(hpe_t_burden),
        'events': int(hpe_days),
        'burden_per_event': float(hpe_pm_burden + hpe_t_burden*10) / max(int(hpe_days),1)  # rough combined metric
    },
    {
        'framework': 'SCE-30 (Sequential)',
        'exposure_days': sce_total_days,
        'person_exposure_days': sce_total_days * POP,
        'pm_burden_ug_m3_days': float(sce_pm_burden),
        'heat_burden_K_days': float(sce_t_burden),
        'events': sce_events,
        'burden_per_event': float(sce_pm_burden + sce_t_burden*10) / max(sce_events,1)
    }
])

comparison['ratio_to_hpe'] = comparison['exposure_days'] / max(comparison.iloc[0]['exposure_days'], 1)

print("\n" + comparison.to_string(index=False))
comparison.to_csv(OUTPUT_DIR / "standardized_hpe_sce_comparison.csv", index=False)
print(f"\nSaved: standardized_hpe_sce_comparison.csv")

print("""
INTERPRETATION:
- HPE counts days where BOTH thresholds are crossed simultaneously.
- SCE-30 counts ALL days that are part of a sequential extreme episode
  (either as a PM extreme day or as the following heat extreme day).
- The '5×' claim in the original abstract compared 3 HPE days vs 15 SCE events.
  This was invalid because 'days' ≠ 'events'.
- The standardized comparison shows the ratio of TOTAL EXPOSURE DAYS.
  If SCE-30 captures 5× more exposure days than HPE, THAT is a valid claim.
- Replace '5-fold underestimation' with:
  'SCE-30 captures [X]× more high-burden exposure days than HPE'
  OR
  'SCE-30 identifies [X]× more extreme episodes than HPE captures co-occurrence days'
""")

cams.close(); era5.close()
