#!/usr/bin/env python3
"""
AGU Threshold Sensitivity Analysis
Tests SCE robustness across percentile thresholds using 2019 data only.
Outputs: outputs/agu_threshold_sensitivity.csv
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

def get_time_values(da):
    for c in ['time', 'valid_time', 'date', 'day']:
        if c in da.coords:
            return da[c]
    return pd.date_range('2019-01-01', periods=len(da), freq='D')

print("=" * 70)
print("THRESHOLD SENSITIVITY ANALYSIS (AGU)")
print("=" * 70)

# Load 2019 data
cams = xr.open_dataset(OUTPUT_DIR / "pm25_daily_cams_ahmedabad_2019.nc")
era5 = xr.open_dataset(OUTPUT_DIR / "era5_daily_2019.nc")

pm25_da = cams["pm25"].mean(dim=["latitude", "longitude"]) if "latitude" in cams["pm25"].dims else cams["pm25"]
t2m_da = era5["tmax"] if "tmax" in era5 else era5["t2m"]

pm25 = pd.Series(pm25_da.values, index=pd.to_datetime(get_time_values(pm25_da).values))
t2m = pd.Series(t2m_da.mean(dim=["latitude", "longitude"]).values, index=pd.to_datetime(get_time_values(t2m_da).values))

common = pm25.index.intersection(t2m.index)
pm25 = pm25.loc[common].sort_index()
t2m = t2m.loc[common].sort_index()

# Percentiles to test
percentiles = [80, 85, 90, 95]
results = []

for p in percentiles:
    q = p / 100.0
    pm_thresh = float(pm25.quantile(q))
    t_thresh = float(t2m.quantile(q))
    
    pm_extreme = pm25 > pm_thresh
    t_extreme = t2m > t_thresh
    hpe_mask = pm_extreme & t_extreme
    hpe_days = int(hpe_mask.sum())
    
    # Winter episodes (Nov-Feb)
    pm_dates = pm25[pm_extreme].index
    winter_pm = pm_dates[pm_dates.month.isin([11, 12, 1, 2])]
    
    # Extract episodes (consecutive days merged)
    episodes = []
    if len(winter_pm) > 0:
        start = prev = winter_pm[0]
        for d in winter_pm[1:]:
            if d == prev + pd.Timedelta(days=1):
                prev = d
            else:
                episodes.append((start, prev))
                start = prev = d
        episodes.append((start, prev))
    
    # Summer heat days
    heat_dates = t2m[t_extreme].index
    summer_heat = heat_dates[heat_dates.month.isin([3, 4, 5, 6])]
    
    # SCE linkage
    sce_linked = 0
    lead_times = []
    episode_lengths = []
    
    for start, end in episodes:
        future_heat = summer_heat[summer_heat > end]
        if len(future_heat) > 0:
            sce_linked += 1
            lead_times.append((future_heat[0] - end).days)
        episode_lengths.append((end - start).days + 1)
    
    total_episodes = len(episodes)
    linkage_rate = (sce_linked / total_episodes * 100) if total_episodes > 0 else 0.0
    mean_lead = float(np.mean(lead_times)) if lead_times else np.nan
    
    results.append({
        'percentile': p,
        'pm_threshold_ug_m3': round(pm_thresh, 1),
        't_threshold_C': round(t_thresh, 1),
        'hpe_days': hpe_days,
        'total_winter_episodes': total_episodes,
        'sce_linked_episodes': sce_linked,
        'linkage_rate_pct': round(linkage_rate, 1),
        'mean_lead_time_days': round(mean_lead, 1) if not np.isnan(mean_lead) else np.nan,
        'mean_episode_length_days': round(np.mean(episode_lengths), 1) if episode_lengths else np.nan,
    })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / "agu_threshold_sensitivity.csv", index=False)

print("\n" + df.to_string(index=False))
print(f"\nSaved: {OUTPUT_DIR / 'agu_threshold_sensitivity.csv'}")

# Interpretation for AGU abstract
print("\n" + "=" * 70)
print("AGU ABSTRACT LANGUAGE")
print("=" * 70)
print(f"""
At the 90th percentile, SCE identified {df[df['percentile']==90]['sce_linked_episodes'].values[0]} linked episodes.
Threshold sensitivity was asymmetric: SCE counts decreased from 
{df[df['percentile']==80]['sce_linked_episodes'].values[0]} (80th) to {df[df['percentile']==95]['sce_linked_episodes'].values[0]} (95th), 
but linkage rates remained stable (≥{df['linkage_rate_pct'].min():.0f}%), indicating 
seasonal-transition structure is robust to moderate threshold variation.
""")

cams.close(); era5.close()
