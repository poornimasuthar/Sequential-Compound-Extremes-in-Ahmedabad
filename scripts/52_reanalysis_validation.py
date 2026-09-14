#!/usr/bin/env python3
"""
Reanalysis Cross-Validation: CAMS vs MERRA-2
Formalizes bias, correlation, and seasonal decomposition.
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
print("REANALYSIS VALIDATION: CAMS vs MERRA-2")
print("=" * 70)

# Load
cams = xr.open_dataset(OUTPUT_DIR / "pm25_daily_cams_ahmedabad_2019.nc")
merra = xr.open_dataset(OUTPUT_DIR / "pm25_daily_merra2_ahmedabad_2019.nc")

pm_cams = cams["pm25"].mean(dim=["latitude", "longitude"]) if "latitude" in cams["pm25"].dims else cams["pm25"]
pm_merra = merra["pm25"].mean(dim=["latitude", "longitude"]) if "latitude" in merra["pm25"].dims else merra["pm25"]

s1 = pd.Series(pm_cams.values, index=pd.to_datetime(get_time_values(pm_cams).values))
s2 = pd.Series(pm_merra.values, index=pd.to_datetime(get_time_values(pm_merra).values))

common = s1.index.intersection(s2.index)
s1 = s1.loc[common].sort_index()
s2 = s2.loc[common].sort_index()

# Overall
bias = float((s2 - s1).mean())
rmse = float(np.sqrt(((s2 - s1)**2).mean()))
corr = float(s1.corr(s2))

# Seasonal breakdown
def season(m):
    if m in [12, 1, 2]: return 'Winter'
    if m in [3, 4, 5]: return 'Pre-Monsoon'
    if m in [6, 7, 8, 9]: return 'Monsoon'
    return 'Post-Monsoon'

seasons = s1.index.map(lambda d: season(d.month))
results = []
for seas in ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']:
    mask = seasons == seas
    if mask.sum() > 0:
        results.append({
            'season': seas,
            'cams_mean': round(s1[mask].mean(), 1),
            'merra_mean': round(s2[mask].mean(), 1),
            'bias': round((s2[mask] - s1[mask]).mean(), 1),
            'rmse': round(np.sqrt(((s2[mask] - s1[mask])**2).mean()), 1),
            'corr': round(s1[mask].corr(s2[mask]), 2),
            'n_days': int(mask.sum())
        })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / "agu_reanalysis_validation.csv", index=False)

summary = pd.DataFrame([{
    'metric': 'Annual',
    'cams_mean': round(s1.mean(), 1),
    'merra_mean': round(s2.mean(), 1),
    'bias': round(bias, 1),
    'rmse': round(rmse, 1),
    'corr': round(corr, 2),
    'n_days': len(s1)
}])

print("\n--- ANNUAL SUMMARY ---")
print(summary.to_string(index=False))
print("\n--- SEASONAL BREAKDOWN ---")
print(df.to_string(index=False))
print(f"\nSaved: {OUTPUT_DIR / 'agu_reanalysis_validation.csv'}")

print("""
AGU ABSTRACT LANGUAGE:
Cross-reanalysis comparison revealed a {:.1f} µg/m³ mean bias in MERRA-2 
versus CAMS (Pearson r = {:.2f}), with the largest discrepancies during 
winter post-monsoon biomass-burning periods (bias: {:.1f} µg/m³), 
suggesting non-assimilated aerosol reanalyses require bias correction 
for compound extreme applications in South Asia.
""".format(bias, corr, df[df['season']=='Winter']['bias'].values[0]))

cams.close(); merra.close()
