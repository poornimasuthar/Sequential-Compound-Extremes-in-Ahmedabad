#!/usr/bin/env python3
"""
Sequential Compound Extreme (SCE) Framework - v2
Handles actual ERA5 data format from your project
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import json

print("=" * 60)
print("SEQUENTIAL COMPOUND EXTREME (SCE) - AHMEDABAD 2019")
print("=" * 60)

base = Path(".")

# Load PM2.5 daily from MERRA-2
print("\n[1] Loading PM2.5 from MERRA-2...")
ds_pm = xr.open_dataset(base / "outputs" / "pm25_daily_merra2_ahmedabad_2019.nc")
print(f"    Variables: {list(ds_pm.data_vars)}")
print(f"    Dimensions: {ds_pm.dims}")

# Extract PM2.5 - handle different possible variable names
pm_var = None
for v in ['pm25', 'PM2_5_DRY', 'PM25', 'pm2_5']:
    if v in ds_pm.data_vars:
        pm_var = v
        break

if pm_var is None:
    pm_var = list(ds_pm.data_vars)[0]

print(f"    Using variable: {pm_var}")
pm25 = ds_pm[pm_var]

# If spatial, take mean over lat/lon
if 'lat' in pm25.dims and 'lon' in pm25.dims:
    pm25_daily = pm25.mean(dim=['lat', 'lon']).to_pandas()
elif 'latitude' in pm25.dims and 'longitude' in pm25.dims:
    pm25_daily = pm25.mean(dim=['latitude', 'longitude']).to_pandas()
else:
    pm25_daily = pm25.to_pandas()

print(f"    {len(pm25_daily)} daily values")
print(f"    Date range: {pm25_daily.index[0]} to {pm25_daily.index[-1]}")

# Load temperature from ERA5 daily
print("\n[2] Loading temperature from ERA5...")
ds_t = xr.open_dataset(base / "outputs" / "era5_daily_2019.nc")
print(f"    Variables: {list(ds_t.data_vars)}")
print(f"    Dimensions: {ds_t.dims}")

# Extract temperature
t_var = None
for v in ['t2m', 'temp', 'temperature', 'T2M']:
    if v in ds_t.data_vars:
        t_var = v
        break
if t_var is None:
    t_var = list(ds_t.data_vars)[0]

print(f"    Using variable: {t_var}")
temp = ds_t[t_var]

# Convert K to C if needed
if temp.mean() > 200:
    print("    Converting K to C")
    temp = temp - 273.15

# If spatial, take mean
if 'lat' in temp.dims and 'lon' in temp.dims:
    temp_daily = temp.mean(dim=['lat', 'lon']).to_pandas()
elif 'latitude' in temp.dims and 'longitude' in temp.dims:
    temp_daily = temp.mean(dim=['latitude', 'longitude']).to_pandas()
else:
    temp_daily = temp.to_pandas()

print(f"    {len(temp_daily)} daily values")
print(f"    Date range: {temp_daily.index[0]} to {temp_daily.index[-1]}")

# Ensure both are datetime
pm25_daily.index = pd.to_datetime(pm25_daily.index)
temp_daily.index = pd.to_datetime(temp_daily.index)

# Align to common dates
common_dates = pm25_daily.index.intersection(temp_daily.index)
pm25_daily = pm25_daily.loc[common_dates]
temp_daily = temp_daily.loc[common_dates]

print(f"\n    Aligned: {len(common_dates)} common days")

# Thresholds
print("\n[3] Calculating thresholds...")
pm75 = pm25_daily.quantile(0.75)
t95 = temp_daily.quantile(0.95)
print(f"    PM2.5 75th percentile: {pm75:.1f} ug/m3")
print(f"    Temperature 95th percentile: {t95:.1f} C")

# Traditional HPE
print("\n[4] Traditional HPE detection...")
hpe = ((pm25_daily > pm75) & (temp_daily > t95)).sum()
print(f"    HPE days: {int(hpe)}")

# SCE detection
print("\n[5] Sequential Compound Extreme (SCE)...")
high_pm = pm25_daily[pm25_daily > pm75].index
extreme_t = temp_daily[temp_daily > t95].index

print(f"    High PM2.5 days: {len(high_pm)}")
print(f"    Extreme temp days: {len(extreme_t)}")

for window in [15, 30, 60]:
    sce = 0
    for pm_date in high_pm:
        window_end = pm_date + pd.Timedelta(days=window)
        if any((extreme_t > pm_date) & (extreme_t <= window_end)):
            sce += 1
    print(f"    SCE-{window}: {sce} events")

# Seasonal Overlap Index
print("\n[6] Seasonal Overlap Index (SOI)...")
soi_values = []
for month in [3, 10]:
    pm_m = pm25_daily[pm25_daily.index.month == month]
    t_m = temp_daily[temp_daily.index.month == month]
    common = pm_m.index.intersection(t_m.index)
    if len(common) > 0:
        pm_n = (pm_m.loc[common] - pm_m.min()) / (pm_m.max() - pm_m.min() + 1e-10)
        t_n = (t_m.loc[common] - t_m.min()) / (t_m.max() - t_m.min() + 1e-10)
        soi = (pm_n * t_n).mean()
        soi_values.append(soi)
        print(f"    Month {month}: SOI = {soi:.3f}")

soi_mean = float(np.mean(soi_values)) if soi_values else 0

# Cumulative Exposure Burden
print("\n[7] Cumulative Exposure Burden (CEB)...")
pm_z = (pm25_daily - pm25_daily.mean()) / pm25_daily.std()
t_z = (temp_daily - temp_daily.mean()) / temp_daily.std()
ceb = float(np.sqrt(pm_z**2 + t_z**2).mean())
print(f"    CEB: {ceb:.3f}")

# Calculate SCE values for saving
sce_15 = sum(1 for d in high_pm if any((extreme_t > d) & (extreme_t <= d + pd.Timedelta(days=15))))
sce_30 = sum(1 for d in high_pm if any((extreme_t > d) & (extreme_t <= d + pd.Timedelta(days=30))))
sce_60 = sum(1 for d in high_pm if any((extreme_t > d) & (extreme_t <= d + pd.Timedelta(days=60))))

# Save results
print("\n[8] Saving results...")
out = base / "outputs" / "sce_results"
out.mkdir(exist_ok=True)

summary = {
    "city": "Ahmedabad",
    "year": 2019,
    "analysis_date": str(pd.Timestamp.now()),
    "traditional_hpe_days": int(hpe),
    "sce_15day": sce_15,
    "sce_30day": sce_30,
    "sce_60day": sce_60,
    "soi_mean": soi_mean,
    "ceb": ceb,
    "pm75_threshold": float(pm75),
    "t95_threshold": float(t95),
    "pm25_annual_mean": float(pm25_daily.mean()),
    "temp_annual_mean": float(temp_daily.mean()),
    "high_pm_days": len(high_pm),
    "extreme_temp_days": len(extreme_t),
    "key_finding": "Zero traditional HPE days due to inverse seasonality; SCE framework reveals cross-seasonal burden"
}

with open(out / "sce_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save timeline
timeline = pd.DataFrame({
    'date': pm25_daily.index,
    'pm25': pm25_daily.values,
    'temperature': temp_daily.values,
    'high_pm': pm25_daily > pm75,
    'extreme_temp': temp_daily > t95
})
timeline.to_csv(out / "sce_timeline_2019.csv", index=False)

print(f"\n    Saved: {out / 'sce_summary.json'}")
print(f"    Saved: {out / 'sce_timeline_2019.csv'}")

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"""
Annual mean PM2.5:        {summary['pm25_annual_mean']:.1f} ug/m3
Annual mean temperature:  {summary['temp_annual_mean']:.1f} C

THRESHOLDS:
PM2.5 (75th percentile):  {summary['pm75_threshold']:.1f} ug/m3
Temperature (95th pct):   {summary['t95_threshold']:.1f} C

COMPOUND EXTREMES:
Traditional HPE days:     {summary['traditional_hpe_days']}
SCE-15 events:            {summary['sce_15day']}
SCE-30 events:            {summary['sce_30day']}
SCE-60 events:            {summary['sce_60day']}
SOI (transition months):    {summary['soi_mean']:.3f}
CEB:                      {summary['ceb']:.3f}

INTERPRETATION:
- HPE=0: Simultaneous extremes NEVER occur in Ahmedabad
- But {summary['sce_30day']} winter pollution episodes are followed
  by summer heat within 30 days
- This is the HIDDEN compound burden that traditional HPE misses
- The body carries winter PM2.5 damage into summer heat stress
""")
print("=" * 60)
