#!/usr/bin/env python3
"""
SCE Window Sensitivity Analysis v2
Tests fixed windows (7-90 days) AND seasonal transition definition
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = PROJECT_DIR / "figures"

def get_time_values(da):
    for c in ['time', 'valid_time', 'date', 'day']:
        if c in da.coords:
            return da[c]
    return pd.date_range('2019-01-01', periods=len(da), freq='D')

print("=" * 70)
print("SCE WINDOW SENSITIVITY ANALYSIS v2")
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

# Thresholds
pm90 = pm25.quantile(0.90)
t90 = t2m.quantile(0.90)

pm_extreme = pm25 > pm90
t_extreme = t2m > t90
hpe = pm_extreme & t_extreme

print(f"PM2.5 90th: {pm90:.1f} µg/m³")
print(f"Tmax 90th: {t90:.1f} K ({t90-273.15:.1f}°C)")
print(f"PM extreme days: {pm_extreme.sum()} | Heat extreme days: {t_extreme.sum()} | HPE days: {hpe.sum()}")

# Identify PM extreme dates (single days, not just episodes)
pm_dates = pm25[pm_extreme].index
heat_dates = t2m[t_extreme].index

# Also identify PM episodes (consecutive)
pm_episodes = []
if len(pm_dates) > 0:
    s = p = pm_dates[0]
    for d in pm_dates[1:]:
        if d == p + pd.Timedelta(days=1):
            p = d
        else:
            pm_episodes.append((s, p))
            s = p = d
    pm_episodes.append((s, p))

print(f"\nPM episodes (consecutive): {len(pm_episodes)}")
print(f"PM single extreme days: {len(pm_dates)}")

# Test fixed windows
windows = [7, 14, 30, 45, 60, 90]
results = []

for w in windows:
    # Method A: Any PM extreme day (single or episode end) triggers window
    sce_events = 0
    sce_pm_days = 0
    sce_heat_days = set()
    
    for d in pm_dates:
        window_end = d + pd.Timedelta(days=w)
        heat_in_window = heat_dates[(heat_dates > d) & (heat_dates <= window_end)]
        if len(heat_in_window) > 0:
            sce_events += 1
            sce_pm_days += 1
            for hd in heat_in_window:
                sce_heat_days.add(hd)
    
    # Method B: Episode-based (only episode end triggers window)
    sce_events_ep = 0
    for start, end in pm_episodes:
        window_end = end + pd.Timedelta(days=w)
        heat_in_window = heat_dates[(heat_dates > end) & (heat_dates <= window_end)]
        if len(heat_in_window) > 0:
            sce_events_ep += 1
    
    results.append({
        'window': w,
        'sce_events_single': sce_events,
        'sce_events_episode': sce_events_ep,
        'sce_heat_days': len(sce_heat_days),
        'total_exposure_days': sce_pm_days + len(sce_heat_days)
    })

df_fixed = pd.DataFrame(results)
print("\n--- FIXED WINDOW RESULTS ---")
print(df_fixed.to_string(index=False))

# SEASONAL TRANSITION DEFINITION
# Winter PM (Nov-Feb) followed by Summer heat (Mar-Jun)
winter_months = [11, 12, 1, 2]
summer_months = [3, 4, 5, 6]

winter_pm_dates = pm_dates[pm_dates.month.isin(winter_months)]
summer_heat_dates = heat_dates[heat_dates.month.isin(summer_months)]

# Count winter PM days that have a summer heat day after them
sce_seasonal = 0
for d in winter_pm_dates:
    future_heat = summer_heat_dates[summer_heat_dates > d]
    if len(future_heat) > 0:
        sce_seasonal += 1

print(f"\n--- SEASONAL TRANSITION RESULTS ---")
print(f"Winter PM extreme days: {len(winter_pm_dates)}")
print(f"Summer heat extreme days: {len(summer_heat_dates)}")
print(f"SCE-Seasonal (Winter PM → Summer Heat): {sce_seasonal} events")

# Save
df_fixed.to_csv(OUTPUT_DIR / "sce_window_sensitivity_v2.csv", index=False)
seasonal_df = pd.DataFrame([{
    'definition': 'Seasonal Transition (Winter→Summer)',
    'winter_pm_days': len(winter_pm_dates),
    'summer_heat_days': len(summer_heat_dates),
    'sce_events': sce_seasonal
}])
seasonal_df.to_csv(OUTPUT_DIR / "sce_seasonal_definition.csv", index=False)

# Figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(df_fixed['window'], df_fixed['sce_events_single'], 'o-', color='#E63946', lw=2.5, ms=8, label='SCE Events (single-day trigger)')
ax.plot(df_fixed['window'], df_fixed['sce_events_episode'], 's--', color='#457B9D', lw=2, ms=8, label='SCE Events (episode-end trigger)')
ax.axhline(y=hpe.sum(), color='black', ls=':', lw=2, label=f'HPE Days ({hpe.sum()})')
ax.set_xlabel('Window Length (days)')
ax.set_ylabel('Event Count')
ax.set_title('(a) SCE Events vs. Fixed Window')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
methods = ['HPE\n(Simultaneous)', 'SCE-30', 'SCE-60', 'SCE-90', 'SCE-Seasonal\n(Winter→Summer)']
counts = [hpe.sum(), 
          df_fixed[df_fixed['window']==30]['sce_events_single'].values[0],
          df_fixed[df_fixed['window']==60]['sce_events_single'].values[0],
          df_fixed[df_fixed['window']==90]['sce_events_single'].values[0],
          sce_seasonal]
colors = ['#2A9D8F', '#F4A261', '#E9C46A', '#E63946', '#A23B72']
bars = ax.bar(methods, counts, color=colors, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            str(int(val)), ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Event Count')
ax.set_title('(b) Comparison of Frameworks')
ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig8_window_sensitivity_v2.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {FIG_DIR / 'fig8_window_sensitivity_v2.png'}")

cams.close(); era5.close()
print("\nDone.")
