#!/usr/bin/env python3
"""
HPE Detection using MERRA-2 PM2.5 + ERA5 Temperature
Compare with CAMS-based HPE results
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

print("=" * 60)
print("HPE DETECTION - MERRA-2 + ERA5")
print("=" * 60)

# Load MERRA-2 daily PM2.5
merra = xr.open_dataset(OUTPUT_DIR / "pm25_daily_merra2_ahmedabad_2019.nc")
pm25_merra = merra["pm25"]

# Load ERA5 daily temperature
era5 = xr.open_dataset(OUTPUT_DIR / "era5_daily_2019.nc")
t2m = era5["tmax"] if "tmax" in era5 else era5["t2m"]

# Get time values
merra_time = pd.to_datetime(pm25_merra.time.values) if hasattr(pm25_merra, 'time') else pd.date_range('2019-01-01', periods=len(pm25_merra), freq='D')
era5_time = pd.to_datetime(t2m.time.values) if hasattr(t2m, 'time') else pd.date_range('2019-01-01', periods=len(t2m), freq='D')

# Convert to pandas Series
pm25_series = pd.Series(pm25_merra.values, index=merra_time)
t2m_series = pd.Series(t2m.mean(dim=['latitude', 'longitude']).values, index=era5_time)

# Align to common dates
common_dates = pm25_series.index.intersection(t2m_series.index)
pm25_aligned = pm25_series.loc[common_dates]
t2m_aligned = t2m_series.loc[common_dates]

print(f"\nCommon days: {len(common_dates)}")
print(f"PM2.5 range: {pm25_aligned.min():.1f} - {pm25_aligned.max():.1f} µg/m³")
print(f"T2m range: {t2m_aligned.min():.1f} - {t2m_aligned.max():.1f} K")

# Calculate percentiles
pm25_p90 = pm25_aligned.quantile(0.90)
t2m_p90 = t2m_aligned.quantile(0.90)

print(f"\nPercentile thresholds:")
print(f"  PM2.5 90th: {pm25_p90:.1f} µg/m³")
print(f"  T2m 90th: {t2m_p90:.1f} K")

# HPE detection: PM2.5 > 90th AND Tmax > 90th
hpe_mask = (pm25_aligned > pm25_p90) & (t2m_aligned > t2m_p90)
hpe_days = pm25_aligned[hpe_mask]

print(f"\n{'='*60}")
print("HPE DETECTION RESULTS (MERRA-2 + ERA5)")
print(f"{'='*60}")
print(f"Total days analyzed: {len(common_dates)}")
print(f"HPE days detected: {len(hpe_days)}")
print(f"HPE percentage: {len(hpe_days)/len(common_dates)*100:.1f}%")

if len(hpe_days) > 0:
    print(f"\nHPE Day Details:")
    print(f"{'Date':<12} {'PM2.5':>10} {'T2m':>10}")
    print("-" * 35)
    for date, pm25_val in hpe_days.items():
        t2m_val = t2m_aligned.loc[date]
        print(f"{date.strftime('%Y-%m-%d'):<12} {pm25_val:>10.1f} {t2m_val:>10.1f}")
else:
    print("\nNo HPE days detected with MERRA-2 + ERA5.")
    print("This is consistent with CAMS result (0 days).")
    print("Reason: Inverse seasonal pattern in Ahmedabad —")
    print("  PM2.5 peaks in winter, temperature peaks in summer.")

# Also check individual extremes
pm25_extreme = (pm25_aligned > pm25_p90).sum()
t2m_extreme = (t2m_aligned > t2m_p90).sum()

print(f"\n{'='*60}")
print("INDIVIDUAL EXTREMES")
print(f"{'='*60}")
print(f"Days with PM2.5 > 90th percentile: {pm25_extreme}")
print(f"Days with T2m > 90th percentile: {t2m_extreme}")
print(f"Days with BOTH extremes (HPE): {len(hpe_days)}")

# Monthly breakdown
print(f"\n{'='*60}")
print("MONTHLY HPE BREAKDOWN")
print(f"{'='*60}")
monthly = pd.DataFrame({
    'pm25': pm25_aligned,
    't2m': t2m_aligned,
    'hpe': hpe_mask
})
monthly['month'] = monthly.index.month
monthly_summary = monthly.groupby('month').agg({
    'pm25': 'mean',
    't2m': 'mean',
    'hpe': 'sum'
})
print(f"{'Month':<8} {'PM2.5':>8} {'T2m':>8} {'HPE Days':>10}")
print("-" * 40)
for month, row in monthly_summary.iterrows():
    month_name = pd.Timestamp(2019, month, 1).strftime('%b')
    print(f"{month_name:<8} {row['pm25']:>8.1f} {row['t2m']:>8.1f} {int(row['hpe']):>10}")

merra.close()
era5.close()

print("\n" + "=" * 60)
