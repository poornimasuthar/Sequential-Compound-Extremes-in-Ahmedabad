#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 9: HPE Detection with Grid Alignment (FIXED)
Author: Poornima Suthar
Date: 2026-06-27

Handles duplicate coordinates by selecting common overlapping point.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

def main():
    print("="*60)
    print("HPE DETECTION - DAILY RESOLUTION (GRID ALIGNED)")
    print("="*60)

    # Load ERA5
    era5 = xr.open_dataset(OUTPUT_DIR / "era5_daily_2019.nc")
    tmax = era5["tmax"]
    print(f"ERA5 Tmax grid: lat={tmax.latitude.values}, lon={tmax.longitude.values}")

    # Load CAMS PM2.5
    cams = xr.open_dataset(OUTPUT_DIR / "pm25_3day_cams_ahmedabad_2019.nc")
    pm25 = cams["pm25"]
    print(f"CAMS PM2.5 grid: lat={pm25.latitude.values}, lon={pm25.longitude.values}")

    # Find common longitude (the overlapping point)
    era5_lon = tmax.longitude.values
    cams_lon = pm25.longitude.values
    common_lon = np.intersect1d(era5_lon, cams_lon)

    print(f"\nCommon longitude points: {common_lon}")

    if len(common_lon) == 0:
        print("ERROR: No common longitude points!")
        print("Using nearest neighbor for ERA5 lon 72.5...")
        # Select CAMS at lon 72.75 (closest to both ERA5 points)
        pm25_aligned = pm25.sel(longitude=72.75, method="nearest")
        tmax_aligned = tmax.sel(longitude=72.75, method="nearest")
    else:
        # Use common point(s)
        pm25_aligned = pm25.sel(longitude=common_lon)
        tmax_aligned = tmax.sel(longitude=common_lon)

    # Also align latitude
    # ERA5 lat is 23.0, CAMS has 22.5 and 23.25
    # Use nearest
    pm25_aligned = pm25_aligned.sel(latitude=tmax_aligned.latitude, method="nearest")

    print(f"\nAligned ERA5 grid: lat={tmax_aligned.latitude.values}, lon={tmax_aligned.longitude.values}")
    print(f"Aligned CAMS grid: lat={pm25_aligned.latitude.values}, lon={pm25_aligned.longitude.values}")

    # Rename valid_time to time
    if "valid_time" in pm25_aligned.dims:
        pm25_aligned = pm25_aligned.rename({"valid_time": "time"})

    # Align time
    tmax_start = pd.Timestamp(tmax_aligned.time.min().values)
    tmax_end = pd.Timestamp(tmax_aligned.time.max().values)
    pm25_start = pd.Timestamp(pm25_aligned.time.min().values)
    pm25_end = pd.Timestamp(pm25_aligned.time.max().values)

    start = max(tmax_start, pm25_start)
    end = min(tmax_end, pm25_end)

    tmax_sub = tmax_aligned.sel(time=slice(start, end))
    pm25_sub = pm25_aligned.sel(time=slice(start, end))

    print(f"\nCommon period: {str(start)[:10]} to {str(end)[:10]}")
    print(f"Days: {len(tmax_sub.time)}")

    # Thresholds
    tmax_90th = tmax_sub.quantile(0.90, dim="time")
    pm25_75th = pm25_sub.quantile(0.75, dim="time")
    pm25_threshold = xr.where(pm25_75th > 15, pm25_75th, 15.0)

    print(f"\nTmax 90th percentile: {float(tmax_90th.mean().values):.1f} C")
    print(f"PM2.5 75th percentile: {float(pm25_75th.mean().values):.1f} ug/m3")
    print(f"PM2.5 threshold: {float(pm25_threshold.mean().values):.1f} ug/m3")

    # HPE detection
    hot_days = tmax_sub > tmax_90th
    polluted_days = pm25_sub > pm25_threshold
    hpe_days = hot_days & polluted_days

    # Summary
    total = len(tmax_sub.time)
    hot = int(hot_days.sum().values)
    pol = int(polluted_days.sum().values)
    hpe = int(hpe_days.sum().values)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total days: {total}")
    print(f"Hot days: {hot} ({100*hot/total:.1f}%)")
    print(f"Polluted days: {pol} ({100*pol/total:.1f}%)")
    print(f"HPE days: {hpe} ({100*hpe/total:.1f}%)")

    # Monthly breakdown
    print(f"\nMonthly breakdown:")
    months = pd.date_range(start=start, end=end, freq="MS")
    for m in months:
        ms = slice(m, m + pd.offsets.MonthEnd(1))
        try:
            h = int(hpe_days.sel(time=ms).sum().values)
            print(f"  {m.strftime('%Y-%m')}: {h} HPE days")
        except:
            pass

    # Save
    # Drop conflicting attributes before saving
    hpe_days_clean = hpe_days.drop_vars('quantile', errors='ignore')
    hot_days_clean = hot_days.drop_vars('quantile', errors='ignore')
    polluted_days_clean = polluted_days.drop_vars('quantile', errors='ignore')

    hpe_ds = xr.Dataset({
    "hpe": hpe_days_clean,
    "hot": hot_days_clean,
    "polluted": polluted_days_clean
})
    hpe_ds.to_netcdf(OUTPUT_DIR / "hpe_daily_results_aligned.nc")
    print(f"\nSaved: hpe_daily_results_aligned.nc")

    era5.close()
    cams.close()

if __name__ == "__main__":
    main()
