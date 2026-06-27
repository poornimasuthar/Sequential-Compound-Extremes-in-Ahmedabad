#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

def main():
    print("="*60)
    print("HPE DETECTION - DAILY RESOLUTION")
    print("="*60)
    
    era5 = xr.open_dataset(OUTPUT_DIR / "era5_daily_2019.nc")
    tmax = era5["tmax"]
    print(f"ERA5 Tmax: {tmax.dims}")    
    pm25_ds = xr.open_dataset(OUTPUT_DIR / "pm25_3day_cams_ahmedabad_2019.nc")
    pm25 = pm25_ds["pm25"]
    print(f"CAMS PM2.5: {pm25.dims}")
    
    # Thresholds
    tmax_90th = float(tmax.quantile(0.90, dim="time").mean().values)
    pm25_75th = float(pm25.quantile(0.75, dim="valid_time").mean().values)
    pm25_threshold = max(pm25_75th, 15.0)
    
    print(f"\nThresholds:")
    print(f"  Tmax 90th: {tmax_90th:.1f} C")
    print(f"  PM2.5 75th: {pm25_75th:.1f} ug/m3")
    print(f"  PM2.5 threshold: {pm25_threshold:.1f} ug/m3")
    
    # Align time
    tmax_start = pd.Timestamp(tmax.time.min().values)
    tmax_end = pd.Timestamp(tmax.time.max().values)
    pm25_start = pd.Timestamp(pm25.valid_time.min().values)
    pm25_end = pd.Timestamp(pm25.valid_time.max().values)
    
    start = max(tmax_start, pm25_start)
    end = min(tmax_end, pm25_end)
    
    tmax_sub = tmax.sel(time=slice(start, end))
    pm25_sub = pm25.sel(valid_time=slice(start, end))
    
    print(f"\nCommon period: {str(start)[:10]} to {str(end)[:10]}")
    print(f"Days: {len(tmax_sub.time)}")
    
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
    # Rename valid_time to time for alignment
    pm25_sub = pm25_sub.rename({"valid_time": "time"})
    print(f"Total days: {total}")
    print(f"Hot days: {hot} ({100*hot/total:.1f}%)")
    hot_days = tmax_sub > tmax_90th
    polluted_days = pm25_sub > pm25_threshold
    hpe_days = hot_days & polluted_days
    # Monthly breakdown
    print(f"\nMonthly HPE breakdown:")
    months = pd.date_range(start=start, end=end, freq="MS")
    for m in months:
        ms = slice(m, m + pd.offsets.MonthEnd(1))
        try:
            h = int(hpe_days.sel(time=ms).sum().values)
            print(f"  {m.strftime('%Y-%m')}: {h} HPE days")
        except:
            pass
    
    # Save
    ds_out = xr.Dataset({
        "hpe": hpe_days,
        "hot": hot_days,
        "polluted": polluted_days
    })
    ds_out.to_netcdf(OUTPUT_DIR / "hpe_daily_results.nc")
    print(f"\nSaved: hpe_daily_results.nc")
    
    era5.close()
    pm25_ds.close()

if __name__ == "__main__":
    main()
