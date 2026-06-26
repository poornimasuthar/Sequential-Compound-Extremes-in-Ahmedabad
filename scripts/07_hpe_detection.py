#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 7: HPE Detection
Author: Poornima Suthar
Date: 2026-06-27
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

def main():
    print("HPE DETECTION")
    print("="*60)

    # Load ERA5
    era5_file = OUTPUT_DIR / "era5_daily_2019.nc"
    if not era5_file.exists():
        print("ERROR: ERA5 not found")
        return

    era5 = xr.open_dataset(era5_file)
    print(f"ERA5 loaded: {dict(era5.dims)}")

    # Load PM2.5
    pm25_file = OUTPUT_DIR / "pm25_monthly_ahmedabad_2019_final.nc"
    if not pm25_file.exists():
        print("ERROR: Monthly PM2.5 not found. Run 04b first.")
        era5.close()
        return

    pm25 = xr.open_dataset(pm25_file)
    print(f"PM2.5 loaded: {dict(pm25.dims)}")

    # Thresholds
    tmax = era5['tmax']
    tmax_90th = float(tmax.quantile(0.90, dim='time').mean().values)

    pm25_data = pm25['pm25']
    pm25_75th = float(pm25_data.quantile(0.75).mean().values)
    pm25_threshold = max(pm25_75th, 15.0)

    print(f"\nThresholds:")
    print(f"  Tmax 90th: {tmax_90th:.1f} C")
    print(f"  PM2.5 75th: {pm25_75th:.1f} ug/m3")
    print(f"  PM2.5 threshold: {pm25_threshold:.1f} ug/m3")

    # Monthly Tmax
    tmax_monthly = tmax.resample(time='ME').max()
    
    # Results
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    results = []

    n_months = min(12, len(tmax_monthly), len(pm25_data.time))

    for i in range(n_months):
        tmax_val = float(tmax_monthly.isel(time=i).mean().values)
        pm25_val = float(pm25_data.isel(time=i).mean().values)

        hot = tmax_val > tmax_90th
        polluted = pm25_val > pm25_threshold
        hpe = hot and polluted

        results.append({
            'month': months[i],
            'tmax': tmax_val,
            'pm25': pm25_val,
            'hot': hot,
            'polluted': polluted,
            'hpe': hpe
        })

    # Print
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Month':<8} {'Tmax':<8} {'PM2.5':<8} {'Hot':<6} {'Polluted':<10} {'HPE':<6}")
    print("-" * 60)

    hpe_count = 0
    for r in results:
        print(f"{r['month']:<8} {r['tmax']:<8.1f} {r['pm25']:<8.1f} "
              f"{'Yes' if r['hot'] else 'No':<6} "
              f"{'Yes' if r['polluted'] else 'No':<10} "
              f"{'Yes' if r['hpe'] else 'No':<6}")
        if r['hpe']:
            hpe_count += 1

    print(f"\nHPE months: {hpe_count}/{n_months}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "hpe_results.csv", index=False)
    print(f"\nSaved: hpe_results.csv")

    era5.close()
    pm25.close()

if __name__ == "__main__":
    main()
