#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 9: HPE Detection with Grid Alignment
Author: Poornima Suthar
Date: 2026-06-27

Detects HPEs with proper spatial alignment between ERA5 and CAMS.
Uses nearest-neighbor selection to align CAMS to ERA5 grid.
Following Yim et al. (2025) methodology.
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

    # Align CAMS to ERA5 grid using nearest neighbor
    print("\nAligning CAMS to ERA5 grid (nearest neighbor)...")
    pm25_aligned = pm25.sel(
        latitude=tmax.latitude,
        longitude=tmax.longitude,
        method="nearest"
    )
    print(f"Aligned PM2.5 grid: lat={pm25_aligned.latitude.values}, lon={pm25_aligned.longitude.values}")

    # Rename valid_time to time for alignment
    if "valid_time" in pm25_aligned.dims:
        pm25_aligned = pm25_aligned.rename({"valid_time": "time"})

    # Align time periods
    tmax_start = pd.Timestamp(tmax.time.min().values)
    tmax_end = pd.Timestamp(tmax.time.max().values)
    pm25_start = pd.Timestamp(pm25_aligned.time.min().values)
    pm25_end = pd.Timestamp(pm25_aligned.time.max().values)

    start = max(tmax_start, pm25_start)
    end = min(tmax_end, pm25_end)

    tmax_sub = tmax.sel(time=slice(start, end))
    pm25_sub = pm25_aligned.sel(time=slice(start, end))

    print(f"\nCommon period: {str(start)[:10]} to {str(end)[:10]}")
    print(f"Days: {len(tmax_sub.time)}")

    # Yim et al. (2025) thresholds
    # Grid-specific 90th percentile for Tmax
    tmax_90th = tmax_sub.quantile(0.90, dim="time")
    print(f"\nTmax 90th percentile: {float(tmax_90th.mean().values):.1f} C (mean across grid)")

    # Grid-specific 75th percentile for PM2.5, minimum 15 ug/m3 (WHO)
    pm25_75th = pm25_sub.quantile(0.75, dim="time")
    pm25_threshold = xr.where(pm25_75th > 15, pm25_75th, 15.0)
    print(f"PM2.5 75th percentile: {float(pm25_75th.mean().values):.1f} ug/m3 (mean across grid)")
    print(f"PM2.5 threshold (max of 75th and 15): {float(pm25_threshold.mean().values):.1f} ug/m3")

    # HPE detection: co-occurrence on same day
    hot_days = tmax_sub > tmax_90th
    polluted_days = pm25_sub > pm25_threshold
    hpe_days = hot_days & polluted_days

    # Summary statistics
    total = len(tmax_sub.time)
    hot = int(hot_days.sum().values)
    pol = int(polluted_days.sum().values)
    hpe = int(hpe_days.sum().values)

    print(f"\n{'='*60}")
    print("HPE DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Total days analyzed: {total}")
    print(f"Hot days: {hot} ({100*hot/total:.1f}%)")
    print(f"Polluted days: {pol} ({100*pol/total:.1f}%)")
    print(f"HPE days: {hpe} ({100*hpe/total:.1f}%)")

    # Monthly breakdown
    print(f"\nMonthly HPE breakdown:")
    months = pd.date_range(start=start, end=end, freq="MS")
    monthly_results = []

    for m in months:
        ms = slice(m, m + pd.offsets.MonthEnd(1))
        try:
            tmax_m = tmax_sub.sel(time=ms)
            pm25_m = pm25_sub.sel(time=ms)
            hpe_m = hpe_days.sel(time=ms)
            hot_m = hot_days.sel(time=ms)
            pol_m = polluted_days.sel(time=ms)

            n_days = len(tmax_m.time)
            n_hot = int(hot_m.sum().values)
            n_pol = int(pol_m.sum().values)
            n_hpe = int(hpe_m.sum().values)

            tmax_mean = float(tmax_m.mean().values)
            pm25_mean = float(pm25_m.mean().values)

            monthly_results.append({
                "month": m.strftime("%Y-%m"),
                "days": n_days,
                "hot_days": n_hot,
                "polluted_days": n_pol,
                "hpe_days": n_hpe,
                "tmax_mean": tmax_mean,
                "pm25_mean": pm25_mean
            })

            print(f"  {m.strftime('%Y-%m')}: {n_hpe} HPE days (hot={n_hot}, pol={n_pol}, tmax={tmax_mean:.1f}C, pm25={pm25_mean:.1f})")
        except:
            pass

    # Save monthly results
    df = pd.DataFrame(monthly_results)
    df.to_csv(OUTPUT_DIR / "hpe_monthly_breakdown.csv", index=False)
    print(f"\nSaved: hpe_monthly_breakdown.csv")

    # HPE characteristics (Yim 2025)
    print(f"\n{'='*60}")
    print("HPE CHARACTERISTICS (per grid cell)")
    print(f"{'='*60}")

    # Frequency: number of HPE days per year
    hpe_frequency = hpe_days.sum(dim="time")
    print(f"HPE frequency range: {int(hpe_frequency.min().values)} to {int(hpe_frequency.max().values)} days")
    print(f"Mean HPE frequency: {float(hpe_frequency.mean().values):.1f} days")

    # Duration: longest consecutive HPE period
    def longest_consecutive(arr):
        if not arr.any():
            return 0
        runs = []
        current = 0
        for val in arr:
            if val:
                current += 1
            else:
                if current > 0:
                    runs.append(current)
                    current = 0
        if current > 0:
            runs.append(current)
        return max(runs) if runs else 0

    # Apply to each grid cell
    hpe_duration = xr.apply_ufunc(
        lambda x: longest_consecutive(x),
        hpe_days,
        input_core_dims=[["time"]],
        vectorize=True
    )
    print(f"Max HPE duration: {int(hpe_duration.max().values)} days")

    # PM2.5 intensity during HPEs
    pm25_hpe = pm25_sub.where(hpe_days)
    pm25_intensity = pm25_hpe.max(dim="time")
    print(f"Max PM2.5 during HPEs: {float(pm25_intensity.max().values):.1f} ug/m3")

    # Temperature intensity during HPEs
    tmax_hpe = tmax_sub.where(hpe_days)
    tmax_intensity = tmax_hpe.max(dim="time")
    print(f"Max Tmax during HPEs: {float(tmax_intensity.max().values):.1f} C")

    # Save HPE results
    hpe_ds = xr.Dataset({
        "hpe": hpe_days,
        "hot": hot_days,
        "polluted": polluted_days,
        "hpe_frequency": hpe_frequency,
        "hpe_duration": hpe_duration,
        "pm25_intensity": pm25_intensity,
        "tmax_intensity": tmax_intensity
    })
    hpe_ds.attrs["method"] = "Yim et al. (2025) HPE definition"
    hpe_ds.attrs["tmax_threshold"] = "90th percentile, grid-specific"
    hpe_ds.attrs["pm25_threshold"] = "max(75th percentile, 15 ug/m3), grid-specific"
    hpe_ds.attrs["spatial_alignment"] = "CAMS nearest-neighbor to ERA5 grid"

    output_file = OUTPUT_DIR / "hpe_daily_results_aligned.nc"
    hpe_ds.to_netcdf(output_file)
    print(f"\nSaved: {output_file}")

    # Save summary CSV
    summary = {
        "total_days": total,
        "hot_days": hot,
        "polluted_days": pol,
        "hpe_days": hpe,
        "hpe_percent": 100*hpe/total,
        "mean_hpe_frequency": float(hpe_frequency.mean().values),
        "max_hpe_duration": int(hpe_duration.max().values),
        "max_pm25_intensity": float(pm25_intensity.max().values),
        "max_tmax_intensity": float(tmax_intensity.max().values)
    }

    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(OUTPUT_DIR / "hpe_daily_summary_aligned.csv", index=False)
    print(f"Saved: hpe_daily_summary_aligned.csv")

    era5.close()
    cams.close()

    print(f"\n{'='*60}")
    print("HPE DETECTION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
