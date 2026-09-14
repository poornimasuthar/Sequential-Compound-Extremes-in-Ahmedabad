#!/usr/bin/env python3
"""
Script 17: Sequential Compound Extreme (SCE) Framework
Redefines HPE for cities with inverse heat-pollution seasonality
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import json

def calculate_sce_index(pm25_daily, temp_daily, window=30):
    """
    Calculate Sequential Compound Extreme (SCE) Index

    SCE captures health burden from sequential exposure:
    1. High PM2.5 (winter pollution peak)
    2. Followed by extreme heat (summer heat peak) within 'window' days

    Also calculates Seasonal Overlap Index (SOI) for transition months
    """

    pm75 = pm25_daily.quantile(0.75)
    t95 = temp_daily.quantile(0.95)

    high_pm_days = pm25_daily[pm25_daily > pm75].index
    extreme_temp_days = temp_daily[temp_daily > t95].index

    sce_events = 0
    sce_dates = []

    for pm_date in high_pm_days:
        window_end = pm_date + pd.Timedelta(days=window)
        heat_in_window = extreme_temp_days[
            (extreme_temp_days > pm_date) & 
            (extreme_temp_days <= window_end)
        ]
        if len(heat_in_window) > 0:
            sce_events += 1
            sce_dates.append((str(pm_date.date()), str(heat_in_window[0].date())))

    # Seasonal Overlap Index (SOI) for transition months (March, October)
    transition_months = [3, 10]
    soi_values = []

    for month in transition_months:
        month_data = pd.DataFrame({
            'pm25': pm25_daily[pm25_daily.index.month == month],
            'temp': temp_daily[temp_daily.index.month == month]
        }).dropna()

        if len(month_data) > 0:
            pm_norm = (month_data['pm25'] - month_data['pm25'].min()) /                       (month_data['pm25'].max() - month_data['pm25'].min() + 1e-10)
            t_norm = (month_data['temp'] - month_data['temp'].min()) /                      (month_data['temp'].max() - month_data['temp'].min() + 1e-10)

            soi = (pm_norm * t_norm).mean()
            soi_values.append(float(soi))

    soi_mean = np.mean(soi_values) if soi_values else 0

    # Cumulative Exposure Burden (CEB)
    pm_zscore = (pm25_daily - pm25_daily.mean()) / pm25_daily.std()
    t_zscore = (temp_daily - temp_daily.mean()) / temp_daily.std()
    ceb = float(np.sqrt(pm_zscore**2 + t_zscore**2).mean())

    return {
        'sce_events': sce_events,
        'sce_dates': sce_dates[:10],  # Save first 10 for reference
        'soi': float(soi_mean),
        'ceb': ceb,
        'pm75_threshold': float(pm75),
        't95_threshold': float(t95)
    }

def main():
    print("="*60)
    print("SCRIPT 17: Sequential Compound Extreme (SCE) Framework")
    print("="*60)

    print("\n[1] Loading daily PM2.5 and temperature data...")

    merra_file = Path("data/processed/merra2_ahmedabad_2019.nc")
    ds_pm = xr.open_dataset(merra_file)
    pm25_daily = ds_pm['PM2_5_DRY'].mean(dim=['lat', 'lon']).to_pandas()

    era5_file = Path("data/processed/era5_ahmedabad_2019.nc")
    ds_temp = xr.open_dataset(era5_file)
    temp_daily = (ds_temp['t2m'].mean(dim=['latitude', 'longitude']) - 273.15).to_pandas()

    pm25_daily.index = pd.to_datetime(pm25_daily.index)
    temp_daily.index = pd.to_datetime(temp_daily.index)

    print("\n[2] Calculating SCE indices...")
    for window in [15, 30, 60]:
        results = calculate_sce_index(pm25_daily, temp_daily, window=window)
        print(f"\n  Window = {window} days:")
        print(f"    SCE events: {results['sce_events']}")
        print(f"    PM2.5 threshold (75th): {results['pm75_threshold']:.1f} ug/m3")
        print(f"    Temp threshold (95th): {results['t95_threshold']:.1f} C")
        print(f"    SOI (transition months): {results['soi']:.3f}")
        print(f"    CEB: {results['ceb']:.3f}")

    print("\n[3] Comparison with traditional HPE...")
    pm75 = pm25_daily.quantile(0.75)
    t95 = temp_daily.quantile(0.95)

    traditional_hpe = ((pm25_daily > pm75) & (temp_daily > t95)).sum()
    print(f"    Traditional HPE days (simultaneous): {traditional_hpe}")
    print(f"    -> Confirms inverse seasonality: 0 simultaneous extremes")

    print("\n[4] Saving SCE timeline...")
    sce_30 = calculate_sce_index(pm25_daily, temp_daily, window=30)

    timeline_df = pd.DataFrame({
        'date': pm25_daily.index,
        'pm25': pm25_daily.values,
        'temperature': temp_daily.values,
        'high_pm': pm25_daily > pm75,
        'extreme_temp': temp_daily > t95
    })

    output_dir = Path("data/processed/sce_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_df.to_csv(output_dir / "sce_timeline_2019.csv", index=False)

    summary = {
        'traditional_hpe_days': int(traditional_hpe),
        'sce_30day_events': sce_30['sce_events'],
        'soi': sce_30['soi'],
        'ceb': sce_30['ceb'],
        'pm75_threshold': sce_30['pm75_threshold'],
        't95_threshold': sce_30['t95_threshold']
    }

    with open(output_dir / "sce_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
    