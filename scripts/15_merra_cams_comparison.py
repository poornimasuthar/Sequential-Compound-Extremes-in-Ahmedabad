#!/usr/bin/env python3
"""
MERRA-2 vs CAMS PM2.5 comparison for Ahmedabad 2019
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

def main():
    print("=" * 60)
    print("MERRA-2 vs CAMS PM2.5 COMPARISON")
    print("=" * 60)
    
    # Load both datasets
    merra = xr.open_dataset(OUTPUT_DIR / "pm25_daily_merra2_ahmedabad_2019.nc")
    cams = xr.open_dataset(OUTPUT_DIR / "pm25_daily_cams_ahmedabad_2019.nc")
    
    merra_pm25 = merra["pm25"]
    cams_pm25 = cams["pm25"]
    
    # Align time
    cams_pm25 = cams_pm25.rename({"valid_time": "time"})
    
    # For CAMS, take spatial mean (2x2 grid) to compare with MERRA-2 single point
    cams_mean = cams_pm25.mean(dim=["latitude", "longitude"])
    
    # Find common period
    start = max(pd.Timestamp(merra_pm25.time.min().values), 
                pd.Timestamp(cams_mean.time.min().values))
    end = min(pd.Timestamp(merra_pm25.time.max().values), 
              pd.Timestamp(cams_mean.time.max().values))
    
    merra_sub = merra_pm25.sel(time=slice(start, end))
    cams_sub = cams_mean.sel(time=slice(start, end))
    
    print(f"Common period: {str(start)[:10]} to {str(end)[:10]}")
    print(f"Days: {len(merra_sub.time)}")
    
    # Statistics
    print(f"\n{'='*60}")
    print("DAILY STATISTICS")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'MERRA-2':>12} {'CAMS':>12} {'Diff':>12}")
    print("-" * 60)
    
    metrics = {
        'Mean': (float(merra_sub.mean().values), float(cams_sub.mean().values)),
        'Median': (float(merra_sub.median().values), float(cams_sub.median().values)),
        'Std Dev': (float(merra_sub.std().values), float(cams_sub.std().values)),
        'Min': (float(merra_sub.min().values), float(cams_sub.min().values)),
        'Max': (float(merra_sub.max().values), float(cams_sub.max().values)),
        '75th %ile': (float(merra_sub.quantile(0.75).values), float(cams_sub.quantile(0.75).values)),
        '90th %ile': (float(merra_sub.quantile(0.9).values), float(cams_sub.quantile(0.9).values)),
    }
    
    for name, (m, c) in metrics.items():
        diff = m - c
        print(f"{name:<20} {m:>12.1f} {c:>12.1f} {diff:>+12.1f}")
    
    # Correlation
    m_vals = merra_sub.values
    c_vals = cams_sub.values
    
    # Remove NaN
    mask = ~(np.isnan(m_vals) | np.isnan(c_vals))
    if mask.sum() > 10:
        corr = np.corrcoef(m_vals[mask], c_vals[mask])[0, 1]
        rmse = np.sqrt(np.mean((m_vals[mask] - c_vals[mask])**2))
        mae = np.mean(np.abs(m_vals[mask] - c_vals[mask]))
        
        print(f"\n{'='*60}")
        print("CORRELATION")
        print(f"{'='*60}")
        print(f"Pearson r: {corr:.3f}")
        print(f"RMSE: {rmse:.1f} ug/m3")
        print(f"MAE: {mae:.1f} ug/m3")
        print(f"Bias (MERRA-CAMS): {np.mean(m_vals[mask] - c_vals[mask]):+.1f} ug/m3")
    
    # Monthly breakdown
    print(f"\n{'='*60}")
    print("MONTHLY MEANS")
    print(f"{'='*60}")
    print(f"{'Month':<10} {'MERRA-2':>12} {'CAMS':>12} {'Diff':>12}")
    print("-" * 50)
    
    months = pd.date_range("2019-01-01", "2019-12-01", freq="MS")
    for m in months:
        ms = slice(m, m + pd.offsets.MonthEnd(1))
        try:
            mm = float(merra_sub.sel(time=ms).mean().values)
            cm = float(cams_sub.sel(time=ms).mean().values)
            print(f"{m.strftime('%Y-%m'):<10} {mm:>12.1f} {cm:>12.1f} {mm-cm:>+12.1f}")
        except:
            pass
    
    # Save comparison
    df = pd.DataFrame({
        'time': merra_sub.time.values,
        'merra2': m_vals,
        'cams': c_vals[:len(m_vals)] if len(c_vals) > len(m_vals) else c_vals
    })
    df.to_csv(OUTPUT_DIR / "merra_cams_comparison.csv", index=False)
    print(f"\nSaved: merra_cams_comparison.csv")
    
    merra.close()
    cams.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

    