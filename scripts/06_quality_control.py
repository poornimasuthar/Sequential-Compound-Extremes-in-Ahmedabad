#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 6: Quality Control
Author: Poornima Suthar
Date: 2026-06-27

Comprehensive QC for all processed datasets.
Checks: ranges, missing values, spatial consistency, temporal coverage.
"""

import xarray as xr
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

def qc_era5():
    """Quality control for ERA5 daily data."""
    print("=" * 60)
    print("ERA5 DAILY DATA QC")
    print("=" * 60)
    
    file = OUTPUT_DIR / "era5_daily_2019.nc"
    if not file.exists():
        print("ERROR: File not found")
        return False
    
    ds = xr.open_dataset(file)
    
    # Check dimensions
    print(f"\nDimensions: {dict(ds.dims)}")
    
    # Check time coverage
    days = len(ds.time)
    expected_days = 365
    print(f"\nTime coverage: {days} days (expected: {expected_days})")
    if days != expected_days:
        print(f"WARNING: Missing {expected_days - days} days")
    
    # Check each variable
    checks = {
        'tmax': {'min': -10, 'max': 55, 'unit': '°C'},
        'wspd': {'min': 0, 'max': 50, 'unit': 'm/s'},
        'wdir': {'min': 0, 'max': 360, 'unit': '°'},
        'blh': {'min': 50, 'max': 3000, 'unit': 'm'},
    }
    
    all_pass = True
    for var, limits in checks.items():
        if var not in ds:
            print(f"\nWARNING: Variable '{var}' not found")
            continue
            
        data = ds[var]
        vmin = float(data.min().values)
        vmax = float(data.max().values)
        n_missing = int(np.isnan(data).sum().values)
        
        print(f"\n{var}:")
        print(f"  Range: {vmin:.2f} to {vmax:.2f} {limits['unit']}")
        print(f"  Expected: {limits['min']} to {limits['max']} {limits['unit']}")
        print(f"  Missing values: {n_missing}")
        
        if vmin < limits['min'] or vmax > limits['max']:
            print(f"  WARNING: Values outside expected range!")
            all_pass = False
        if n_missing > 0:
            print(f"  WARNING: Missing values detected!")
            all_pass = False
    
    ds.close()
    
    if all_pass:
        print(f"\n{'='*60}")
        print("ERA5 QC: PASSED")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("ERA5 QC: ISSUES FOUND")
        print(f"{'='*60}")
    
    return all_pass

def qc_pm25():
    """Quality control for PM2.5 data."""
    print("\n" + "=" * 60)
    print("PM2.5 DATA QC")
    print("=" * 60)
    
    # Check monthly file
    file = OUTPUT_DIR / "pm25_monthly_ahmedabad_2019_final.nc"
    if not file.exists():
        print("Monthly file not found, checking annual...")
        file = OUTPUT_DIR / "pm25_annual_ahmedabad_2019.nc"
    
    if not file.exists():
        print("ERROR: No PM2.5 file found")
        return False
    
    ds = xr.open_dataset(file)
    var = 'pm25' if 'pm25' in ds else 'GWRPM25'
    data = ds[var]
    
    print(f"\nVariable: {var}")
    print(f"Dimensions: {dict(data.dims)}")
    print(f"Shape: {data.shape}")
    
    vmin = float(data.min().values)
    vmax = float(data.max().values)
    vmean = float(data.mean().values)
    n_missing = int(np.isnan(data).sum().values)
    
    print(f"\nStatistics:")
    print(f"  Min: {vmin:.1f} µg/m³")
    print(f"  Max: {vmax:.1f} µg/m³")
    print(f"  Mean: {vmean:.1f} µg/m³")
    print(f"  Missing: {n_missing}")
    
    # WHO/India standards
    print(f"\nStandards comparison:")
    print(f"  WHO annual: 5 µg/m³ | India NAAQS: 40 µg/m³")
    print(f"  Exceeds WHO: {vmean > 5}")
    print(f"  Exceeds India: {vmean > 40}")
    
    # Spatial check
    if len(data.shape) >= 2:
        print(f"\nSpatial check:")
        print(f"  Grid cells: {data.shape[-2]} x {data.shape[-1]}")
        print(f"  All same value: {np.allclose(data, data.mean())}")
    
    ds.close()
    
    if vmin < 0 or vmax > 500:
        print(f"\n{'='*60}")
        print("PM2.5 QC: ISSUES FOUND (suspicious values)")
        print(f"{'='*60}")
        return False
    
    print(f"\n{'='*60}")
    print("PM2.5 QC: PASSED")
    print(f"{'='*60}")
    return True

def main():
    """Run all QC checks."""
    print("AHMEDABAD HPE PROJECT - QUALITY CONTROL")
    print("=" * 60)
    
    era5_ok = qc_era5()
    pm25_ok = qc_pm25()
    
    print("\n" + "=" * 60)
    print("FINAL QC SUMMARY")
    print("=" * 60)
    print(f"ERA5: {'PASS' if era5_ok else 'FAIL'}")
    print(f"PM2.5: {'PASS' if pm25_ok else 'FAIL'}")
    
    if era5_ok and pm25_ok:
        print("\nALL QC CHECKS PASSED - READY FOR ANALYSIS")
    else:
        print("\nSOME QC CHECKS FAILED - REVIEW BEFORE PROCEEDING")

if __name__ == "__main__":
    main()
    