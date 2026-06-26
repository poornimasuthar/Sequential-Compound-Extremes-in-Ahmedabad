#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 4: Process Washington U. PM2.5
Author: Poornima Suthar
Date: 2026-06-26

Crops global PM2.5 to Ahmedabad and calculates daily/monthly means.
"""

import xarray as xr
import numpy as np
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Ahmedabad bounding box
BBOX = {
    'lat_min': 22.9,
    'lat_max': 23.15,
    'lon_min': 72.4,
    'lon_max': 72.75
}

def process_pm25():
    """Process Washington U. PM2.5 for Ahmedabad."""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find PM2.5 file
pm25_files = list(DATA_DIR.glob("V5GL05*.nc"))    
    if not pm25_files:
        print("No PM2.5 file found in data/")
        print("Download from: https://sites.wustl.edu/acag/datasets/surface-pm2-5/")
        return
    
    input_file = pm25_files[0]
    print(f"Processing: {input_file.name}")
    
    # Load dataset
    print("Loading global PM2.5...")
    ds = xr.open_dataset(input_file)
    
    print(f"Dataset dimensions: {ds.dims}")
    print(f"Variables: {list(ds.data_vars)}")
    
    # Find PM2.5 variable (usually 'PM25' or 'GWRPM25')
    pm25_var = None
    for var in ds.data_vars:
        if 'PM25' in var.upper() or 'PM2.5' in var:
            pm25_var = var
            break
    
    if not pm25_var:
        print(f"Available variables: {list(ds.data_vars)}")
        pm25_var = input("Enter PM2.5 variable name: ")
    
    print(f"Using variable: {pm25_var}")
    
    # Crop to Ahmedabad
    print("Cropping to Ahmedabad...")
    
    # Handle different coordinate names
    lat_name = 'lat' if 'lat' in ds.dims else 'latitude'
    lon_name = 'lon' if 'lon' in ds.dims else 'longitude'
    
    ds_cropped = ds.sel(
        **{lat_name: slice(BBOX['lat_min'], BBOX['lat_max'])},
        **{lon_name: slice(BBOX['lon_min'], BBOX['lon_max'])}
    )
    
    # Extract PM2.5
    pm25 = ds_cropped[pm25_var]
    
    print(f"Cropped dimensions: {pm25.dims}")
    print(f"Shape: {pm25.shape}")
    print(f"Value range: {pm25.min().values:.1f} to {pm25.max().values:.1f} µg/m³")
    
    # Calculate monthly means (if daily data)
    if 'time' in pm25.dims:
        print("Calculating monthly means...")
        pm25_monthly = pm25.resample(time='1M').mean()
        
        # Also keep daily if available
        pm25_daily = pm25 if len(pm25.time) > 365 else None
    else:
        pm25_monthly = pm25
        pm25_daily = None
    
    # Save outputs
    output_monthly = OUTPUT_DIR / "pm25_monthly_ahmedabad_2019.nc"
    print(f"Saving monthly data to: {output_monthly}")
    pm25_monthly.to_netcdf(output_monthly)
    
    if pm25_daily is not None:
        output_daily = OUTPUT_DIR / "pm25_daily_ahmedabad_2019.nc"
        print(f"Saving daily data to: {output_daily}")
        pm25_daily.to_netcdf(output_daily)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PM2.5 PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Monthly mean range: {pm25_monthly.min().values:.1f} to {pm25_monthly.max().values:.1f} µg/m³")
    print(f"Monthly mean: {pm25_monthly.mean().values:.1f} µg/m³")
    print(f"Grid: {len(pm25_monthly[lat_name])} x {len(pm25_monthly[lon_name])}")
    
    ds.close()

if __name__ == "__main__":
    process_pm25()
    