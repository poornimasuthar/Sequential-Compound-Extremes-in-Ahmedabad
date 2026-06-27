#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 4: Process Washington U. PM2.5
Author: Poornima Suthar
Date: 2026-06-26

Crops global PM2.5 to Ahmedabad and calculates monthly means.
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
    
    # Find PM2.5 variable
    pm25_var = None
    for var in ds.data_vars:
        var_upper = str(var).upper()
        if 'PM25' in var_upper or 'PM2.5' in var_upper or 'GWRPM25' in var_upper:
            pm25_var = var
            break
    
    if not pm25_var:
        print(f"Available variables: {list(ds.data_vars)}")
        pm25_var = input("Enter PM2.5 variable name: ")
    
    print(f"Using variable: {pm25_var}")
    
    # Find coordinate names
    lat_name = 'lat' if 'lat' in ds.dims else 'latitude'
    lon_name = 'lon' if 'lon' in ds.dims else 'longitude'
    
    # Crop to Ahmedabad
    print("Cropping to Ahmedabad...")
    ds_cropped = ds.sel(
        **{lat_name: slice(BBOX['lat_min'], BBOX['lat_max'])},
        **{lon_name: slice(BBOX['lon_min'], BBOX['lon_max'])}
    )
    
    # Extract PM2.5
    pm25 = ds_cropped[pm25_var]
    
    print(f"Cropped dimensions: {pm25.dims}")
    print(f"Shape: {pm25.shape}")
    print(f"Value range: {float(pm25.min().values):.1f} to {float(pm25.max().values):.1f} µg/m³")
    
    # Check if time dimension exists
    if 'time' in pm25.dims or 'month' in pm25.dims:
        print("Dataset has time dimension.")
        
        # If monthly data, keep as is
        if 'month' in pm25.dims:
            pm25_monthly = pm25
        else:
            # Resample to monthly if daily
            print("Calculating monthly means...")
            pm25_monthly = pm25.resample(time='1M').mean()
    else:
        print("No time dimension found. Treating as single time slice.")
        pm25_monthly = pm25
    
    # Save output
    output_monthly = OUTPUT_DIR / "pm25_monthly_ahmedabad_2019.nc"
    print(f"Saving monthly data to: {output_monthly}")
    pm25_monthly.to_netcdf(output_monthly)
    
    # Also save as annual mean if monthly
    if len(pm25_monthly.shape) > 2:
        print("Calculating annual mean...")
        pm25_annual = pm25_monthly.mean(dim='time' if 'time' in pm25_monthly.dims else 'month')
        output_annual = OUTPUT_DIR / "pm25_annual_ahmedabad_2019.nc"
        print(f"Saving annual mean to: {output_annual}")
        pm25_annual.to_netcdf(output_annual)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PM2.5 PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Monthly mean range: {float(pm25_monthly.min().values):.1f} to {float(pm25_monthly.max().values):.1f} µg/m³")
    print(f"Monthly mean average: {float(pm25_monthly.mean().values):.1f} µg/m³")
    print(f"Grid: {len(pm25_monthly[lat_name])} x {len(pm25_monthly[lon_name])}")
    
    ds.close()

if __name__ == "__main__":
    process_pm25()
    