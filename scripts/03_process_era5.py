#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 3: Process ERA5 Data
Author: Poornima Suthar
Date: 2026-06-26

Processes raw ERA5 hourly data into daily aggregates:
- Daily maximum 2m temperature (°C)
- Daily mean wind speed and direction
- Daily mean boundary layer height
"""

import xarray as xr
import numpy as np
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "era5_daily_2019.nc"

def process_era5():
    """Process all ERA5 monthly files into daily aggregates."""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find all ERA5 files
    era5_files = sorted(DATA_DIR.glob("era5_ahmedabad_2019_*.nc"))
    
    if not era5_files:
        print("No ERA5 files found in data/")
        print("Run 01_download_era5.py first.")
        return
    
    print(f"Found {len(era5_files)} ERA5 monthly files.")
    
    # Load and combine all months
    print("Loading data...")
    datasets = []
    for f in era5_files:
        print(f"  Loading {f.name}")
        ds = xr.open_dataset(f)
        datasets.append(ds)
    
    # Combine along time dimension
    ds = xr.concat(datasets, dim='time')
    ds = ds.sortby('time')
    
    print(f"Combined dataset: {ds.dims}")
    
    # Convert temperature from K to °C
    if 't2m' in ds.variables:
        print("Converting temperature to Celsius...")
        ds['t2m'] = ds['t2m'] - 273.15
        ds['t2m'].attrs['units'] = '°C'
    
    # Calculate daily maximum temperature
    print("Calculating daily maximum temperature...")
    daily_tmax = ds['t2m'].resample(time='1D').max()
    
    # Calculate daily mean wind components
    print("Calculating daily wind...")
    daily_u = ds['u10'].resample(time='1D').mean()
    daily_v = ds['v10'].resample(time='1D').mean()
    
    # Wind speed
    daily_wspd = np.sqrt(daily_u**2 + daily_v**2)
    daily_wspd.attrs['units'] = 'm/s'
    
    # Wind direction (meteorological: from which wind blows)
    daily_wdir = (180 / np.pi) * np.arctan2(daily_u, daily_v) + 180
    daily_wdir.attrs['units'] = 'degrees'
    
    # Daily mean boundary layer height
    print("Calculating daily boundary layer height...")
    daily_blh = ds['blh'].resample(time='1D').mean()
    daily_blh.attrs['units'] = 'm'
    
    # Daily mean relative humidity
    print("Calculating daily relative humidity...")
    daily_rh = ds['r2'].resample(time='1D').mean()
    daily_rh.attrs['units'] = '%'
    
    # Create output dataset
    daily_ds = xr.Dataset({
        'tmax': daily_tmax,
        'wspd': daily_wspd,
        'wdir': daily_wdir,
        'blh': daily_blh,
        'rh': daily_rh
    })
    
    # Add metadata
    daily_ds.attrs['title'] = 'ERA5 Daily Aggregates for Ahmedabad 2019'
    daily_ds.attrs['source'] = 'Copernicus CDS ERA5 reanalysis'
    daily_ds.attrs['processing'] = 'Daily max T2m, daily mean wind/BLH/RH'
    
    # Save
    print(f"\nSaving to: {OUTPUT_FILE}")
    daily_ds.to_netcdf(OUTPUT_FILE)
    
    print(f"Done! File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Time range: {daily_ds.time.min().values} to {daily_ds.time.max().values}")
    print(f"Grid: {len(daily_ds.latitude)} x {len(daily_ds.longitude)}")
    print(f"Tmax range: {daily_ds['tmax'].min().values:.1f} to {daily_ds['tmax'].max().values:.1f} °C")
    print(f"Wspd range: {daily_ds['wspd'].min().values:.1f} to {daily_ds['wspd'].max().values:.1f} m/s")
    print(f"BLH range: {daily_ds['blh'].min().values:.1f} to {daily_ds['blh'].max().values:.1f} m")

if __name__ == "__main__":
    process_era5()
    