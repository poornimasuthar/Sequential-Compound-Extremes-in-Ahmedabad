#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 3: Process ERA5 Data
Author: Poornima Suthar
Date: 2026-06-26

Processes raw ERA5 hourly data into daily aggregates.
"""

import xarray as xr
import numpy as np
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "era5_daily_2019.nc"

def standardize_time(ds):
    """Standardize time coordinate."""
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    if not np.issubdtype(ds['time'].dtype, np.datetime64):
        ds['time'] = xr.decode_cf(ds)['time']
    return ds

def process_era5():
    """Process all ERA5 monthly files into daily aggregates."""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    era5_files = sorted(DATA_DIR.glob("era5_ahmedabad_2019_*.nc"))
    
    if not era5_files:
        print("No ERA5 files found in data/")
        return
    
    print(f"Found {len(era5_files)} ERA5 monthly files.")
    
    # Load and standardize
    datasets = []
    for f in era5_files:
        print(f"  Loading {f.name}")
        ds = xr.open_dataset(f)
        ds = standardize_time(ds)
        datasets.append(ds)
    
    # Combine
    print("Combining datasets...")
    ds = xr.concat(datasets, dim='time', combine_attrs='override')
    ds = ds.sortby('time')
    
    print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
    print(f"Total time steps: {len(ds.time)}")
    
    lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'
    print(f"Grid: {len(ds[lat_name])} x {len(ds[lon_name])}")
    
    # Convert temperature
    if 't2m' in ds.variables:
        print("Converting temperature to Celsius...")
        ds['t2m'] = ds['t2m'] - 273.15
        ds['t2m'].attrs['units'] = '°C'
    
    # Calculate daily aggregates
    print("Calculating daily maximum temperature...")
    daily_tmax = ds['t2m'].resample(time='1D').max()
    
    print("Calculating daily wind...")
    daily_u = ds['u10'].resample(time='1D').mean()
    daily_v = ds['v10'].resample(time='1D').mean()
    daily_wspd = np.sqrt(daily_u**2 + daily_v**2)
    daily_wspd.attrs['units'] = 'm/s'
    
    daily_wdir = (180 / np.pi) * np.arctan2(daily_u, daily_v) + 180
    daily_wdir = np.mod(daily_wdir, 360)
    daily_wdir.attrs['units'] = 'degrees'
    
    print("Calculating daily boundary layer height...")
    daily_blh = ds['blh'].resample(time='1D').mean()
    daily_blh.attrs['units'] = 'm'
    
    # Relative humidity - check variable name
    rh_var = None
    for var in ['r', 'rh', 'r2', 'relative_humidity']:
        if var in ds.variables:
            rh_var = var
            break
    
    if rh_var:
        print(f"Calculating daily relative humidity from '{rh_var}'...")
        daily_rh = ds[rh_var].resample(time='1D').mean()
        daily_rh.attrs['units'] = '%'
    else:
        print("WARNING: No relative humidity variable found. Skipping.")
        daily_rh = None
    
    # Create output dataset
    data_vars = {
        'tmax': daily_tmax,
        'wspd': daily_wspd,
        'wdir': daily_wdir,
        'blh': daily_blh
    }
    if daily_rh is not None:
        data_vars['rh'] = daily_rh
    
    daily_ds = xr.Dataset(data_vars)
    daily_ds.attrs['title'] = 'ERA5 Daily Aggregates for Ahmedabad 2019'
    daily_ds.attrs['source'] = 'Copernicus CDS ERA5 reanalysis'
    
    # Save
    print(f"\nSaving to: {OUTPUT_FILE}")
    daily_ds.to_netcdf(OUTPUT_FILE)
    
    print(f"Done! File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Time range: {daily_ds.time.min().values} to {daily_ds.time.max().values}")
    print(f"Total days: {len(daily_ds.time)}")
    print(f"Grid: {len(daily_ds[lat_name])} x {len(daily_ds[lon_name])}")
    print(f"Tmax range: {daily_tmax.min().values:.1f} to {daily_tmax.max().values:.1f} °C")
    print(f"Tmax mean: {daily_tmax.mean().values:.1f} °C")
    print(f"Wspd mean: {daily_wspd.mean().values:.1f} m/s")
    print(f"BLH mean: {daily_blh.mean().values:.1f} m")
    
    for ds in datasets:
        ds.close()
    daily_ds.close()

if __name__ == "__main__":
    process_era5()
    