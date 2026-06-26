#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 2: Download CAMS PM2.5 Data
Author: Poornima Suthar
Date: 2026-06-26

Downloads CAMS global reanalysis PM2.5 from Copernicus ADS.
Uses global request (no area subset) then crops locally.
"""

import os
import cdsapi
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Ahmedabad bounding box for post-processing crop
BBOX = {
    'north': 23.15,
    'west': 72.4,
    'south': 22.9,
    'east': 72.75
}

def download_cams_month(year, month):
    """Download CAMS PM2.5 for a single month (global, then crop)."""
    
    DATA_DIR.mkdir(exist_ok=True)
    
    output_file = DATA_DIR / f"cams_pm25_global_{year}_{month:02d}.nc"
    cropped_file = DATA_DIR / f"cams_pm25_ahmedabad_{year}_{month:02d}.nc"
    
    if cropped_file.exists():
        print(f"File already exists: {cropped_file} — skipping.")
        return cropped_file
    
    # Step 1: Download global file (if not exists)
    if not output_file.exists():
        print(f"\n{'='*60}")
        print(f"Downloading global CAMS PM2.5 for {year}-{month:02d}...")
        print(f"{'='*60}")
        
        c = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api")
        
        try:
            c.retrieve(
                'cams-global-reanalysis-eac4',
                {
                    'variable': 'particulate_matter_2.5um',
                    'date': f'{year}-{month:02d}-01/to/{year}-{month:02d}-{get_last_day(year, month)}',
                    'time': [
                        '00:00', '03:00', '06:00',
                        '09:00', '12:00', '15:00',
                        '18:00', '21:00'
                    ],
                    'leadtime_hour': '0',
                    'type': 'forecast',
                    'format': 'netcdf',
                },
                str(output_file)
            )
            print(f"Global file saved: {output_file}")
        except Exception as e:
            print(f"ERROR downloading global file: {e}")
            return None
    
    # Step 2: Crop to Ahmedabad
    print(f"Cropping to Ahmedabad...")
    try:
        crop_to_ahmedabad(output_file, cropped_file)
        print(f"Cropped file saved: {cropped_file}")
        
        # Remove global file to save space
        output_file.unlink()
        print(f"Removed global file to save space.")
        
        return cropped_file
        
    except Exception as e:
        print(f"ERROR cropping: {e}")
        return None

def crop_to_ahmedabad(input_file, output_file):
    """Crop global NetCDF to Ahmedabad bounding box."""
    import xarray as xr
    
    ds = xr.open_dataset(input_file)
    
    # Find latitude/longitude names
    lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'
    
    # Crop
    ds_cropped = ds.sel(
        **{lat_name: slice(BBOX['north'], BBOX['south'])},
        **{lon_name: slice(BBOX['west'], BBOX['east'])}
    )
    
    # Save
    ds_cropped.to_netcdf(output_file)
    ds.close()
    ds_cropped.close()

def get_last_day(year, month):
    """Return last day of month."""
    import calendar
    return calendar.monthrange(year, month)[1]

def main():
    """Download all months for 2019."""
    
    year = 2019
    
    for month in range(1, 13):
        download_cams_month(year, month)
    
    print(f"\n{'='*60}")
    print("All CAMS downloads complete!")
    print(f"Files saved in: {DATA_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
    