#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 1: Download ERA5 Data
Author: Poornima Suthar
Date: 2026-06-26

Downloads ERA5 reanalysis from Copernicus CDS for Ahmedabad bounding box.
Variables: 2m temperature, 10m wind (u,v), boundary layer height, relative humidity
"""

import os
import cdsapi
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "era5_ahmedabad_2019.nc"

# Ahmedabad bounding box [N, W, S, E]
BBOX = [23.15, 72.4, 22.9, 72.75]

def download_era5():
    """
    Download ERA5 hourly data for 2019.
    Requires Copernicus CDS API key (see: https://cds.climate.copernicus.eu/)
    """
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    if OUTPUT_FILE.exists():
        print(f"File already exists: {OUTPUT_FILE}")
        print("Skipping download. Delete file to re-download.")
        return
    
    print("Initializing CDS API client...")
    print("NOTE: You need to register at https://cds.climate.copernicus.eu/")
    print("and create a %USERPROFILE%\\.cdsapirc file with your API key.")
    
    c = cdsapi.Client()
    
    print(f"Downloading ERA5 for Ahmedabad ({BBOX})...")
    print("This may take 30-60 minutes depending on server load.")
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_temperature',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'boundary_layer_height',
                'relative_humidity',
            ],
            'year': '2019',
            'month': [f"{m:02d}" for m in range(1, 13)],
            'day': [f"{d:02d}" for d in range(1, 32)],
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': BBOX,
            'format': 'netcdf',
        },
        str(OUTPUT_FILE)
    )
    
    print(f"Download complete: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")

if __name__ == "__main__":
    download_era5()
    
