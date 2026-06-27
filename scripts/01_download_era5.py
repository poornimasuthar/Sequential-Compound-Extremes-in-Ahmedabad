#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 1: Download ERA5 Data (Monthly Chunks)
Author: Poornima Suthar
Date: 2026-06-26

Downloads ERA5 reanalysis in monthly chunks to avoid Copernicus limits.
"""

import os
import cdsapi
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
BBOX = [23.15, 72.4, 22.9, 72.75]  # [N, W, S, E]

# Variables to download
VARIABLES = [
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'boundary_layer_height',
    'relative_humidity',
]

def download_month(year, month):
    """Download ERA5 for a single month."""
    
    DATA_DIR.mkdir(exist_ok=True)
    
    output_file = DATA_DIR / f"era5_ahmedabad_{year}_{month:02d}.nc"
    
    if output_file.exists():
        print(f"File already exists: {output_file} — skipping.")
        return
    
    print(f"\n{'='*60}")
    print(f"Downloading ERA5 for {year}-{month:02d}...")
    print(f"{'='*60}")
    
    c = cdsapi.Client()
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': VARIABLES,
            'year': str(year),
            'month': f"{month:02d}",
            'day': [f"{d:02d}" for d in range(1, 32)],
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': BBOX,
            'format': 'netcdf',
        },
        str(output_file)
    )
    
    print(f"Saved: {output_file}")
    print(f"Size: {output_file.stat().st_size / (1024**2):.1f} MB")

def main():
    """Download all months for 2019."""
    
    year = 2019
    
    for month in range(1, 13):
        try:
            download_month(year, month)
        except Exception as e:
            print(f"ERROR downloading {year}-{month:02d}: {e}")
            print("Continuing to next month...")
            continue
    
    print(f"\n{'='*60}")
    print("All downloads complete!")
    print(f"Files saved in: {DATA_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
    