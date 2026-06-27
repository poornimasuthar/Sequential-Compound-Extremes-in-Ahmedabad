#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 11: Process MERRA-2 PM2.5
Author: Poornima Suthar
Date: 2026-06-27

Calculates PM2.5 from MERRA-2 aerosol components using Yim et al. (2025) formula:
PM2.5 = DUSMASS25 + SSSMASS25 + BCSMASS + OCSMASS + 1.375 * SO4SMASS
"""

import xarray as xr
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "merra2"
OUTPUT_DIR = PROJECT_DIR / "outputs"

def process_merra2_pm25():
    """Process MERRA-2 aerosol components into PM2.5."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find all MERRA-2 files
    merra_files = sorted(DATA_DIR.glob("MERRA2_*.nc"))

    if not merra_files:
        print("No MERRA-2 files found in data/merra2/")
        print("Run 10_download_merra2.py first.")
        return

    print(f"Found {len(merra_files)} MERRA-2 daily files.")

    # Required variables for PM2.5 calculation
    required_vars = ["DUSMASS25", "SSSMASS25", "BCSMASS", "OCSMASS", "SO4SMASS"]

    # Process each file
    datasets = []
    for i, f in enumerate(merra_files):
        if i % 50 == 0:
            print(f"  Processing {i+1}/{len(merra_files)}: {f.name}")

        ds = xr.open_dataset(f)

        # Check all variables exist
        missing = [v for v in required_vars if v not in ds]
        if missing:
            print(f"    WARNING: Missing variables: {missing}")
            ds.close()
            continue

        # Extract components
        dust = ds["DUSMASS25"]      # Dust PM2.5
        seasalt = ds["SSSMASS25"]   # Sea salt PM2.5
        bc = ds["BCSMASS"]          # Black carbon
        oc = ds["OCSMASS"]          # Organic carbon
        so4 = ds["SO4SMASS"]        # Sulfate

        # Calculate PM2.5 using Yim et al. (2025) formula
        # Note: MERRA-2 units are kg/m3, convert to ug/m3 (*1e9)
        pm25 = (dust + seasalt + bc + oc + 1.375 * so4) * 1e9
        pm25.attrs["units"] = "ug/m3"
        pm25.attrs["long_name"] = "PM2.5 surface mass concentration"
        pm25.attrs["formula"] = "DUSMASS25 + SSSMASS25 + BCSMASS + OCSMASS + 1.375*SO4SMASS"

        # Keep time coordinate
        pm25 = pm25.expand_dims("time") if "time" not in pm25.dims else pm25

        datasets.append(pm25)
        ds.close()

    # Combine all days
    print(f"\nCombining {len(datasets)} daily datasets...")
    pm25_all = xr.concat(datasets, dim="time")
    pm25_all = pm25_all.sortby("time")

    print(f"Time range: {str(pm25_all.time.min().values)[:10]} to {str(pm25_all.time.max().values)[:10]}")
    print(f"Total days: {len(pm25_all.time)}")

    # Find coordinate names
    lat_name = "lat" if "lat" in pm25_all.dims else "latitude"
    lon_name = "lon" if "lon" in pm25_all.dims else "longitude"
    print(f"Grid: {len(pm25_all[lat_name])} x {len(pm25_all[lon_name])}")

    # Statistics
    print(f"\nPM2.5 Statistics:")
    print(f"  Min: {float(pm25_all.min().values):.1f} ug/m3")
    print(f"  Max: {float(pm25_all.max().values):.1f} ug/m3")
    print(f"  Mean: {float(pm25_all.mean().values):.1f} ug/m3")

    # Save
    output_file = OUTPUT_DIR / "pm25_daily_merra2_ahmedabad_2019.nc"
    ds_out = xr.Dataset({"pm25": pm25_all})
    ds_out.attrs["title"] = "MERRA-2 Daily PM2.5 for Ahmedabad 2019"
    ds_out.attrs["source"] = "NASA GES DISC MERRA-2 tavg1_2d_aer_Nx"
    ds_out.attrs["formula"] = "DUSMASS25 + SSSMASS25 + BCSMASS + OCSMASS + 1.375*SO4SMASS"
    ds_out.attrs["reference"] = "Yim et al. (2025) GeoHealth"
    ds_out.to_netcdf(output_file)

    print(f"\nSaved: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024**2):.1f} MB")

    # Component breakdown (annual mean)
    print(f"\n{'='*60}")
    print("PM2.5 COMPONENT BREAKDOWN (Annual Mean)")
    print(f"{'='*60}")

    # Re-open first file to get component stats (approximate)
    ds_sample = xr.open_dataset(merra_files[0])
    for var in required_vars:
        if var in ds_sample:
            mean_val = float(ds_sample[var].mean().values) * 1e9
            print(f"  {var}: {mean_val:.1f} ug/m3")
    ds_sample.close()

if __name__ == "__main__":
    process_merra2_pm25()
