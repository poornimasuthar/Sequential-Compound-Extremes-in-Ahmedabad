#!/usr/bin/env python3
"""
Ahmedabad HPE Project - Step 4b: Process Monthly Washington U. PM2.5
Author: Poornima Suthar
Date: 2026-06-27
"""

import xarray as xr
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

BBOX = {
    "lat_min": 22.9,
    "lat_max": 23.15,
    "lon_min": 72.4,
    "lon_max": 72.75
}

def process_monthly_pm25():
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_files = list(DATA_DIR.glob("V5GL0502*.nc"))
    print(f"Found {len(all_files)} total files:")
    for f in all_files:
        print(f"  {f.name}")

    # Monthly files: date part is like "201901-201901" (same month start and end)
    # Annual file: date part is like "201901-201912" (different months)
    monthly_files = []
    for f in all_files:
        parts = f.stem.split(".")
        date_part = parts[3]  # e.g., "201901-201901"
        if "-" in date_part:
            start, end = date_part.split("-")
            if start == end:  # Same month = monthly file
                monthly_files.append(f)

    print(f"\nFiltered to {len(monthly_files)} monthly files")

    if not monthly_files:
        print("ERROR: No monthly files found!")
        return

    datasets = []
    for f in sorted(monthly_files):
        print(f"\nProcessing {f.name}")
        ds = xr.open_dataset(f)

        lat_name = "lat" if "lat" in ds.dims else "latitude"
        lon_name = "lon" if "lon" in ds.dims else "longitude"

        ds_cropped = ds.sel(
            **{lat_name: slice(BBOX["lat_min"], BBOX["lat_max"])},
            **{lon_name: slice(BBOX["lon_min"], BBOX["lon_max"])}
        )

        pm25_var = "GWRPM25" if "GWRPM25" in ds.data_vars else list(ds.data_vars)[0]
        pm25 = ds_cropped[pm25_var]

        print(f"  Shape: {pm25.shape}, Range: {float(pm25.min().values):.1f} to {float(pm25.max().values):.1f}")

        date_part = f.stem.split(".")[3]
        month_str = date_part.split("-")[0]
        year = int(month_str[:4])
        month = int(month_str[4:6])
        time = np.datetime64(f"{year}-{month:02d}-15")

        pm25 = pm25.expand_dims("time")
        pm25["time"] = [time]

        datasets.append(pm25)
        ds.close()

    print(f"\nCombining {len(datasets)} datasets...")
    pm25_monthly = xr.concat(datasets, dim="time")
    pm25_monthly = pm25_monthly.sortby("time")

    ds_out = xr.Dataset({"pm25": pm25_monthly})
    output_file = OUTPUT_DIR / "pm25_monthly_ahmedabad_2019_final.nc"
    ds_out.to_netcdf(output_file)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Saved: {output_file}")
    print(f"Months: {len(pm25_monthly.time)}")
    for i, t in enumerate(pm25_monthly.time.values):
        val = float(pm25_monthly.isel(time=i).mean().values)
        print(f"  {str(t)[:7]}: {val:.1f} ug/m3")

if __name__ == "__main__":
    process_monthly_pm25()
