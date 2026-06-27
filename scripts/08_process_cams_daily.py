#!/usr/bin/env python3
import xarray as xr
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

def process_cams_daily():
    OUTPUT_DIR.mkdir(exist_ok=True)
    cams_files = list(DATA_DIR.glob("data_sfc.nc"))
    if not cams_files:
        print("No CAMS files found")
        return
    input_file = cams_files[0]
    print(f"Processing: {input_file.name}")
    ds = xr.open_dataset(input_file)
    pm25 = ds["pm2p5"]
    print(f"Original shape: {pm25.shape}")
    
    # Crop to Ahmedabad region using nearest coordinates
    # Lat: 22.5 is closest to 22.9, 23.25 is closest to 23.15
    # Lon: 72.75 is exact
    pm25_cropped = pm25.sel(latitude=[22.5, 23.25], longitude=[72.0, 72.75], method="nearest")
    print(f"Cropped shape: {pm25_cropped.shape}")
    print(f"Cropped lat: {pm25_cropped.latitude.values}")
    print(f"Cropped lon: {pm25_cropped.longitude.values}")
    
    # Convert kg/m3 to ug/m3
    pm25_cropped = pm25_cropped * 1e9
    print(f"Range: {float(pm25_cropped.min().values):.1f} to {float(pm25_cropped.max().values):.1f} ug/m3")
    
    # Resample to daily mean
    pm25_daily = pm25_cropped.resample(valid_time="1D").mean()
    pm25_3day = pm25_daily.rolling(valid_time=3, center=True).mean()
    # Save
    ds_daily = xr.Dataset({"pm25": pm25_daily})
    ds_daily.attrs["title"] = "CAMS EAC4 Daily PM2.5 Ahmedabad 2019"
    ds_daily.attrs["source"] = "Copernicus ADS"
    ds_daily.to_netcdf(OUTPUT_DIR / "pm25_daily_cams_ahmedabad_2019.nc")
    print("Saved daily")
    
    ds_3day = xr.Dataset({"pm25": pm25_3day})
    ds_3day.to_netcdf(OUTPUT_DIR / "pm25_3day_cams_ahmedabad_2019.nc")
    print("Saved 3-day MA")
    print(f"Daily mean: {float(pm25_daily.mean().values):.1f} ug/m3")
    
    ds.close()

if __name__ == "__main__":
    process_cams_daily()
    