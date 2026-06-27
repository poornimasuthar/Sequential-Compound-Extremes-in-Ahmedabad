#!/usr/bin/env python3
"""
Process MERRA-2 aerosol components to PM2.5
Yim's formula: PM2.5 = DUSMASS25 + SSSMASS25 + BCSMASS + OCSMASS + 1.375*SO4SMASS
MERRA-2 already cropped to single point (lat 23.0, lon 72.5)
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import re

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "merra2"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def extract_date(fname):
    """Extract YYYYMMDD from filename."""
    # Try to find date pattern in filename
    match = re.search(r'(\d{8})', fname)
    if match:
        return pd.Timestamp(match.group(1))
    return None


def process_merra2():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    files = sorted(DATA_DIR.glob("*.nc4"))
    files = [f for f in files if "README" not in f.name]
    
    print(f"Found {len(files)} MERRA-2 files")
    if len(files) == 0:
        print("No files found")
        return
    
    daily_pm25 = []
    dates = []
    
    for i, fpath in enumerate(files, 1):
        try:
            ds = xr.open_dataset(fpath)
            
            # Calculate PM2.5 (kg/m3)
            pm25 = (ds["DUSMASS25"] + ds["SSSMASS25"] + ds["BCSMASS"] + 
                    ds["OCSMASS"] + 1.375 * ds["SO4SMASS"])
            
            # Convert to ug/m3
            pm25 = pm25 * 1e9
            
            # Daily mean - squeeze out time dimension (already 1 time step)
            pm25_day = pm25.mean(dim="time").squeeze()
            
            # Extract date
            date = extract_date(fpath.name)
            if date is None:
                date = pd.Timestamp("2019-01-01") + pd.Timedelta(days=i-1)
            
            daily_pm25.append(float(pm25_day.values))
            dates.append(date)
            
            if i % 50 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {date.strftime('%Y-%m-%d')} - PM2.5: {daily_pm25[-1]:.1f} ug/m3")
            
            ds.close()
            
        except Exception as e:
            print(f"  ERROR {fpath.name}: {e}")
    
    # Create DataArray
    print(f"\nProcessed {len(daily_pm25)} days")
    
    pm25_da = xr.DataArray(
        daily_pm25,
        coords=[dates],
        dims=["time"],
        name="pm25"
    )
    pm25_da = pm25_da.sortby("time")
    
    print(f"Time range: {str(dates[0])[:10]} to {str(dates[-1])[:10]}")
    print(f"Daily mean PM2.5: {float(pm25_da.mean().values):.1f} ug/m3")
    print(f"Min: {float(pm25_da.min().values):.1f} ug/m3")
    print(f"Max: {float(pm25_da.max().values):.1f} ug/m3")
    
    # Save as Dataset
    ds_out = xr.Dataset({"pm25": pm25_da})
    ds_out.attrs["title"] = "MERRA-2 Daily PM2.5 Ahmedabad 2019"
    ds_out.attrs["formula"] = "DUSMASS25 + SSSMASS25 + BCSMASS + OCSMASS + 1.375*SO4SMASS"
    ds_out.attrs["location"] = "lat 23.0, lon 72.5"
    ds_out.to_netcdf(OUTPUT_DIR / "pm25_daily_merra2_ahmedabad_2019.nc")
    print("\nSaved: pm25_daily_merra2_ahmedabad_2019.nc")
    
    # 3-day moving average
    pm25_3day = pm25_da.rolling(time=3, center=True).mean()
    ds_3day = xr.Dataset({"pm25": pm25_3day})
    ds_3day.to_netcdf(OUTPUT_DIR / "pm25_3day_merra2_ahmedabad_2019.nc")
    print("Saved: pm25_3day_merra2_ahmedabad_2019.nc")


if __name__ == "__main__":
    process_merra2()
    