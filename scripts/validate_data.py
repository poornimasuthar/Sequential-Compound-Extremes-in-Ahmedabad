#!/usr/bin/env python3
"""
Ahmedabad HPE Project - DATA VALIDATION
Author: Poornima Suthar
Date: 2026-06-27

Checks all datasets for quality, consistency, and correctness.
"""

import xarray as xr
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

def check_era5():
    """Validate ERA5 daily data."""
    print("="*60)
    print("ERA5 DAILY VALIDATION")
    print("="*60)

    file = OUTPUT_DIR / "era5_daily_2019.nc"
    if not file.exists():
        print("ERROR: File not found")
        return False

    ds = xr.open_dataset(file)
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Time range: {str(ds.time.min().values)[:10]} to {str(ds.time.max().values)[:10]}")
    print(f"Days: {len(ds.time)}")

    tmax = ds['tmax']
    print(f"\nTmax stats:")
    print(f"  Mean: {float(tmax.mean().values):.1f} C")
    print(f"  Min: {float(tmax.min().values):.1f} C")
    print(f"  Max: {float(tmax.max().values):.1f} C")
    print(f"  90th percentile: {float(tmax.quantile(0.9).values):.1f} C")

    # Check for Ahmedabad reasonableness
    # Ahmedabad typical Tmax: 15-45C
    if float(tmax.min().values) < 10 or float(tmax.max().values) > 50:
        print("  WARNING: Tmax outside expected range for Ahmedabad (10-50C)")
    else:
        print("  OK: Tmax in reasonable range")

    # Check spatial coverage
    if 'latitude' in ds.dims:
        print(f"\nGrid: {len(ds.latitude)} x {len(ds.longitude)}")
        print(f"  Lat: {float(ds.latitude.min().values):.2f} to {float(ds.latitude.max().values):.2f}")
        print(f"  Lon: {float(ds.longitude.min().values):.2f} to {float(ds.longitude.max().values):.2f}")

    ds.close()
    return True

def check_cams():
    """Validate CAMS PM2.5 data."""
    print("\n" + "="*60)
    print("CAMS PM2.5 VALIDATION")
    print("="*60)

    for fname in ["pm25_daily_cams_ahmedabad_2019.nc", "pm25_3day_cams_ahmedabad_2019.nc"]:
        file = OUTPUT_DIR / fname
        if not file.exists():
            continue

        print(f"\nFile: {fname}")
        ds = xr.open_dataset(file)
        pm25 = ds['pm25']

        time_name = 'time' if 'time' in pm25.dims else 'valid_time'
        print(f"Dimensions: {dict(pm25.dims)}")
        print(f"Time range: {str(pm25[time_name].min().values)[:10]} to {str(pm25[time_name].max().values)[:10]}")
        print(f"Days: {len(pm25[time_name])}")

        print(f"\nPM2.5 stats:")
        print(f"  Mean: {float(pm25.mean().values):.1f} ug/m3")
        print(f"  Min: {float(pm25.min().values):.1f} ug/m3")
        print(f"  Max: {float(pm25.max().values):.1f} ug/m3")
        print(f"  75th percentile: {float(pm25.quantile(0.75).values):.1f} ug/m3")

        # Check for Ahmedabad reasonableness
        # Ahmedabad typical PM2.5: 30-150 ug/m3 (CPCB data)
        mean_val = float(pm25.mean().values)
        max_val = float(pm25.max().values)

        if mean_val > 200:
            print("  WARNING: Mean PM2.5 very high (>200). Possible unit issue.")
        elif mean_val > 100:
            print("  WARNING: Mean PM2.5 high (>100). Check if units are ug/m3 not mg/m3.")
        elif mean_val < 10:
            print("  WARNING: Mean PM2.5 very low. Possible unit issue.")
        else:
            print("  OK: PM2.5 in reasonable range for Ahmedabad")

        if max_val > 500:
            print("  WARNING: Max PM2.5 extremely high. Check for data spikes.")

        # Check spatial
        lat_name = 'latitude' if 'latitude' in pm25.dims else 'lat'
        lon_name = 'longitude' if 'longitude' in pm25.dims else 'lon'
        print(f"\nGrid: {len(pm25[lat_name])} x {len(pm25[lon_name])}")
        print(f"  Lat: {float(pm25[lat_name].min().values):.2f} to {float(pm25[lat_name].max().values):.2f}")
        print(f"  Lon: {float(pm25[lon_name].min().values):.2f} to {float(pm25[lon_name].max().values):.2f}")

        ds.close()
    return True

def check_washington_u():
    """Validate Washington U. PM2.5 data."""
    print("\n" + "="*60)
    print("WASHINGTON U. PM2.5 VALIDATION")
    print("="*60)

    file = OUTPUT_DIR / "pm25_monthly_ahmedabad_2019_final.nc"
    if not file.exists():
        print("File not found")
        return False

    ds = xr.open_dataset(file)
    pm25 = ds['pm25']

    print(f"Dimensions: {dict(pm25.dims)}")
    print(f"Months: {len(pm25.time)}")

    print(f"\nMonthly means:")
    for i in range(len(pm25.time)):
        val = float(pm25.isel(time=i).mean().values)
        print(f"  {str(pm25.time.values[i])[:7]}: {val:.1f} ug/m3")

    annual_mean = float(pm25.mean().values)
    print(f"\nAnnual mean: {annual_mean:.1f} ug/m3")

    if annual_mean > 100:
        print("  WARNING: Annual mean very high. Check units.")
    elif annual_mean < 20:
        print("  WARNING: Annual mean low. Check units.")
    else:
        print("  OK: Annual mean reasonable")

    ds.close()
    return True

def check_spatial_alignment():
    """Check if ERA5 and CAMS grids align."""
    print("\n" + "="*60)
    print("SPATIAL ALIGNMENT CHECK")
    print("="*60)

    era5 = xr.open_dataset(OUTPUT_DIR / "era5_daily_2019.nc")
    cams = xr.open_dataset(OUTPUT_DIR / "pm25_daily_cams_ahmedabad_2019.nc")

    era5_lat = era5.latitude.values if 'latitude' in era5.dims else era5.lat.values
    era5_lon = era5.longitude.values if 'longitude' in era5.dims else era5.lon.values

    cams_lat = cams.latitude.values if 'latitude' in cams.dims else cams.lat.values
    cams_lon = cams.longitude.values if 'longitude' in cams.dims else cams.lon.values

    print(f"ERA5 lat: {era5_lat}")
    print(f"ERA5 lon: {era5_lon}")
    print(f"CAMS lat: {cams_lat}")
    print(f"CAMS lon: {cams_lon}")

    # Check overlap
    lat_overlap = set(era5_lat) & set(cams_lat)
    lon_overlap = set(era5_lon) & set(cams_lon)

    print(f"\nLat overlap: {lat_overlap}")
    print(f"Lon overlap: {lon_overlap}")

    if not lat_overlap or not lon_overlap:
        print("WARNING: No spatial overlap between ERA5 and CAMS!")
        print("  ERA5 and CAMS grids do not align.")
        print("  HPE detection will fail or give wrong results.")
    else:
        print("OK: Spatial overlap exists")

    era5.close()
    cams.close()

def main():
    print("AHMEDABAD HPE PROJECT - DATA VALIDATION")
    print("="*60)

    check_era5()
    check_cams()
    check_washington_u()
    check_spatial_alignment()

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
