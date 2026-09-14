#!/usr/bin/env python3
"""
Script 61: Ward-Level PM2.5 Hotspot Analysis
Aggregates PM2.5 to the actual 48 AMC ward boundaries, for true
comparability with the published LST study's ward-level hotspots.
"""

import geopandas as gpd
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Update this path to wherever your ward shapefile actually lives
WARD_SHAPEFILE = r"D:\Poornima Suthar\QGIS\New Project\VECTOR FINAL CRS\REPROJECTEDSHAPEFILEWARDAHMEDABAD.shp"

print("=" * 70)
print("WARD-LEVEL PM2.5 HOTSPOT ANALYSIS")
print("=" * 70)

wards = gpd.read_file(WARD_SHAPEFILE)
print(f"Loaded {len(wards)} wards")
print(f"Ward name column candidates: {list(wards.columns)}")

# Load PM2.5 raster
wu = xr.open_dataset(OUTPUT_DIR / "pm25_monthly_ahmedabad_2019_final.nc")
pm25_var = "pm25" if "pm25" in wu.data_vars else list(wu.data_vars)[0]
pm25 = wu[pm25_var]

time_dim = [d for d in pm25.dims if d in ["time", "month"]]
pm25_annual = pm25.mean(dim=time_dim[0]) if time_dim else pm25

lat_name = "lat" if "lat" in pm25_annual.dims else "latitude"
lon_name = "lon" if "lon" in pm25_annual.dims else "longitude"
pm25_annual = pm25_annual.rename({lat_name: "y", lon_name: "x"})
pm25_annual = pm25_annual.rio.write_crs("EPSG:4326")

wards = wards.to_crs("EPSG:4326")

ward_col = None
for candidate in ["WARD_NAME", "Ward_Name", "ward_name", "NAME", "name"]:
    if candidate in wards.columns:
        ward_col = candidate
        break
if ward_col is None:
    ward_col = wards.columns[0]
    print(f"WARNING: guessing ward name column = '{ward_col}', verify manually")

results = []
for idx, row in wards.iterrows():
    geom = [row.geometry.__geo_interface__]
    try:
        clipped = pm25_annual.rio.clip(geom, wards.crs, drop=True, all_touched=True)
        mean_val = float(clipped.mean(skipna=True).values)
        if not np.isnan(mean_val):
            results.append({"ward": row[ward_col], "pm25_annual": mean_val})
    except Exception as e:
        print(f"  Skipped ward {row[ward_col]}: {e}")

df = pd.DataFrame(results).sort_values("pm25_annual", ascending=False)
print(f"\nSuccessfully processed {len(df)} of {len(wards)} wards")
print("\n--- Top 15 PM2.5 wards ---")
print(df.head(15).to_string(index=False))
print("\n--- Bottom 10 PM2.5 wards ---")
print(df.tail(10).to_string(index=False))

df.to_csv(OUTPUT_DIR / "pm25_ward_level.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'pm25_ward_level.csv'}")

wu.close()
