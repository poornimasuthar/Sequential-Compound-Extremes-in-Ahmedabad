#!/usr/bin/env python3
"""
Script 60: True PM2.5 Spatial Hotspot Identification
Identifies which wards/grid cells have the highest PM2.5 exposure directly,
without deriving it circularly from the mortality field.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

print("=" * 70)
print("TRUE PM2.5 SPATIAL HOTSPOT ANALYSIS")
print("=" * 70)

# Load Washington University 1km PM2.5 (highest spatial resolution you have)
wu = xr.open_dataset(OUTPUT_DIR / "pm25_monthly_ahmedabad_2019_final.nc")
pm25_var = "pm25" if "pm25" in wu.data_vars else list(wu.data_vars)[0]
pm25 = wu[pm25_var]

lat_name = "lat" if "lat" in pm25.dims else "latitude"
lon_name = "lon" if "lon" in pm25.dims else "longitude"

# Annual mean PM2.5 per grid cell
time_dim = [d for d in pm25.dims if d in ["time", "month"]]
pm25_annual = pm25.mean(dim=time_dim[0]) if time_dim else pm25

print(f"\nGrid: {pm25_annual[lat_name].size} x {pm25_annual[lon_name].size}")
print(f"PM2.5 annual mean range: {float(pm25_annual.min()):.1f} to {float(pm25_annual.max()):.1f} ug/m3")

# Build a flat dataframe of lat, lon, pm25
lats = pm25_annual[lat_name].values
lons = pm25_annual[lon_name].values

rows = []
for lat in lats:
    for lon in lons:
        val = float(pm25_annual.sel(**{lat_name: lat, lon_name: lon}).values)
        if not np.isnan(val):
            rows.append({"lat": lat, "lon": lon, "pm25_annual": val})

df = pd.DataFrame(rows)

# Ahmedabad's Sabarmati river runs roughly N-S; east/west split is approximate
# Use city center longitude as the dividing reference (adjust if you have exact river coords)
lon_center = df["lon"].median()
df["side"] = np.where(df["lon"] >= lon_center, "East", "West")

print(f"\nApprox east/west divide at longitude: {lon_center:.3f}")

# Top 10% hotspot cells by PM2.5
threshold_90 = df["pm25_annual"].quantile(0.90)
hotspots = df[df["pm25_annual"] >= threshold_90].copy()
hotspots = hotspots.sort_values("pm25_annual", ascending=False)

print(f"\nPM2.5 90th percentile threshold: {threshold_90:.1f} ug/m3")
print(f"Number of hotspot cells: {len(hotspots)}")

print(f"\n--- East vs West mean PM2.5 ---")
side_summary = df.groupby("side")["pm25_annual"].agg(["mean", "max", "count"])
print(side_summary)

print(f"\n--- East vs West hotspot cell counts ---")
hotspot_side = hotspots.groupby("side").size()
print(hotspot_side)

print(f"\n--- Top 15 hottest PM2.5 cells (lat, lon, value, side) ---")
print(hotspots.head(15)[["lat", "lon", "pm25_annual", "side"]].to_string(index=False))

# Save
df.to_csv(OUTPUT_DIR / "pm25_spatial_full.csv", index=False)
hotspots.to_csv(OUTPUT_DIR / "pm25_true_hotspots.csv", index=False)

print(f"\nSaved: {OUTPUT_DIR / 'pm25_spatial_full.csv'}")
print(f"Saved: {OUTPUT_DIR / 'pm25_true_hotspots.csv'}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
side_means = side_summary["mean"]
if side_means["East"] > side_means["West"]:
    print(f"PM2.5 is HIGHER in the EAST (mean {side_means['East']:.1f} vs {side_means['West']:.1f} ug/m3)")
    print("This would ALIGN with your LST paper's eastern industrial hotspots.")
else:
    print(f"PM2.5 is HIGHER in the WEST (mean {side_means['West']:.1f} vs {side_means['East']:.1f} ug/m3)")
    print("This would CONTRADICT your LST paper's eastern hotspots -")
    print("meaning heat and PM2.5 hotspots are spatially DIFFERENT, which is itself a finding.")

wu.close()
