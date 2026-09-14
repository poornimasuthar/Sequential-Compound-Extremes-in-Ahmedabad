#!/usr/bin/env python3
"""Verify temperature units and document correct thresholds"""
import xarray as xr
from pathlib import Path

era5 = xr.open_dataset(Path("outputs/era5_daily_2019.nc"))
tmax = era5["tmax"] if "tmax" in era5 else era5["t2m"]
print(f"ERA5 tmax min: {float(tmax.min()):.1f}")
print(f"ERA5 tmax max: {float(tmax.max()):.1f}")
print(f"ERA5 tmax mean: {float(tmax.mean()):.1f}")
print(f"90th percentile: {float(tmax.quantile(0.90)):.1f}")
print("\nIf min is ~22 and max is ~44, units are CELSIUS (correct).")
print("If min is ~295 and max is ~317, units are KELVIN.")
era5.close()
