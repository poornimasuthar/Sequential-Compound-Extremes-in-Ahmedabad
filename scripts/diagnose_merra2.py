#!/usr/bin/env python3
import xarray as xr
from pathlib import Path

DATA_DIR = Path("data/merra2")
files = sorted(DATA_DIR.glob("*.nc4"))
files = [f for f in files if "README" not in f.name]

if not files:
    print("No files found")
    exit()

# Check first file
ds = xr.open_dataset(files[0])
print("Variables:", list(ds.data_vars))
print("Dimensions:", dict(ds.dims))

for var in ["DUSMASS25", "SSSMASS25", "BCSMASS", "OCSMASS", "SO4SMASS"]:
    if var in ds:
        da = ds[var]
        print(f"\n{var}:")
        print(f"  Dims: {da.dims}")
        for dim in da.dims:
            coord = da.coords[dim]
            print(f"  {dim}: {float(coord.min().values):.2f} to {float(coord.max().values):.2f}, n={coord.size}")
        break

ds.close()
