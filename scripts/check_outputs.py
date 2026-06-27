import xarray as xr
from pathlib import Path

OUTPUT_DIR = Path("outputs")

print("CHECKING ALL OUTPUT FILES")
print("=" * 60)

files_to_check = [
    "era5_daily_2019.nc",
    "pm25_monthly_ahmedabad_2019_final.nc",
    "pm25_daily_cams_ahmedabad_2019.nc",
    "pm25_3day_cams_ahmedabad_2019.nc",
    "hpe_daily_results_aligned.nc",
    "hpe_results.csv",
]

for fname in files_to_check:
    fpath = OUTPUT_DIR / fname
    if fpath.exists():
        print(f"\n✅ {fname} EXISTS ({fpath.stat().st_size / 1024:.1f} KB)")
        try:
            ds = xr.open_dataset(fpath)
            print(f"   Dims: {dict(ds.dims)}")
            ds.close()
        except Exception as e:
            print(f"   ERROR reading: {e}")
    else:
        print(f"\n❌ {fname} MISSING")

print("\n" + "=" * 60)
