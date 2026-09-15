#!/usr/bin/env python3
"""
Quick status check - what do we already have?
"""
from pathlib import Path
import json

print("=" * 60)
print("AHMEDABAD HPE PROJECT - STATUS CHECK")
print("=" * 60)

base = Path(".")

# Check processed data
processed = base / "data" / "processed"
files_to_check = [
    "merra2_ahmedabad_2019.nc",
    "era5_ahmedabad_2019.nc", 
    "cams_ahmedabad_2019.nc",
    "washington_u_pm25_2019.nc",
]

print("\n[1] PROCESSED DATA FILES:")
for f in files_to_check:
    path = processed / f
    status = "✅ EXISTS" if path.exists() else "❌ MISSING"
    size = f"{path.stat().st_size / (1024*1024):.1f} MB" if path.exists() else ""
    print(f"  {f:40s} {status} {size}")

# Check if IER already run
ier_dir = processed / "ier_results"
print(f"\n[2] IER RESULTS:")
if ier_dir.exists():
    for f in ier_dir.glob("*"):
        print(f"  ✅ {f.name}")
else:
    print("  ❌ Not yet run")

# Check if SCE already run  
sce_dir = processed / "sce_results"
print(f"\n[3] SCE RESULTS:")
if sce_dir.exists():
    for f in sce_dir.glob("*"):
        print(f"  ✅ {f.name}")
else:
    print("  ❌ Not yet run")

# Check figures
fig_dir = base / "figures"
print(f"\n[4] FIGURES:")
if fig_dir.exists():
    figs = list(fig_dir.glob("*.png"))
    if figs:
        for f in figs:
            print(f"  ✅ {f.name}")
    else:
        print("  ⚠️  Folder exists but empty")
else:
    print("  ❌ Not yet generated")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)

# Recommend what to run next
if not ier_dir.exists():
    print("  → Run: python scripts/15_ier_health_impact.py")
if not sce_dir.exists():
    print("  → Run: python scripts/17_sequential_compound_extreme.py")
if not fig_dir.exists() or not list(fig_dir.glob("*.png")):
    print("  → Run: python scripts/18_publication_figures.py")

print("=" * 60)
