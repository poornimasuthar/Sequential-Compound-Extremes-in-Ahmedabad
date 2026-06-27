#!/usr/bin/env python3
"""
Ahmedabad HPE Project - MERRA-2 Batch Download
Author: Poornima Suthar
Date: 2026-06-27

Downloads MERRA-2 daily subset files from NASA GES DISC.
Requires NASA Earthdata account and .netrc file.
"""

import os
import requests
from pathlib import Path
from urllib.parse import urlparse

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "merra2"
LINKS_FILE = Path(input("Enter path to MERRA-2 links file (e.g., C:/Users/.../links.txt): ").strip().replace('"', ''))

def download_file(url, output_dir, timeout=120):
    """Download a single file with NASA Earthdata auth."""

    filename = os.path.basename(urlparse(url).path)
    output_path = output_dir / filename

    if output_path.exists():
        print(f"  SKIP (exists): {filename}")
        return True

    print(f"  DOWNLOAD: {filename}")

    try:
        # Uses .netrc for NASA Earthdata authentication
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"  DONE: {filename} ({output_path.stat().st_size / 1024:.1f} KB)")
        return True

    except Exception as e:
        print(f"  ERROR: {filename} - {e}")
        return False

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Read links
    with open(LINKS_FILE, 'r') as f:
        links = [line.strip() for line in f if line.strip() and line.strip().startswith('http')]

    print(f"Found {len(links)} files to download")
    print(f"Saving to: {DATA_DIR}")
    print(f"\nNOTE: Make sure you have a .netrc file with NASA Earthdata credentials")
    print(f"      Location: %USERPROFILE%\.netrc")
    print(f"      Format: machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASS")
    print()

    # Download all
    success = 0
    failed = 0

    for i, url in enumerate(links, 1):
        print(f"[{i}/{len(links)}]", end="")
        if download_file(url, DATA_DIR):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Success: {success}/{len(links)}")
    print(f"Failed: {failed}/{len(links)}")
    print(f"Files in: {DATA_DIR}")

if __name__ == "__main__":
    main()
