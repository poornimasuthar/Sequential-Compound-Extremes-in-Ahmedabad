#!/usr/bin/env python3
"""
Retry MERRA-2 downloads - reads .netrc for NASA Earthdata auth.
Skips existing files. Saves actual .nc4 files (not HTML redirects).
"""
import os
import sys
import time
import netrc
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "merra2"
LINKS_FILE = Path.home() / "Downloads" / "subset_M2T1NXAER_5.12.4_20260626_204418_.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Read NASA credentials from .netrc
try:
    nrc = netrc.netrc()
    username, _, password = nrc.authenticators("urs.earthdata.nasa.gov")
    print(f"Loaded credentials for: {username}")
except Exception as e:
    print(f"ERROR: Could not read .netrc: {e}")
    print("Make sure .netrc exists at C:\\Users\\Poornima Suthar\\.netrc")
    sys.exit(1)

# Session with retries and auth
session = requests.Session()
session.auth = (username, password)

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[401, 429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Read links
with open(LINKS_FILE, 'r') as f:
    urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

print(f"Total links in file: {len(urls)}")
print(f"Saving to: {DATA_DIR}")
print("=" * 60)

success = 0
failed = 0
skipped = 0

for i, url in enumerate(urls, 1):
    # Extract filename from URL
    parsed = urlparse(url)
    # The actual filename is in the FILENAME parameter
    if "FILENAME=" in url:
        fname = url.split("FILENAME=")[1].split("&")[0].split("/")[-1]
    else:
        fname = parsed.path.split("/")[-1]
    
    if not fname.endswith(".nc4"):
        fname += ".nc4"
    
    out_path = DATA_DIR / fname
    
    # Skip if exists and non-empty
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"[{i}/{len(urls)}] SKIP (exists): {fname}")
        skipped += 1
        continue
    
    print(f"[{i}/{len(urls)}] DOWNLOAD: {fname}")
    
    try:
        # First request - NASA will redirect to auth, then to data
        response = session.get(url, stream=True, timeout=60, allow_redirects=True)
        response.raise_for_status()
        
        # Check if we got HTML instead of NetCDF
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type or response.content[:100].startswith(b'<'):
            print(f"  WARNING: Got HTML, not NetCDF. Auth may have failed.")
            failed += 1
            continue
        
        # Save file
        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        size_kb = out_path.stat().st_size / 1024
        print(f"  DONE: {fname} ({size_kb:.1f} KB)")
        success += 1
        
        # Small delay to be polite
        time.sleep(0.5)
        
    except Exception as e:
        print(f"  ERROR: {e}")
        failed += 1
        # Clean up partial file
        if out_path.exists():
            out_path.unlink()

print("=" * 60)
print(f"Complete: Success={success}, Failed={failed}, Skipped={skipped}")
print(f"Files in: {DATA_DIR}")

