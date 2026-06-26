# Ahmedabad Hot-and-Polluted Episodes (HPE) Analysis

## Overview
Characterization of compound hot-and-polluted episodes in Ahmedabad, India, 
using satellite-derived PM2.5, land surface temperature (LST), and ERA5 reanalysis.

**Methodology:** Yim et al. (2025) HPE definition + GBD 2015 IER + GWR

## Project Structure

## Data Sources
| Data | Source | File |
|------|--------|------|
| PM2.5 | CAMS Reanalysis (Copernicus ADS) | `data/cams_pm25_ahmedabad_2019.nc` |
| Temperature | ERA5 (Copernicus CDS) | `data/era5_ahmedabad_2019.nc` |
| LST | Landsat 8/9 (Google Earth Engine) | `data/Ahmedabad_LST_*.tif` |
| Population | WorldPop 2020 (GEE) | `data/Ahmedabad_Population.tif` |
| Mortality | GBD 2019 (IHME) | `data/gbd_2019_india.csv` |

## Installation
```bash
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python scripts/01_download_era5.py
python scripts/02_download_cams.py

**Save** (Ctrl+S)

---

## 2. Create `requirements.txt`

**In VS Code Terminal (Command Prompt with venv):**
```cmd
cd C:\Users\Poornima Suthar\ahmedabad_hpe_project
venv\Scripts\activate.bat
pip freeze > requirements.txt

