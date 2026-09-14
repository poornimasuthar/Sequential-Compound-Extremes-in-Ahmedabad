#!/usr/bin/env python3
"""
Machine Learning LUR + Climate Projection Framework
Random Forest PM2.5 prediction + CMIP6 climate scenarios
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
import warnings

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = PROJECT_DIR / "figures"

print("=" * 70)
print("MACHINE LEARNING LUR + CLIMATE PROJECTION FRAMEWORK")
print("=" * 70)
print("""
This framework implements:
1. Random Forest Land Use Regression (LUR) for PM2.5
2. SHAP-based feature importance for source attribution
3. CMIP6 climate model downscaling for 2030/2050 projections
4. Scenario-based health impact assessment

METHODOLOGY OVERVIEW:
=====================
""")

# ============================================
# PART 1: RANDOM FOREST LUR
# ============================================
print("\n" + "=" * 70)
print("PART 1: RANDOM FOREST LAND USE REGRESSION")
print("=" * 70)

print("""
STEP 1: FEATURE ENGINEERING
----------------------------
Features (X) for each grid cell:
  1. Satellite AOD (aerosol optical depth) - MODIS/MISR
  2. Meteorology: Temperature, humidity, wind speed, PBL height
  3. Land Use:
     - Road density (OpenStreetMap)
     - Industrial area fraction (GHSL)
     - NDVI (vegetation)
     - Population density (WorldPop/GPW)
     - Elevation (SRTM)
     - Distance to coast
     - Distance to major roads
  4. Temporal:
     - Month of year (seasonal cycle)
     - Day of week (weekly cycle)
     - Holiday flag

Target (y):
  - Washington U. PM2.5 (1km, monthly) - training
  - Or CAMS PM2.5 (downscaled) - validation

STEP 2: MODEL ARCHITECTURE
---------------------------
Random Forest Regressor:
  - n_estimators: 500 trees
  - max_depth: 20 (prevent overfitting)
  - min_samples_leaf: 5
  - max_features: sqrt(n_features)
  - bootstrap: True

Why Random Forest?
  - Handles non-linear relationships
  - Robust to outliers
  - Provides feature importance
  - No need for feature scaling
  - Works with mixed data types

STEP 3: VALIDATION
-------------------
Spatial cross-validation:
  - K-fold by geographic clusters (not random)
  - Prevents spatial autocorrelation inflation
  - Metrics: R², RMSE, MAE, bias

STEP 4: UNCERTAINTY
-------------------
Prediction intervals:
  - Quantile Regression Forest (10th, 90th percentiles)
  - Or bootstrap aggregation of residuals
""")

# ============================================
# PART 2: SHAP ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("PART 2: SHAP-BASED SOURCE ATTRIBUTION")
print("=" * 70)

print("""
SHAP (SHapley Additive exPlanations):
--------------------------------------
For each prediction, SHAP tells you:
  "How much did each feature contribute to this prediction?"

Example output for a high-pollution cell:
  Feature          | SHAP Value | Interpretation
  -----------------|------------|--------------------------------
  Road density     | +15.2      | High traffic → +15 µg/m³
  Industrial area  | +8.7       | Factories nearby → +9 µg/m³
  AOD              | +6.3       | Satellite sees aerosols → +6
  Temperature      | -2.1       | Hot, buoyant air → -2 (disperses)
  NDVI             | -3.4       | Green space cleans air → -3

Source Apportionment:
---------------------
Aggregate SHAP values by source category:
  - Traffic: road_density + distance_to_highway
  - Industry: industrial_area + power_plant_proximity
  - Dust: elevation + bare_soil + wind_speed
  - Biomass: NDVI_seasonal + rural_population
  - Secondary: temperature + humidity + AOD (chemical formation)

Output: Pie chart of sources for each ward
""")

# ============================================
# PART 3: CMIP6 CLIMATE PROJECTIONS
# ============================================
print("\n" + "=" * 70)
print("PART 3: CMIP6 CLIMATE PROJECTIONS (2030/2050)")
print("=" * 70)

print("""
CMIP6 (Coupled Model Intercomparison Project Phase 6):
------------------------------------------------------
Provides future climate simulations from ~100 global climate models.

Scenarios (Shared Socioeconomic Pathways - SSPs):
  - SSP1-2.6: Sustainability (green transition, low emissions)
  - SSP2-4.5: Middle of the road (moderate action)
  - SSP3-7.0: Regional rivalry (high emissions, fragmented)
  - SSP5-8.5: Fossil-fueled development (worst case)

For PM2.5 projections, we need:
  1. Future meteorology (temperature, humidity, wind)
  2. Future emissions (anthropogenic + natural)
  3. Future land use (urbanization, deforestation)

DOWNSCALING METHODOLOGY:
------------------------
Bias Correction Spatial Disaggregation (BCSD):

  Step 1: Bias correction
    - Compare CMIP6 historical (1995-2014) with ERA5
    - Calculate bias: delta = CMIP6 - ERA5
    - Apply correction: CMIP6_corrected = CMIP6 - delta

  Step 2: Spatial disaggregation
    - CMIP6 resolution: ~100km
    - Target resolution: 1km (or 100m with LUR)
    - Use historical ERA5 spatial patterns as template
    - Apply future delta to high-res baseline

  Formula:
    PM2.5_future(x,y) = PM2.5_baseline(x,y) × (CMIP6_future / CMIP6_baseline)

STEP 3: EMISSION SCENARIOS
--------------------------
For Ahmedabad specifically:
  - Traffic: Scale by projected vehicle ownership (SSP-dependent)
  - Industry: Use CMIP6 emission inventories (CEDS)
  - Dust: Scale by projected aridity (precipitation changes)
  - Biomass: Assume constant or policy-driven reduction

STEP 4: HEALTH PROJECTIONS
--------------------------
For each scenario (2030, 2050):
  1. Project PM2.5 using BCSD + emission scaling
  2. Project population (UN projections)
  3. Project baseline mortality (aging population)
  4. Run IER with future PM2.5
  5. Compare to baseline (2019/2025)

Output table:
  Scenario    | Year | PM2.5 | Excess Deaths | Avoidable Deaths
  ------------|------|-------|---------------|------------------
  Baseline    | 2019 | 77.6  | 1,907         | 0
  SSP1-2.6    | 2030 | 45.2  | 1,120         | 787
  SSP1-2.6    | 2050 | 28.4  | 680           | 1,227
  SSP3-7.0    | 2030 | 89.3  | 2,340         | -433
  SSP3-7.0    | 2050 | 112.5 | 3,120         | -1,213
""")

# ============================================
# PART 4: IMPLEMENTATION SCRIPT
# ============================================
print("\n" + "=" * 70)
print("PART 4: IMPLEMENTATION")
print("=" * 70)

print("""
SCRIPTS TO CREATE:
==================

21_download_ancillary_data.py
  - Download OSM, GHSL, SRTM, MODIS NDVI
  - Preprocess to common grid

22_prepare_lur_features.py
  - Rasterize vector data
  - Compute road density, distance features
  - Stack all features into NetCDF

23_train_lur_model.py
  - Load Washington U. PM2.5 as target
  - Train Random Forest
  - Cross-validation
  - Save model (joblib)

24_shap_analysis.py
  - Compute SHAP values
  - Source apportionment
  - Ward-level aggregation

25_predict_2025.py
  - Apply trained model to 2025 features
  - Bias-correct with CAMS
  - Generate uncertainty maps

26_download_cmip6.py
  - Access ESGF (Earth System Grid Federation)
  - Download SSP1-2.6, SSP3-7.0 for India domain
  - Variables: tas, huss, sfcWind, pr

27_downscale_cmip6.py
  - BCSD methodology
  - Generate 2030/2050 PM2.5 projections

28_ier_scenarios.py
  - Run IER for each scenario
  - Population projections (UN World Population Prospects)
  - Baseline mortality projections (GBD future)

29_create_ml_figures.py
  - LUR prediction maps
  - SHAP summary plots
  - Climate scenario comparison
  - Uncertainty visualizations
  - Policy brief figure

PACKAGES TO INSTALL:
====================
pip install scikit-learn shap joblib rasterio geopandas xesmf
pip install cartopy cmocean seaborn
""")

# ============================================
# PART 5: EXPECTED OUTPUTS
# ============================================
print("\n" + "=" * 70)
print("PART 5: EXPECTED OUTPUTS FOR PHD APPLICATION")
print("=" * 70)

outputs = """
NEW FIGURES:
============
1. ML_LUR_prediction_map.png
   - 100m PM2.5 with uncertainty bounds
   - Inset: zoom to industrial zone

2. SHAP_feature_importance.png
   - Global feature importance (bar chart)
   - Local explanation for 3 example cells

3. Source_apportionment_pie.png
   - Ward-level source breakdown
   - Traffic vs Industry vs Dust vs Biomass

4. CMIP6_scenarios.png
   - Time series 2019-2050 for 4 SSPs
   - Shaded: uncertainty range

5. Health_projection_2030_2050.png
   - Excess deaths trajectory
   - Avoidable deaths with policy action

6. Uncertainty_map.png
   - 95% prediction interval width
   - Shows where model is confident vs uncertain

7. Policy_brief.png
   - One-page summary for AMC
   - Key numbers, maps, recommendations

NEW DATA PRODUCTS:
==================
- lur_model_ahmedabad.joblib (trained model)
- pm25_100m_2025_pred.tif (GeoTIFF)
- pm25_100m_2025_uncertainty.tif
- ward_source_apportionment_2025.csv
- cmip6_pm25_projections_2030_2050.nc
- scenario_health_impacts_2030_2050.csv

PUBLICATION POTENTIAL:
======================
This work could be submitted to:
- Environmental Science & Technology
- Atmospheric Environment
- Environmental Health Perspectives
- Science of the Total Environment

Novel contributions:
1. First ML-LUR for Ahmedabad at 100m
2. CMIP6 downscaling for PM2.5 in India
3. Source apportionment using SHAP
4. Policy-relevant health projections
"""

print(outputs)

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("""
For your PhD application to Yim's lab:

FOCUS ON:
---------
1. Random Forest LUR (2 weeks)
   - Shows spatial statistics + ML skills
   - Novel for Ahmedabad
   - Feasible with existing data

2. SHAP source attribution (3 days)
   - Cutting-edge explainable AI
   - Yim's lab values interpretability
   - Directly answers "where does pollution come from?"

3. Simple climate projection (1 week)
   - Download CMIP6 SSP1-2.6 and SSP3-7.0
   - Apply BCSD using existing code patterns
   - Run IER for 2030/2050
   - Shows you understand climate-health nexus

SKIP FOR NOW:
-------------
- WRF-Chem (too complex for application timeline)
- Deep learning (needs more data, GPU)
- Full CMIP6 ensemble (use 2-3 models, not 20)

This gives you:
- 3 new scripts (21, 22, 23)
- 2 new figures (SHAP, scenarios)
- 1 strong talking point for interviews
- Foundation for PhD research expansion
""")

print("\n" + "=" * 70)
print("NEXT STEP")
print("=" * 70)
print("""
Do you want me to write the implementation scripts?

Option A: Write ALL scripts (21-29) - comprehensive
Option B: Write CORE scripts only (21, 22, 23, 25, 28) - minimum viable
Option C: Start with 21_download_ancillary_data.py and see how it goes

Reply with A, B, or C, or ask questions about any part.
""")
