# Ahmedabad Heat–Pollution Extremes (HPE/SCE) Project

Compound heat–air pollution extreme event detection for Ahmedabad, India (2019), comparing a conventional simultaneous-day Heat–Pollution Extreme (HPE) framework with a Sequential Compound Extreme (SCE) framework that accounts for lagged and seasonal co-exposure.

## Key Finding

A conventional same-day HPE definition — **PM2.5 > 90th percentile AND Tmax > 90th percentile on the same day** — finds **zero compound extreme days** in Ahmedabad.

This does not necessarily indicate an absence of compound risk. Ahmedabad exhibits strong **inverse seasonality**: PM2.5 extremes are concentrated mainly during winter, while extreme heat occurs predominantly during the pre-monsoon summer.

A sequential framework that allows a lag between a PM2.5 extreme and a subsequent heat extreme identifies **15 events within a 90-day window**. A climatologically motivated seasonal definition linking winter PM2.5 extremes to subsequent summer heat extremes identifies **16 events**.

These results illustrate how same-day compound-extreme definitions can miss potentially relevant heat–pollution relationships in cities where the seasonal cycles of pollution and heat are strongly decoupled.

## Repository Structure

```text
ahmedabad_hpe_project/
├── scripts/
│   ├── diagnostics/
│   └── *.py
├── outputs/
├── figures/
├── data/
├── README.md
└── requirements.txt
```

- `scripts/` — numbered analysis and pipeline scripts.
- `scripts/diagnostics/` — debugging and validation utilities that are not part of the main production pipeline.
- `outputs/` — generated summaries, CSV files, and analysis results.
- `figures/` — generated figures used to examine the HPE/SCE framework.
- `data/` — raw downloaded datasets. Large raw NetCDF files are excluded from version control.

## Data Sources

### PM2.5

The current pipeline uses **CAMS reanalysis** as its primary PM2.5 dataset:

```text
pm25_daily_cams_ahmedabad_2019.nc
```

MERRA-2 PM2.5 was also processed for comparison:

```text
pm25_daily_merra2_ahmedabad_2019.nc
```

The two reanalysis products differ substantially in their estimated 2019 PM2.5 magnitude. The repository contains comparison and validation outputs used to assess the differences between the products.

### Temperature

Daily maximum temperature is obtained from **ERA5**:

```text
era5_daily_2019.nc
```

### Extreme Thresholds

The HPE and SCE analyses use **90th-percentile thresholds** for PM2.5 and Tmax.

## How to Reproduce

### Full Pipeline

Run the main pipeline from the repository root:

```bash
python scripts/99_generate_all_outputs_FIXED.py
```

This generates the principal analysis outputs and figures used by the project.

### Standardized HPE–SCE Comparison

For the detailed standardized comparison:

```bash
python scripts/31_standardized_hpe_sce_comparison_v2.py
python scripts/32_disentangled_mortality_v2.py
python scripts/30_sce_window_sensitivity_v2.py
```

Scripts numbered **50–61** contain threshold-sensitivity, reanalysis-validation, and spatial PM2.5 analyses.

Scripts **33–35** provide verification and quality-assurance checks for the main results.

## Repository History

Earlier stages of the project explored alternative HPE formulations, spatial grid alignment, IER-based health-impact analysis, and GWR-based spatial analysis.

The current repository focuses on the **SCE framework** and the analysis supporting the heat–pollution seasonal-decoupling finding. Superseded analysis stages were removed from the active codebase but remain recoverable through Git history.

## Figures

| Figure | Description | File |
|---|---|---|
| 1 | Complete PM2.5/Tmax time series with extreme-day flags | `fig1_complete_timeseries.png` |
| 2 | Seasonal cycle showing inverse seasonality | `fig2_seasonal_cycle.png` |
| 3 | Traditional HPE vs. SCE-15/30/60/90 event counts | `fig3_hpe_vs_sce_comparison.png` |
| 4 | SCE-90 connections between PM2.5 extremes and subsequent heat extremes | `fig4_sce_connections.png` |
| 5 | Summary dashboard | `fig5_summary_dashboard.png` |
| 6 | AGU threshold sensitivity analysis | `agu_threshold_sensitivity.png` / `agu_threshold_sensitivity.pdf` |
| 7 | SCE event timeline | `fig8_sce_timeline.png` |
| 8 | SCE window sensitivity | `fig8_window_sensitivity_v2.png` |

## Technical Skills Demonstrated

- **Reanalysis data processing:** `xarray`, `pandas`, NetCDF handling, and analysis of CAMS, MERRA-2, and ERA5 products.
- **Compound-extreme detection:** percentile-threshold methods, fixed-window analysis, and seasonally anchored sequential compound-extreme frameworks.
- **Health-burden analysis:** IER-based attributable mortality analysis and separation of chronic and acute exposure contributions.
- **Spatial and temporal analysis:** temporal event detection, seasonal analysis, reanalysis comparison, and spatial PM2.5 analysis.
- **Visualization:** `matplotlib` and publication-quality figures.
- **Reproducible research:** structured Python pipelines, Git version control, and iterative quality assurance.

## Limitations and Future Work

1. **Population distribution:** Uniform grid allocation is used for exposure estimates. Ward-level Census 2011 population data could improve the spatial resolution of health-burden estimates.

2. **Temperature resolution:** ERA5 has a spatial resolution of approximately 25 km in the dataset used here and may smooth urban-scale heat variability within Ahmedabad. Higher-resolution products such as MODIS LST or downscaled WRF simulations could provide additional spatial detail.

3. **PM2.5 data-source uncertainty:** CAMS and MERRA-2 produce substantially different PM2.5 estimates for 2019. Ground-based measurements were not available in this study for direct validation of the two reanalysis products.

4. **SCE–DLNM integration:** Estimating a synergistic mortality contribution specifically associated with sequential compound events would require a distributed lag non-linear model (DLNM) together with suitable daily mortality data.

5. **Single-year analysis:** The current analysis focuses on 2019. Multi-year analysis is required to assess whether the observed inverse seasonality and SCE event frequencies are persistent across years.

## References

- Burnett, R. T., et al. (2014). An integrated risk function for estimating the global burden of disease attributable to ambient fine particulate matter exposure. *Environmental Health Perspectives*, 122(4), 397–403.
- van Donkelaar, A., et al. (2021). Monthly Global Estimates of Fine Particulate Matter and Their Uncertainty. *Environmental Science & Technology*.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049.

## Contact

**Poornima Suthar**

Email: poornimajk2019@gmail.com

GitHub: [poornimasuthar](https://github.com/poornimasuthar)
