Ahmedabad Heat–Pollution Extremes (HPE/SCE) Project

Compound heat–air pollution extreme event detection for Ahmedabad, India (2019), comparing a conventional simultaneous-day (HPE) framework against a sequential compound-extreme (SCE) framework that accounts for lagged and seasonal co-exposure.

Key Finding

A conventional same-day HPE definition (PM2.5 > 90th percentile AND Tmax > 90th percentile, same day) finds zero compound extreme days in Ahmedabad — not because compound risk is absent, but because Ahmedabad's pollution season (winter) and heat season (pre-monsoon summer) are largely non-overlapping ("inverse seasonality"). A sequential framework that allows a lag between a PM2.5 extreme and a later heat extreme reveals 15 events within a 90-day window, and a climatologically-motivated seasonal definition (winter PM extreme → summer heat extreme) identifies 16 events. This demonstrates that fixed-window or same-day compound-extreme definitions, standard in the literature, systematically miss compound risk in cities with strong seasonal decoupling between pollution and heat.

Repository Structure
scripts/          Numbered pipeline scripts (see "How to Reproduce" below)
scripts/diagnostics/  One-off debugging/validation scripts, not part of the main pipeline
outputs/           Generated data files, summaries, and CSVs (raw .nc files gitignored)
figures/           Generated figures (see table below)
data/              Raw downloaded data (gitignored — regenerate via download scripts)
Data Sources
PM2.5: CAMS reanalysis (pm25_daily_cams_ahmedabad_2019.nc) — used as the primary PM2.5 source throughout the current pipeline. MERRA-2 PM2.5 was also processed and is retained for comparison (pm25_daily_merra2_ahmedabad_2019.nc); the two products differ substantially in magnitude (CAMS mean ≈ 78 µg/m³ vs. MERRA-2 mean ≈ 40 µg/m³ for 2019). See outputs/agu_reanalysis_validation.csv and outputs/merra_cams_comparison.csv for the cross-validation basis for preferring CAMS.
Temperature: ERA5 daily maximum temperature (era5_daily_2019.nc)
Extreme thresholds: 90th percentile for both PM2.5 and Tmax, applied consistently across the HPE and SCE frameworks
How to Reproduce

Run the full pipeline end to end:

bash
python scripts/99_generate_all_outputs_FIXED.py

This regenerates outputs/MASTER_SUMMARY.json, outputs/complete_timeline_2019.csv, and the five current figures (see table below).

For the full standardized HPE-vs-SCE comparison, including the recommended seasonal framework:

bash
python scripts/31_standardized_hpe_sce_comparison_v2.py
python scripts/32_disentangled_mortality_v2.py
python scripts/30_sce_window_sensitivity_v2.py

Scripts numbered 50–61 cover threshold sensitivity, reanalysis validation, and spatial PM2.5 hotspot analysis. Scripts 33–35 are verification/QA checks on the main results.

Note on repository history: an earlier stage of this project (scripts 07, 09, 11, 13, 16, 17, 18, 19, 24) explored HPE detection with spatial grid alignment and an IER-based health impact / GWR spatial regression analysis. That work has been superseded by the current SCE framework and was removed from the active codebase; it remains recoverable from git history if needed.

Figures
Figure	Description	File
1	Complete PM2.5/Tmax time series with extreme-day flags	fig1_complete_timeseries.png
2	Seasonal cycle showing inverse seasonality (PM2.5 winter peak vs. heat summer peak)	fig2_seasonal_cycle.png
3	Traditional HPE vs. SCE-15/30/60/90 event counts	fig3_hpe_vs_sce_comparison.png
4	SCE-90 connections: winter PM extremes linked to later summer heat extremes	fig4_sce_connections.png
5	Summary dashboard (monthly cycles, HPE vs. SCE, distributions)	fig5_summary_dashboard.png
6	AGU threshold sensitivity analysis	agu_threshold_sensitivity.png / .pdf
7	SCE event timeline	fig8_sce_timeline.png
8	SCE window sensitivity (event count vs. window length)	fig8_window_sensitivity_v2.png
Technical Skills Demonstrated
Reanalysis data processing: xarray, pandas, NetCDF handling across CAMS, MERRA-2, and ERA5 products, including grid alignment and cross-product validation
Compound extreme detection: percentile-threshold methods, fixed-window and seasonally-anchored sequential compound extreme (SCE) frameworks
Health burden modeling: IER-based attributable mortality, disentangling chronic vs. acute exposure contributions
Visualization: matplotlib, publication-quality multi-panel figures
Version control: Git, reproducible pipeline structure, iterative cleanup of a multi-month research codebase
Limitations & Future Work
Population distribution: Uniform grid allocation for exposure estimates; ward-level Census 2011 data would improve spatial resolution of health burden estimates.
Temperature downscaling: ERA5's ~25 km resolution may smooth out urban heat island effects specific to central Ahmedabad; MODIS LST or a downscaled WRF product would help.
PM2.5 data source uncertainty: CAMS and MERRA-2 diverge by roughly 2x in mean PM2.5 for 2019; no ground-based monitoring data was available for direct validation against either product for this study year.
SCE-DLNM integration: the synergistic (interaction) mortality fraction for SCE events requires a distributed lag non-linear model (DLNM) with daily mortality data; this is flagged as future work in outputs/disentangled_mortality_v2.csv rather than estimated here.
Single-year analysis: results are based on 2019 only; multi-year analysis would clarify whether the observed inverse seasonality and SCE event counts are typical or anomalous.
References
Burnett, R.T., et al. (2014). An integrated risk function for estimating the global burden of disease attributable to ambient fine particulate matter exposure. Environmental Health Perspectives, 122(4), 397-403.
van Donkelaar, A., et al. (2021). Monthly Global Estimates of Fine Particulate Matter and Their Uncertainty. Environmental Science & Technology.
Hersbach, H., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
Contact

Poornima Suthar Email: poornimajk2019@gmail.com GitHub: https://github.com/poornimasuthar
