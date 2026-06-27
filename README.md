
## Figures

| Figure | Description | File |
|--------|-------------|------|
| 1 | CAMS vs MERRA-2 daily time series | `fig1_pm25_comparison.png` |
| 2 | PM2.5 and excess deaths spatial maps | `fig2_ier_map.png` |
| 3 | IER cause-specific attributable deaths | `fig3_ier_bars.png` |
| 4 | GWR spatial coefficients and hotspots | `fig4_gwr_spatial.png` |
| 5 | Combined 4-panel publication figure | `fig5_combined_publication.png` |

## Technical Skills Demonstrated

- **Data processing:** xarray, pandas, NetCDF handling
- **Satellite data:** CAMS, MERRA-2, ERA5 APIs
- **Spatial analysis:** GWR, interpolation, grid alignment
- **Health modeling:** IER, attributable fraction, excess deaths
- **Visualization:** matplotlib, publication-quality figures
- **Version control:** Git, reproducible workflows

## Limitations & Future Work

1. **Population distribution:** Uniform grid allocation; future work will use ward-level Census 2011 data
2. **Temperature downscaling:** ERA5 coarse resolution (25 km); future work will use MODIS LST or WRF
3. **HPE detection:** 0 days found due to inverse seasonality; future work will examine compound cold-PM extremes
4. **Uncertainty:** Monte Carlo simulation for IER confidence intervals not implemented
5. **Validation:** No ground-based PM2.5 measurements available for Ahmedabad in 2019

## References

- Burnett, R.T., et al. (2014). An integrated risk function for estimating the global burden of disease attributable to ambient fine particulate matter exposure. *Environmental Health Perspectives*, 122(4), 397-403.
- van Donkelaar, A., et al. (2021). Monthly Global Estimates of Fine Particulate Matter and Their Uncertainty. *Environmental Science & Technology*.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.

## Contact

Poornima Suthar  
Email: [poornimajk2019@gmail.com]  
GitHub: [https://github.com/poornimasuthar]