#!/usr/bin/env python3
"""
IER Health Impact Assessment - Ahmedabad 2019
Uses Washington U. 1km PM2.5 grid (uniform with project data)
Burnett et al. (2014) IER functions
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Ahmedabad district population 2019 (Census 2011 + growth)
POPULATION = 8_450_000

# GBD 2019 baseline mortality for India/Gujarat (per 100,000)
BASELINE = {
    'ihd':    125.0,  # Ischemic heart disease
    'stroke':  95.0,  # Cerebrovascular disease
    'copd':    45.0,  # COPD
    'lc':      15.0,  # Lung cancer
    'lri':     25.0,  # Lower respiratory infection
}

# IER parameters (Burnett et al. 2014)
IER = {
    'ihd':    {'alpha': 0.847, 'gamma': 0.015, 'delta': 0.511},
    'stroke': {'alpha': 0.557, 'gamma': 0.019, 'delta': 0.514},
    'copd':   {'alpha': 0.802, 'gamma': 0.018, 'delta': 0.417},
    'lc':     {'alpha': 0.827, 'gamma': 0.015, 'delta': 0.445},
    'lri':    {'alpha': 0.749, 'gamma': 0.021, 'delta': 0.562},
}

TMREL = 7.3  # µg/m³, Burnett et al. 2014 counterfactual


def calc_rr(pm25, params):
    """IER relative risk."""
    c = np.maximum(pm25 - TMREL, 0)
    return 1 + params['alpha'] * (1 - np.exp(-params['gamma'] * (c ** params['delta'])))


def calc_af(pm25, params):
    """Attributable fraction."""
    rr = calc_rr(pm25, params)
    return (rr - 1) / rr


def main():
    print("=" * 60)
    print("IER HEALTH IMPACT - AHMEDABAD 2019")
    print("=" * 60)
    
    # Load monthly PM2.5
    fpath = OUTPUT_DIR / "pm25_monthly_ahmedabad_2019_final.nc"
    if not fpath.exists():
        print(f"ERROR: {fpath} not found")
        return
    
    ds = xr.open_dataset(fpath)
    
    # Auto-detect variable
    pm25_var = None
    for v in ds.data_vars:
        if 'pm25' in v.lower() or 'pm2' in v.lower():
            pm25_var = v
            break
    if not pm25_var:
        pm25_var = list(ds.data_vars)[0]
    
    pm25 = ds[pm25_var]
    print(f"Loaded: {pm25_var}, dims: {pm25.dims}")
    
    # Annual mean
    time_dim = 'time' if 'time' in pm25.dims else list(pm25.dims)[0]
    pm25_annual = pm25.mean(dim=time_dim)
    print(f"Annual mean PM2.5: {float(pm25_annual.mean().values):.1f} µg/m³")
    print(f"Range: {float(pm25_annual.min().values):.1f} - {float(pm25_annual.max().values):.1f} µg/m³")
    
    # Grid info
    n_cells = int(pm25_annual.size)
    pop_per_cell = POPULATION / n_cells
    print(f"Grid: {n_cells} cells, {pop_per_cell:,.0f} people/cell")
    
    # Calculate per cause
    total_excess = 0
    results = []
    datasets = {}
    
    print(f"\n{'Cause':<10} {'Baseline':>10} {'AF_mean':>10} {'Excess':>12}")
    print("-" * 50)
    
    for cause, rate in BASELINE.items():
        af = calc_af(pm25_annual, IER[cause])
        baseline = pop_per_cell * rate / 100_000
        excess = baseline * af
        excess_total = float(excess.sum().values)
        baseline_total = baseline * n_cells
        
        total_excess += excess_total
        
        results.append({
            'cause': cause.upper(),
            'baseline_deaths': baseline_total,
            'af_mean': float(af.mean().values),
            'excess_deaths': excess_total,
        })
        
        datasets[f'af_{cause}'] = af
        datasets[f'excess_{cause}'] = excess
        
        print(f"{cause.upper():<10} {baseline_total:>10.0f} {float(af.mean().values):>10.3f} {excess_total:>12.1f}")
    
    # Total excess deaths
    excess_total_grid = sum(datasets[f'excess_{c}'] for c in BASELINE)
    datasets['excess_total'] = excess_total_grid
    datasets['pm25_annual'] = pm25_annual
    datasets['population'] = xr.full_like(pm25_annual, pop_per_cell)
    
    print("-" * 50)
    print(f"{'TOTAL':<10} {'':>10} {'':>10} {total_excess:>12.1f}")
    
    # Save gridded NetCDF (for GWR)
    ds_out = xr.Dataset(datasets)
    ds_out.attrs['title'] = 'IER Health Impact - Ahmedabad 2019'
    ds_out.attrs['population_total'] = str(POPULATION)
    ds_out.attrs['tmrel'] = str(TMREL)
    ds_out.attrs['source'] = 'Burnett et al. 2014 IER'
    
    out_nc = OUTPUT_DIR / "ier_health_impact_ahmedabad_2019.nc"
    ds_out.to_netcdf(out_nc)
    print(f"\nSaved: {out_nc}")
    
    # Save summary CSV
    df = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "ier_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    
    # Key metrics
    print("\n" + "=" * 60)
    print(f"Total excess deaths (2019): {total_excess:,.0f}")
    print(f"Deaths per 100,000: {total_excess / POPULATION * 100_000:.1f}")
    print(f"% of all deaths attributable: {total_excess / sum(r['baseline_deaths'] for r in results) * 100:.1f}%")
    print("=" * 60)
    
    ds.close()


if __name__ == "__main__":
    main()

    