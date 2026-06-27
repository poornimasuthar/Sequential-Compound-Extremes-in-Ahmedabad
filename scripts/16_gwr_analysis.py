#!/usr/bin/env python3
"""
Geographically Weighted Regression (GWR) for Ahmedabad PM2.5 Health Impact
Handles constant variables (temperature, population) by dropping them
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_DIR = PROJECT_DIR / "data"

try:
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW
    HAS_MGWR = True
except ImportError:
    HAS_MGWR = False
    print("WARNING: mgwr not installed. Using numpy fallback.")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not installed. Using pure numpy regression.")


def find_era5_file():
    """Auto-detect ERA5 temperature file."""
    for pattern in [
        OUTPUT_DIR.glob("era5*.nc"),
        DATA_DIR.glob("era5*.nc"),
        OUTPUT_DIR.glob("*t2m*.nc"),
        DATA_DIR.glob("*t2m*.nc"),
    ]:
        files = list(pattern)
        if files:
            return files[0]
    return None


def get_dim_names(da):
    """Get lat/lon dimension names from a DataArray."""
    for lat_candidate in ['lat', 'latitude', 'y', 'Lat', 'Latitude']:
        if lat_candidate in da.dims:
            lat_name = lat_candidate
            break
    else:
        lat_name = da.dims[-2] if len(da.dims) >= 2 else da.dims[0]
    
    for lon_candidate in ['lon', 'longitude', 'x', 'Lon', 'Longitude']:
        if lon_candidate in da.dims:
            lon_name = lon_candidate
            break
    else:
        lon_name = da.dims[-1] if len(da.dims) >= 2 else da.dims[0]
    
    return lat_name, lon_name


def interpolate_era5_to_washingtonu_grid():
    """Interpolate ERA5 temperature to Washington U. grid."""
    print("=" * 60)
    print("INTERPOLATING ERA5 TEMPERATURE TO 1KM GRID")
    print("=" * 60)
    
    wu = xr.open_dataset(OUTPUT_DIR / "pm25_monthly_ahmedabad_2019_final.nc")
    pm25 = wu["pm25"] if "pm25" in wu else list(wu.data_vars)[0]
    wu_lat_name, wu_lon_name = get_dim_names(pm25)
    wu_lats = pm25[wu_lat_name].values
    wu_lons = pm25[wu_lon_name].values
    
    era5_path = find_era5_file()
    
    if era5_path is None:
        print("WARNING: No ERA5 file found. Using constant temperature (300K).")
        t2m_interp = xr.DataArray(
            np.full((len(wu_lats), len(wu_lons)), 300.0),
            coords=[wu_lats, wu_lons],
            dims=[wu_lat_name, wu_lon_name]
        )
        wu.close()
        return t2m_interp
    
    print(f"Found ERA5 file: {era5_path.name}")
    
    era5 = xr.open_dataset(era5_path)
    
    t2m_var = None
    for v in era5.data_vars:
        vlower = v.lower()
        if any(x in vlower for x in ['t2m', 'temp', 'tmax', 'tmin', 'tmean']):
            t2m_var = v
            break
    if t2m_var is None:
        t2m_var = list(era5.data_vars)[0]
    
    print(f"  Using temperature variable: {t2m_var}")
    
    t2m = era5[t2m_var]
    era5_lat_name, era5_lon_name = get_dim_names(t2m)
    
    time_dims = [d for d in t2m.dims if d in ['time', 'valid_time', 'date', 'day']]
    time_dim = time_dims[0] if time_dims else None
    
    if time_dim:
        t2m_annual = t2m.mean(dim=time_dim)
    else:
        t2m_annual = t2m
    
    try:
        target_coords = {era5_lat_name: wu_lats, era5_lon_name: wu_lons}
        t2m_interp = t2m_annual.reindex(**target_coords, method="nearest")
        if np.isnan(t2m_interp.values).all():
            raise ValueError("All NaN after reindex")
    except Exception as e:
        print(f"  Reindex failed ({e}), using manual broadcast")
        t2m_mean = float(t2m_annual.mean().values)
        if np.isnan(t2m_mean):
            t2m_mean = 300.0
        print(f"  Using uniform temperature: {t2m_mean:.1f} K")
        t2m_interp = xr.DataArray(
            np.full((len(wu_lats), len(wu_lons)), t2m_mean),
            coords=[wu_lats, wu_lons],
            dims=[wu_lat_name, wu_lon_name]
        )
    
    rename_dict = {}
    if era5_lat_name != wu_lat_name and era5_lat_name in t2m_interp.dims:
        rename_dict[era5_lat_name] = wu_lat_name
    if era5_lon_name != wu_lon_name and era5_lon_name in t2m_interp.dims:
        rename_dict[era5_lon_name] = wu_lon_name
    if rename_dict:
        t2m_interp = t2m_interp.rename(rename_dict)
    
    print(f"Final grid: {t2m_interp[wu_lat_name].size} x {t2m_interp[wu_lon_name].size}")
    print(f"Annual mean T2m: {float(t2m_interp.mean().values):.1f} K")
    print(f"Range: {float(t2m_interp.min().values):.1f} - {float(t2m_interp.max().values):.1f} K")
    
    era5.close()
    wu.close()
    
    return t2m_interp


def prepare_gwr_data():
    """Prepare dataset for GWR."""
    print("\n" + "=" * 60)
    print("PREPARING GWR DATASET")
    print("=" * 60)
    
    ier = xr.open_dataset(OUTPUT_DIR / "ier_health_impact_ahmedabad_2019.nc")
    wu = xr.open_dataset(OUTPUT_DIR / "pm25_monthly_ahmedabad_2019_final.nc")
    
    pm25_var = [v for v in wu.data_vars if 'pm25' in v.lower() or 'pm2' in v.lower()][0]
    pm25_monthly = wu[pm25_var]
    
    time_dim = [d for d in pm25_monthly.dims if d in ['time', 'month']][0] if any(d in pm25_monthly.dims for d in ['time', 'month']) else pm25_monthly.dims[0]
    pm25_annual = pm25_monthly.mean(dim=time_dim)
    
    t2m_interp = interpolate_era5_to_washingtonu_grid()
    
    excess_total = ier["excess_total"] if "excess_total" in ier else ier["excess_deaths_total"]
    pop = ier["population"] if "population" in ier else xr.full_like(excess_total, 8450000.0 / excess_total.size)
    
    lat_name = "lat" if "lat" in excess_total.dims else "latitude"
    lon_name = "lon" if "lon" in excess_total.dims else "longitude"
    
    lats, lons, y_vals, x1_vals, x2_vals, x3_vals = [], [], [], [], [], []
    
    for lat in excess_total[lat_name].values:
        for lon in excess_total[lon_name].values:
            try:
                y = float(excess_total.sel(**{lat_name: lat, lon_name: lon}).values)
                x1 = float(pm25_annual.sel(**{lat_name: lat, lon_name: lon}).values)
                x2 = float(t2m_interp.sel(**{lat_name: lat, lon_name: lon}).values)
                x3 = float(pop.sel(**{lat_name: lat, lon_name: lon}).values)
                
                if not (np.isnan(y) or np.isnan(x1) or np.isnan(x2)):
                    lats.append(lat)
                    lons.append(lon)
                    y_vals.append(y)
                    x1_vals.append(x1)
                    x2_vals.append(x2)
                    x3_vals.append(x3)
            except:
                pass
    
    df = pd.DataFrame({
        'lat': lats, 'lon': lons,
        'excess_deaths': y_vals,
        'pm25_annual': x1_vals,
        't2m_annual': x2_vals,
        'population': x3_vals
    })
    
    print(f"\nGWR dataset: {len(df)} valid grid cells")
    if len(df) > 0:
        print(f"Excess deaths: {df['excess_deaths'].min():.2f} - {df['excess_deaths'].max():.2f}")
        print(f"PM2.5: {df['pm25_annual'].min():.1f} - {df['pm25_annual'].max():.1f} ug/m3")
        print(f"T2m: {df['t2m_annual'].min():.1f} - {df['t2m_annual'].max():.1f} K")
        print(f"Population: {df['population'].min():.0f} - {df['population'].max():.0f}")
    
    ier.close()
    wu.close()
    
    return df


def numpy_ols(X, y):
    """Pure numpy OLS with pseudo-inverse for singular matrices."""
    # Use pinv instead of inv to handle singular matrices
    beta = np.linalg.pinv(X) @ y
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta, y_pred, r2


def run_basic_local_regression(df):
    """Fallback spatial regression with automatic variable selection."""
    print("\n" + "=" * 60)
    print("RUNNING BASIC SPATIAL REGRESSION")
    print("=" * 60)
    
    # Check which variables actually vary
    var_names = []
    if df['pm25_annual'].std() > 0.01:
        var_names.append('pm25_annual')
    if df['t2m_annual'].std() > 0.01:
        var_names.append('t2m_annual')
    if df['population'].std() > 0.01:
        var_names.append('population')
    
    print(f"Variables with variation: {var_names}")
    
    if len(var_names) == 0:
        print("ERROR: No variables vary across space. Cannot run regression.")
        df['predicted_global'] = df['excess_deaths'].mean()
        df['residuals'] = df['excess_deaths'] - df['predicted_global']
        df['hotspot'] = df['residuals'] > df['residuals'].quantile(0.9)
        df.to_csv(OUTPUT_DIR / "gwr_results.csv", index=False)
        return df
    
    lat_c = df['lat'].mean()
    lon_c = df['lon'].mean()
    
    # Build X matrix with only varying variables
    X_raw = df[var_names].values
    y = df['excess_deaths'].values
    
    # Check rank
    X_global = np.column_stack([np.ones(len(X_raw)), X_raw])
    rank = np.linalg.matrix_rank(X_global)
    
    if rank < X_global.shape[1]:
        print(f"WARNING: Matrix rank {rank} < {X_global.shape[1]} columns. Using pseudo-inverse.")
        use_pinv = True
    else:
        use_pinv = False
    
    if HAS_SKLEARN and not use_pinv:
        model_global = LinearRegression()
        model_global.fit(X_raw, y)
        y_pred_global = model_global.predict(X_raw)
        r2_global = r2_score(y, y_pred_global)
        coefs = model_global.coef_
        intercept = model_global.intercept_
    else:
        beta, y_pred_global, r2_global = numpy_ols(X_global, y)
        intercept = beta[0]
        coefs = beta[1:]
    
    print(f"\nGlobal OLS Results:")
    print(f"  R2: {r2_global:.3f}")
    for i, name in enumerate(var_names):
        print(f"  {name} coef: {coefs[i]:+.4f}")
    print(f"  Intercept: {intercept:.4f}")
    
    # Quadrant analysis
    print(f"\nSpatial Variation (Quadrant Analysis):")
    df['quadrant'] = (df['lat'] > lat_c).astype(int) * 2 + (df['lon'] > lon_c).astype(int)
    
    for q in sorted(df['quadrant'].unique()):
        sub = df[df['quadrant'] == q]
        if len(sub) > 5:
            X_q_raw = sub[var_names].values
            y_q = sub['excess_deaths'].values
            
            if HAS_SKLEARN:
                model_q = LinearRegression()
                model_q.fit(X_q_raw, y_q)
                y_pred_q = model_q.predict(X_q_raw)
                r2_q = r2_score(y_q, y_pred_q)
                pm25_coef = model_q.coef_[0] if 'pm25_annual' in var_names else 0
            else:
                X_q = np.column_stack([np.ones(len(X_q_raw)), X_q_raw])
                beta_q, y_pred_q, r2_q = numpy_ols(X_q, y_q)
                pm25_coef = beta_q[1] if 'pm25_annual' in var_names else 0
            
            print(f"  Q{q} (n={len(sub)}): R2={r2_q:.3f}, PM2.5={pm25_coef:+.4f}")
    
    df['predicted_global'] = y_pred_global
    df['residuals'] = y - y_pred_global
    df['hotspot'] = df['residuals'] > df['residuals'].quantile(0.9)
    
    print(f"\nHotspot cells (top 10% residuals): {df['hotspot'].sum()}")
    
    df.to_csv(OUTPUT_DIR / "gwr_results.csv", index=False)
    print(f"\nSaved: gwr_results.csv")
    
    return df


def run_mgwr(df):
    """Run MGWR if available."""
    print("\n" + "=" * 60)
    print("RUNNING MGWR")
    print("=" * 60)
    
    # Only use varying variables
    var_names = []
    if df['pm25_annual'].std() > 0.01:
        var_names.append('pm25_annual')
    if df['t2m_annual'].std() > 0.01:
        var_names.append('t2m_annual')
    if df['population'].std() > 0.01:
        var_names.append('population')
    
    coords = np.column_stack([df['lon'].values, df['lat'].values])
    y = df['excess_deaths'].values.reshape(-1, 1)
    X = df[var_names].values
    X = np.column_stack([np.ones(len(X)), X])
    
    print("Selecting optimal bandwidth...")
    selector = Sel_BW(coords, y, X)
    bw = selector.search()
    print(f"Optimal bandwidth: {bw:.1f}")
    
    print("Fitting GWR model...")
    model = GWR(coords, y, X, bw)
    results = model.fit()
    
    print(f"\nGWR Results:")
    print(f"  R2: {results.R2:.3f}")
    print(f"  Adj R2: {results.adj_R2:.3f}")
    
    params = results.params
    print(f"\nParameter ranges:")
    print(f"  Intercept: {params[:,0].min():.2f} to {params[:,0].max():.2f}")
    for i, name in enumerate(var_names):
        print(f"  {name}: {params[:,i+1].min():+.4f} to {params[:,i+1].max():+.4f}")
    
    df['gwr_intercept'] = params[:,0]
    for i, name in enumerate(var_names):
        df[f'gwr_{name}_coef'] = params[:,i+1]
    df['gwr_predicted'] = results.predy.flatten()
    df['gwr_residuals'] = results.resid_response.flatten()
    
    df.to_csv(OUTPUT_DIR / "gwr_results.csv", index=False)
    print(f"\nSaved: gwr_results.csv")
    
    return df


def main():
    print("=" * 60)
    print("GEOGRAPHICALLY WEIGHTED REGRESSION - AHMEDABAD 2019")
    print("=" * 60)
    
    df = prepare_gwr_data()
    
    if len(df) == 0:
        print("ERROR: No valid data for GWR")
        return
    
    if HAS_MGWR and len(df) > 20:
        df = run_mgwr(df)
    else:
        df = run_basic_local_regression(df)
    
    print("\n" + "=" * 60)
    print("GWR COMPLETE")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR / 'gwr_results.csv'}")
    print("\nNext: Run scripts/17_create_figures.py")


if __name__ == "__main__":
    main()
    