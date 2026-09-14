#!/usr/bin/env python3
"""
Create publication-quality figures for Ahmedabad HPE project
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150


def get_time_values(da):
    """Extract time values from a DataArray, handling different coordinate names."""
    for coord_name in ['time', 'valid_time', 'date', 'day']:
        if coord_name in da.coords:
            return da[coord_name]
    # If no time coordinate, create a range
    return pd.date_range('2019-01-01', periods=len(da), freq='D')


def figure1_pm25_comparison():
    """Panel A: CAMS vs MERRA-2 daily time series"""
    print("Creating Figure 1: PM2.5 Comparison...")
    
    merra = xr.open_dataset(OUTPUT_DIR / "pm25_daily_merra2_ahmedabad_2019.nc")
    cams = xr.open_dataset(OUTPUT_DIR / "pm25_daily_cams_ahmedabad_2019.nc")
    
    merra_pm = merra["pm25"]
    cams_pm = cams["pm25"].mean(dim=["latitude", "longitude"])
    
    # Get time values
    merra_time = get_time_values(merra_pm)
    cams_time = get_time_values(cams_pm)
    
    # Convert to pandas Series for rolling
    merra_series = pd.Series(merra_pm.values, index=pd.to_datetime(merra_time.values))
    cams_series = pd.Series(cams_pm.values, index=pd.to_datetime(cams_time.values))
    
    # Align to common period
    start = max(merra_series.index.min(), cams_series.index.min())
    end = min(merra_series.index.max(), cams_series.index.max())
    
    merra_sub = merra_series.loc[start:end]
    cams_sub = cams_series.loc[start:end]
    
    # 7-day rolling mean
    merra_7d = merra_sub.rolling(7, center=True).mean()
    cams_7d = cams_sub.rolling(7, center=True).mean()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(merra_sub.index, merra_sub.values, label='MERRA-2', color='#2E86AB', alpha=0.4, linewidth=0.8)
    ax.plot(cams_sub.index, cams_sub.values, label='CAMS', color='#A23B72', alpha=0.4, linewidth=0.8)
    ax.plot(merra_7d.index, merra_7d.values, color='#2E86AB', linewidth=2.5, label='MERRA-2 (7-day)')
    ax.plot(cams_7d.index, cams_7d.values, color='#A23B72', linewidth=2.5, label='CAMS (7-day)')
    
    ax.axhline(y=77.6, color='#A23B72', linestyle='--', alpha=0.5)
    ax.axhline(y=40.3, color='#2E86AB', linestyle='--', alpha=0.5)
    ax.text(pd.Timestamp('2019-12-15'), 82, 'CAMS mean: 77.6', ha='right', color='#A23B72', fontsize=9)
    ax.text(pd.Timestamp('2019-12-15'), 45, 'MERRA-2 mean: 40.3', ha='right', color='#2E86AB', fontsize=9)
    
    ax.set_ylabel('PM$_{2.5}$ (µg m$^{-3}$)')
    ax.set_xlabel('Month (2019)')
    ax.set_title('(a) Daily PM$_{2.5}$ Time Series: CAMS vs MERRA-2', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 200)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_pm25_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'fig1_pm25_comparison.png'}")
    
    merra.close()
    cams.close()
    plt.close()


def figure2_ier_map():
    """Panel B: IER excess deaths spatial map"""
    print("Creating Figure 2: IER Health Impact Map...")
    
    ds = xr.open_dataset(OUTPUT_DIR / "ier_health_impact_ahmedabad_2019.nc")
    
    excess = ds["excess_total"] if "excess_total" in ds else ds["excess_deaths_total"]
    pm25 = ds["pm25_annual"]
    
    lat_name = "lat" if "lat" in pm25.dims else "latitude"
    lon_name = "lon" if "lon" in pm25.dims else "longitude"
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].pcolormesh(pm25[lon_name], pm25[lat_name], pm25, 
                             cmap='YlOrRd', shading='auto', vmin=40, vmax=65)
    axes[0].set_title('(b) Annual Mean PM$_{2.5}$ (µg m$^{-3}$)', fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], label='PM$_{2.5}$ (µg m$^{-3}$)', fraction=0.046)
    
    im2 = axes[1].pcolormesh(excess[lon_name], excess[lat_name], excess, 
                             cmap='Reds', shading='auto', vmin=1.9, vmax=2.4)
    axes[1].set_title('(c) Annual Excess Deaths per Grid Cell', fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], label='Excess deaths', fraction=0.046)
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_ier_map.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'fig2_ier_map.png'}")
    
    ds.close()
    plt.close()


def figure3_ier_bars():
    """Panel C: IER cause-specific excess deaths"""
    print("Creating Figure 3: IER Cause Breakdown...")
    
    df = pd.read_csv(OUTPUT_DIR / "ier_summary.csv")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#E9C46A']
    bars = ax.barh(df['cause'], df['excess_deaths'], color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, df['excess_deaths']):
        ax.text(val + 15, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Annual Excess Deaths', fontsize=12)
    ax.set_title('(d) IER Attributable Deaths by Cause (Ahmedabad, 2019)', fontweight='bold')
    ax.set_xlim(0, df['excess_deaths'].max() * 1.25)
    ax.grid(axis='x', alpha=0.3)
    
    total = df['excess_deaths'].sum()
    ax.text(0.98, 0.02, f'Total: {total:.0f} deaths\n({total/8450000*100000:.1f} per 100,000)', 
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='black'))
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_ier_bars.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'fig3_ier_bars.png'}")
    plt.close()


def figure4_gwr_spatial():
    """Panel D: GWR spatial analysis"""
    print("Creating Figure 4: GWR Spatial Analysis...")
    
    gwr_file = OUTPUT_DIR / "gwr_results.csv"
    if not gwr_file.exists():
        print("  WARNING: gwr_results.csv not found. Skipping.")
        return
    
    df = pd.read_csv(gwr_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # PM2.5 concentration
    sc1 = axes[0].scatter(df['lon'], df['lat'], c=df['pm25_annual'], 
                          cmap='YlOrRd', s=80, edgecolors='black', linewidth=0.5)
    axes[0].set_title('(e) PM$_{2.5}$ Concentration', fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(sc1, ax=axes[0], label='µg m$^{-3}$', fraction=0.046)
    
    # Residuals
    resid_col = 'residuals' if 'residuals' in df.columns else 'gwr_residuals'
    vmax = abs(df[resid_col]).max()
    sc2 = axes[1].scatter(df['lon'], df['lat'], c=df[resid_col], 
                          cmap='RdBu_r', s=80, edgecolors='black', linewidth=0.5,
                          vmin=-vmax, vmax=vmax)
    axes[1].set_title('(f) GWR Residuals', fontweight='bold')
    axes[1].set_xlabel('Longitude')
    plt.colorbar(sc2, ax=axes[1], label='Residuals', fraction=0.046)
    
    # Hotspots
    if 'hotspot' in df.columns:
        colors = ['#2A9D8F' if not h else '#E63946' for h in df['hotspot']]
        sc3 = axes[2].scatter(df['lon'], df['lat'], c=colors, s=80, 
                              edgecolors='black', linewidth=0.5)
        axes[2].set_title(f'(g) Health Risk Hotspots (n={df["hotspot"].sum()})', fontweight='bold')
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2A9D8F', edgecolor='black', label='Normal'),
                          Patch(facecolor='#E63946', edgecolor='black', label='Hotspot')]
        axes[2].legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_gwr_spatial.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'fig4_gwr_spatial.png'}")
    plt.close()


def figure5_combined():
    """Combined 4-panel figure for publication"""
    print("Creating Figure 5: Combined Publication Figure...")
    
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Panel A: Time series
    ax1 = fig.add_subplot(gs[0, :])
    
    merra = xr.open_dataset(OUTPUT_DIR / "pm25_daily_merra2_ahmedabad_2019.nc")
    cams = xr.open_dataset(OUTPUT_DIR / "pm25_daily_cams_ahmedabad_2019.nc")
    merra_pm = merra["pm25"]
    cams_pm = cams["pm25"].mean(dim=["latitude", "longitude"])
    
    merra_time = get_time_values(merra_pm)
    cams_time = get_time_values(cams_pm)
    
    merra_series = pd.Series(merra_pm.values, index=pd.to_datetime(merra_time.values))
    cams_series = pd.Series(cams_pm.values, index=pd.to_datetime(cams_time.values))
    
    start = max(merra_series.index.min(), cams_series.index.min())
    end = min(merra_series.index.max(), cams_series.index.max())
    
    merra_sub = merra_series.loc[start:end]
    cams_sub = cams_series.loc[start:end]
    
    merra_7d = merra_sub.rolling(7, center=True).mean()
    cams_7d = cams_sub.rolling(7, center=True).mean()
    
    ax1.fill_between(cams_sub.index, cams_sub.values, alpha=0.15, color='#A23B72')
    ax1.fill_between(merra_sub.index, merra_sub.values, alpha=0.15, color='#2E86AB')
    ax1.plot(cams_7d.index, cams_7d.values, color='#A23B72', linewidth=2.5, label='CAMS (7-day mean)')
    ax1.plot(merra_7d.index, merra_7d.values, color='#2E86AB', linewidth=2.5, label='MERRA-2 (7-day mean)')
    ax1.axhline(y=77.6, color='#A23B72', linestyle='--', alpha=0.5)
    ax1.axhline(y=40.3, color='#2E86AB', linestyle='--', alpha=0.5)
    ax1.set_ylabel('PM$_{2.5}$ (µg m$^{-3}$)', fontsize=12)
    ax1.set_title('(a) Daily PM$_{2.5}$ Time Series Comparison', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 180)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.grid(alpha=0.3)
    
    merra.close()
    cams.close()
    
    # Panel B: PM2.5 map
    ax2 = fig.add_subplot(gs[1, 0])
    ds = xr.open_dataset(OUTPUT_DIR / "ier_health_impact_ahmedabad_2019.nc")
    pm25 = ds["pm25_annual"]
    lat_name = "lat" if "lat" in pm25.dims else "latitude"
    lon_name = "lon" if "lon" in pm25.dims else "longitude"
    im = ax2.pcolormesh(pm25[lon_name], pm25[lat_name], pm25, cmap='YlOrRd', shading='auto', vmin=40, vmax=65)
    ax2.set_title('(b) Annual Mean PM$_{2.5}$ (µg m$^{-3}$)', fontweight='bold', fontsize=13)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax2, fraction=0.046, label='µg m$^{-3}$')
    ds.close()
    
    # Panel C: Excess deaths map
    ax3 = fig.add_subplot(gs[1, 1])
    ds = xr.open_dataset(OUTPUT_DIR / "ier_health_impact_ahmedabad_2019.nc")
    excess = ds["excess_total"] if "excess_total" in ds else ds["excess_deaths_total"]
    im2 = ax3.pcolormesh(excess[lon_name], excess[lat_name], excess, cmap='Reds', shading='auto', vmin=1.9, vmax=2.4)
    ax3.set_title('(c) Annual Excess Deaths per Grid Cell', fontweight='bold', fontsize=13)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    plt.colorbar(im2, ax=ax3, fraction=0.046, label='Deaths')
    ds.close()
    
    # Panel D: IER bars
    ax4 = fig.add_subplot(gs[2, 0])
    df = pd.read_csv(OUTPUT_DIR / "ier_summary.csv")
    colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#E9C46A']
    bars = ax4.barh(df['cause'], df['excess_deaths'], color=colors, edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars, df['excess_deaths']):
        ax4.text(val + 15, bar.get_y() + bar.get_height()/2, f'{val:.0f}', va='center', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Annual Excess Deaths', fontsize=12)
    ax4.set_title('(d) Attributable Deaths by Cause', fontweight='bold', fontsize=13)
    ax4.set_xlim(0, df['excess_deaths'].max() * 1.3)
    ax4.grid(axis='x', alpha=0.3)
    
    # Panel E: Key metrics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    metrics_text = f"""KEY FINDINGS — AHMEDABAD 2019

PM$_{2.5}$ EXPOSURE
  • CAMS annual mean: 77.6 µg m$^{{-3}}$
  • MERRA-2 annual mean: 40.3 µg m$^{{-3}}$
  • WHO guideline: 5 µg m$^{{-3}}$
  • Exceedance factor: 15.5× WHO

HEALTH IMPACT (IER)
  • Total excess deaths: 1,907 yr$^{{-1}}$
  • Rate: 22.6 per 100,000
  • % attributable: 7.4%
  • Leading cause: IHD (826 deaths)

DATA SOURCES
  • PM$_{2.5}$: CAMS (daily), MERRA-2 (daily),
    Washington U. (monthly, 1 km)
  • Temperature: ERA5 reanalysis
  • Population: Census 2011 projection
  • Health model: Burnett et al. (2014) IER

METHODS
  • HPE detection: 0 days (inverse seasonal)
  • GWR: Spatially varying PM$_{2.5}$-mortality
  • Comparison: r = 0.33 (MERRA-2 vs CAMS)
"""
    ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3, edgecolor='black'))
    
    fig.suptitle('Air Quality and Health Impact Assessment — Ahmedabad, India (2019)', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    fig.savefig(FIG_DIR / 'fig5_combined_publication.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'fig5_combined_publication.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("CREATING PUBLICATION FIGURES")
    print("=" * 60)
    
    figure1_pm25_comparison()
    figure2_ier_map()
    figure3_ier_bars()
    figure4_gwr_spatial()
    figure5_combined()
    
    print("\n" + "=" * 60)
    print("ALL FIGURES COMPLETE")
    print("=" * 60)
    print(f"Figures saved to: {FIG_DIR}")
    print("\nFiles:")
    for f in sorted(FIG_DIR.glob("*.png")):
        size = f.stat().st_size / 1024
        print(f"  {f.name} ({size:.0f} KB)")


if __name__ == "__main__":
    main() 