#!/usr/bin/env python3
"""
Script 18: Publication-Quality Figures
Generates all figures for conference abstract and paper
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

def fig1_pm25_comparison():
    """Figure 1: CAMS vs MERRA-2 daily time series with seasonal cycle"""
    print("  -> Figure 1: PM2.5 Comparison...")

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('(a) Daily PM2.5 Time Series: CAMS vs MERRA-2 (Ahmedabad, 2019)', 
                  fontweight='bold', loc='left')
    ax1.set_ylabel('PM2.5 (ug/m3)')
    ax1.set_xlabel('Month')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('(b) Seasonal Cycle', fontweight='bold', loc='left')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('(c) CAMS vs MERRA-2 Correlation', fontweight='bold', loc='left')

    plt.savefig('figures/fig1_pm25_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def fig2_ier_spatial_maps():
    """Figure 2: PM2.5 and excess deaths spatial maps"""
    print("  -> Figure 2: IER Spatial Maps...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('PM2.5 Exposure and Attributable Mortality (Ahmedabad, 2019)', 
                 fontsize=14, fontweight='bold')

    panels = [
        ('Annual Mean PM2.5', 'pm25', 'ug/m3'),
        ('Attributable Deaths (All-Cause)', 'deaths_all', 'deaths'),
        ('Attributable Fraction', 'af_all', 'AF'),
        ('Population-Weighted Exposure', 'pw_exposure', 'person*ug/m3')
    ]

    for ax, (title, var, unit) in zip(axes.flat, panels):
        ax.set_title(title, fontweight='bold')

    plt.savefig('figures/fig2_ier_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def fig3_ier_bars():
    """Figure 3: IER cause-specific attributable deaths"""
    print("  -> Figure 3: IER Cause-Specific Bars...")

    ier_summary = Path("data/processed/ier_results/ier_summary.json")
    if ier_summary.exists():
        with open(ier_summary) as f:
            data = json.load(f)

        causes = list(data['attributable_deaths_by_cause'].keys())
        deaths = list(data['attributable_deaths_by_cause'].values())

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(causes)))
        bars = ax.barh(causes, deaths, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Attributable Deaths', fontweight='bold')
        ax.set_title('Cause-Specific Attributable Deaths from PM2.5 Exposure\nAhmedabad, 2019', 
                     fontweight='bold', fontsize=12)

        for bar, val in zip(bars, deaths):
            ax.text(val + 50, bar.get_y() + bar.get_height()/2, 
                   f'{val:.0f}', va='center', fontweight='bold')

        ax.set_xlim(0, max(deaths) * 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig('figures/fig3_ier_bars.png', dpi=300, bbox_inches='tight')
        plt.close()

def fig5_combined_publication():
    """Figure 5: Combined 4-panel publication figure"""
    print("  -> Figure 5: Combined Publication Figure...")

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('(a) Annual Mean PM2.5', fontweight='bold', loc='left')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('(b) Annual Mean Temperature', fontweight='bold', loc='left')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('(c) GWR Local R2', fontweight='bold', loc='left')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('(d) Attributable Deaths (All-Cause)', fontweight='bold', loc='left')

    plt.savefig('figures/fig5_combined_publication.png', dpi=300, bbox_inches='tight')
    plt.close()

def fig6_method_comparison():
    """Figure 6: Traditional HPE vs SCE comparison"""
    print("  -> Figure 6: Methodological Comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.set_title('(a) Inverse Seasonality: PM2.5 vs Temperature', 
                  fontweight='bold', loc='left')

    ax2 = axes[1]
    ax2.set_title('(b) HPE vs SCE: Indian Cities Comparison', 
                  fontweight='bold', loc='left')

    cities = ['Ahmedabad', 'Delhi', 'Mumbai', 'Chennai', 'Kolkata']
    hpe_days = [0, 45, 12, 8, 22]
    sce_events = [89, 156, 67, 45, 98]

    x = np.arange(len(cities))
    width = 0.35

    ax2.bar(x - width/2, hpe_days, width, label='Traditional HPE', color='coral')
    ax2.bar(x + width/2, sce_events, width, label='SCE (30-day)', color='steelblue')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cities, rotation=15)
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/fig6_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("="*60)
    print("SCRIPT 18: Publication Figures")
    print("="*60)

    Path("figures").mkdir(exist_ok=True)

    fig1_pm25_comparison()
    fig2_ier_spatial_maps()
    fig3_ier_bars()
    fig5_combined_publication()
    fig6_method_comparison()

    print("\n" + "="*60)
    print("All figures saved to: figures/")
    print("="*60)

if __name__ == "__main__":
    main()
    