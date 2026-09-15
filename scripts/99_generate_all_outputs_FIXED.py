import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("=" * 70)
print("AHMEDABAD HPE PROJECT - COMPLETE OUTPUT GENERATION")
print("=" * 70)

base = Path(".")
out_dir = base / "outputs"
fig_dir = base / "figures"
fig_dir.mkdir(exist_ok=True)

# Load Data
print("\n" + "=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

# PM2.5 - MERRA-2 daily (already spatially averaged, only 'time' dimension)
print("\n[1.1] PM2.5 (MERRA-2)...")
ds_pm = xr.open_dataset(out_dir / "pm25_daily_merra2_ahmedabad_2019.nc")
print(f"    Variables: {list(ds_pm.data_vars)}")
print(f"    Dimensions: {dict(ds_pm.dims)}")

pm_var = 'pm25' if 'pm25' in ds_pm.data_vars else list(ds_pm.data_vars)[0]
pm25 = ds_pm[pm_var]

#pandas conversion
pm25_daily = pm25.to_pandas()
pm25_daily.index = pd.to_datetime(pm25_daily.index)
print(f"    Loaded: {len(pm25_daily)} days, mean={pm25_daily.mean():.1f} ug/m3")

# Temperature - ERA5 daily (tmax)
print("[1.2] Temperature (ERA5 tmax)...")
ds_t = xr.open_dataset(out_dir / "era5_daily_2019.nc")
print(f"    Variables: {list(ds_t.data_vars)}")
print(f"    Dimensions: {dict(ds_t.dims)}")

t_var = 'tmax' if 'tmax' in ds_t.data_vars else list(ds_t.data_vars)[0]
tmax = ds_t[t_var]

# dimensions check and handling
if 'latitude' in tmax.dims and 'longitude' in tmax.dims:
    tmax_daily = tmax.mean(dim=['latitude', 'longitude']).to_pandas()
elif 'lat' in tmax.dims and 'lon' in tmax.dims:
    tmax_daily = tmax.mean(dim=['lat', 'lon']).to_pandas()
else:
    tmax_daily = tmax.to_pandas()

tmax_daily.index = pd.to_datetime(tmax_daily.index)
print(f"    Loaded: {len(tmax_daily)} days, mean={tmax_daily.mean():.1f} C")

# Align
common = pm25_daily.index.intersection(tmax_daily.index)
pm25_daily = pm25_daily.loc[common]
tmax_daily = tmax_daily.loc[common]
print(f"    Aligned: {len(common)} common days")

#THRESHOLDS & HPE/SCE DETECTION
print("\n" + "=" * 70)
print("STEP 2: COMPOUND EXTREME DETECTION")
print("=" * 70)

pm75 = pm25_daily.quantile(0.75)
t95 = tmax_daily.quantile(0.95)

print(f"\n    PM2.5 75th percentile: {pm75:.1f} ug/m3")
print(f"    Tmax 95th percentile: {t95:.1f} C")

# Traditional HPE
hpe_mask = (pm25_daily > pm75) & (tmax_daily > t95)
hpe_days = hpe_mask.sum()
hpe_dates = pm25_daily.index[hpe_mask].tolist()

print(f"\n[2.1] Traditional HPE: {int(hpe_days)} days")
if hpe_dates:
    print(f"    Dates: {[str(d.date()) for d in hpe_dates]}")

# SCE detection
high_pm = pm25_daily[pm25_daily > pm75].index
extreme_t = tmax_daily[tmax_daily > t95].index

print(f"    High PM2.5 days: {len(high_pm)}")
print(f"    Extreme temp days: {len(extreme_t)}")

sce_results = {}
for window in [15, 30, 60]:
    sce = 0
    sce_pairs = []
    for pm_date in high_pm:
        window_end = pm_date + pd.Timedelta(days=window)
        heat_in_window = extreme_t[(extreme_t > pm_date) & (extreme_t <= window_end)]
        if len(heat_in_window) > 0:
            sce += 1
            sce_pairs.append((str(pm_date.date()), str(heat_in_window[0].date())))
    sce_results[f'sce_{window}'] = {'count': sce, 'pairs': sce_pairs[:5]}
    print(f"[2.2] SCE-{window}: {sce} events")

# SOI
soi_values = []
for month in [3, 10]:
    pm_m = pm25_daily[pm25_daily.index.month == month]
    t_m = tmax_daily[tmax_daily.index.month == month]
    common_m = pm_m.index.intersection(t_m.index)
    if len(common_m) > 5:
        pm_n = (pm_m.loc[common_m] - pm_m.min()) / (pm_m.max() - pm_m.min() + 1e-10)
        t_n = (t_m.loc[common_m] - t_m.min()) / (t_m.max() - t_m.min() + 1e-10)
        soi = (pm_n * t_n).mean()
        soi_values.append(soi)
        print(f"[2.3] SOI Month {month}: {soi:.3f}")

soi_mean = float(np.mean(soi_values)) if soi_values else 0

# CEB
pm_z = (pm25_daily - pm25_daily.mean()) / pm25_daily.std()
t_z = (tmax_daily - tmax_daily.mean()) / tmax_daily.std()
ceb = float(np.sqrt(pm_z**2 + t_z**2).mean())

# STEP 3: SAVE ALL SUMMARIES
print("\n" + "=" * 70)
print("STEP 3: SAVING SUMMARIES")
print("=" * 70)

master_summary = {
    "project": "Ahmedabad HPE 2019",
    "date_generated": str(pd.Timestamp.now()),
    "data": {
        "pm25_annual_mean": float(pm25_daily.mean()),
        "pm25_max": float(pm25_daily.max()),
        "pm25_min": float(pm25_daily.min()),
        "tmax_annual_mean": float(tmax_daily.mean()),
        "tmax_max": float(tmax_daily.max()),
        "tmax_min": float(tmax_daily.min()),
        "total_days": len(pm25_daily)
    },
    "thresholds": {
        "pm75": float(pm75),
        "t95": float(t95)
    },
    "compound_extremes": {
        "traditional_hpe_days": int(hpe_days),
        "hpe_dates": [str(d.date()) for d in hpe_dates],
        "sce_15day": sce_results['sce_15']['count'],
        "sce_30day": sce_results['sce_30']['count'],
        "sce_60day": sce_results['sce_60']['count'],
        "soi_mean": soi_mean,
        "ceb": ceb
    },
    "key_finding": f"Traditional HPE={int(hpe_days)} days vs SCE-30={sce_results['sce_30']['count']} events"
}

with open(out_dir / "MASTER_SUMMARY.json", "w") as f:
    json.dump(master_summary, f, indent=2)
print(f"    Saved: {out_dir / 'MASTER_SUMMARY.json'}")

# Save timeline
timeline = pd.DataFrame({
    'date': pm25_daily.index,
    'pm25': pm25_daily.values,
    'tmax': tmax_daily.values,
    'high_pm': pm25_daily > pm75,
    'extreme_temp': tmax_daily > t95,
    'hpe_day': hpe_mask
})
timeline.to_csv(out_dir / "complete_timeline_2019.csv", index=False)
print(f"    Saved: {out_dir / 'complete_timeline_2019.csv'}")


# STEP 4: GENERATE ALL FIGURES
print("\n" + "=" * 70)
print("STEP 4: GENERATING FIGURES")
print("=" * 70)

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

# FIGURE 1: Complete time series
print("\n[4.1] Figure 1: Complete time series...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(timeline['date'], timeline['pm25'], 'b-', alpha=0.7, lw=0.8)
axes[0].axhline(y=pm75, color='r', ls='--', alpha=0.7, label=f'75th percentile ({pm75:.1f})')
axes[0].fill_between(timeline['date'], pm75, timeline['pm25'].max(), 
                     where=timeline['high_pm'], alpha=0.2, color='blue', label='High PM2.5')
axes[0].set_ylabel('PM2.5 (µg/m³)')
axes[0].set_title('(a) Daily PM2.5 (MERRA-2)', fontweight='bold', loc='left')
axes[0].legend(loc='upper right', fontsize=8)

axes[1].plot(timeline['date'], timeline['tmax'], 'orange', alpha=0.7, lw=0.8)
axes[1].axhline(y=t95, color='r', ls='--', alpha=0.7, label=f'95th percentile ({t95:.1f}°C)')
axes[1].fill_between(timeline['date'], t95, timeline['tmax'].max(),
                     where=timeline['extreme_temp'], alpha=0.2, color='orange', label='Extreme heat')
axes[1].set_ylabel('Tmax (°C)')
axes[1].set_title('(b) Daily Maximum Temperature (ERA5)', fontweight='bold', loc='left')
axes[1].legend(loc='upper right', fontsize=8)

axes[2].fill_between(timeline['date'], 0, 0.5, where=timeline['high_pm'], 
                     alpha=0.3, color='blue', label=f'High PM2.5 (n={timeline["high_pm"].sum()})')
axes[2].fill_between(timeline['date'], 0.5, 1, where=timeline['extreme_temp'],
                     alpha=0.3, color='orange', label=f'Extreme heat (n={timeline["extreme_temp"].sum()})')
hpe_y = np.where(timeline['hpe_day'], 0.75, np.nan)
axes[2].scatter(timeline['date'], hpe_y, color='red', s=30, zorder=5, label=f'HPE days (n={int(hpe_days)})')
axes[2].set_ylabel('Event')
axes[2].set_xlabel('Date')
axes[2].set_title(f'(c) Inverse Seasonality: Only {int(hpe_days)} HPE Days, But {sce_results["sce_30"]["count"]} SCE-30 Events', 
                  fontweight='bold', loc='left')
axes[2].legend(loc='upper right', fontsize=8)
axes[2].set_ylim(0, 1)

plt.suptitle('Ahmedabad 2019: Compound Heat-Pollution Extremes', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(fig_dir / 'fig1_complete_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fig1_complete_timeseries.png")

# FIGURE 2: Seasonal cycle
print("[4.2] Figure 2: Seasonal cycle...")
monthly_pm = pm25_daily.resample('ME').mean()
monthly_t = tmax_daily.resample('ME').mean()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

bars = ax1.bar(range(1, 13), monthly_pm.values, alpha=0.6, color='steelblue', label='PM2.5')
line = ax2.plot(range(1, 13), monthly_t.values, 'ro-', markersize=6, label='Tmax')

ax1.set_xlabel('Month')
ax1.set_ylabel('PM2.5 (µg/m³)', color='steelblue')
ax2.set_ylabel('Tmax (°C)', color='red')
ax1.set_title('Seasonal Cycle: Inverse Seasonality in Ahmedabad (2019)', fontweight='bold')
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

ax1.annotate('PM2.5 PEAK', xy=(1, monthly_pm.iloc[0]), xytext=(1, monthly_pm.max()*1.1),
            arrowprops=dict(arrowstyle='->', color='blue'), fontsize=9, color='blue')
ax2.annotate('HEAT PEAK', xy=(5, monthly_t.iloc[4]), xytext=(5, monthly_t.max()*1.05),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

plt.tight_layout()
plt.savefig(fig_dir / 'fig2_seasonal_cycle.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fig2_seasonal_cycle.png")

# FIGURE 3: HPE vs SCE comparison
print("[4.3] Figure 3: HPE vs SCE comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Traditional HPE\n(Simultaneous)', 'SCE-15\n(2 weeks)', 'SCE-30\n(1 month)', 'SCE-60\n(2 months)']
counts = [int(hpe_days), sce_results['sce_15']['count'], sce_results['sce_30']['count'], sce_results['sce_60']['count']]
colors = ['coral', 'gold', 'steelblue', 'darkgreen']

bars = ax.bar(methods, counts, color=colors, edgecolor='black', width=0.6)
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val}', 
            ha='center', fontweight='bold', fontsize=14)

ax.set_ylabel('Number of Events', fontweight='bold')
ax.set_title('Compound Extreme Detection: Traditional vs Sequential Framework\nAhmedabad, 2019', 
             fontweight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

if counts[0] > 0:
    ratio = counts[2] / counts[0]
    ax.text(0.5, max(counts)*0.9, f'SCE-30 detects {ratio:.1f}× more events\nthan traditional HPE', 
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(fig_dir / 'fig3_hpe_vs_sce_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fig3_hpe_vs_sce_comparison.png")

# FIGURE 4: SCE timeline with connections
print("[4.4] Figure 4: SCE timeline visualization...")
fig, ax = plt.subplots(figsize=(16, 6))

ax2 = ax.twinx()
ax.plot(timeline['date'], timeline['pm25'], 'b-', alpha=0.5, lw=0.8, label='PM2.5')
ax2.plot(timeline['date'], timeline['tmax'], 'orange', alpha=0.5, lw=0.8, label='Tmax')

high_pm_dates = timeline[timeline['high_pm']]['date']
ax.scatter(high_pm_dates, [pm75]*len(high_pm_dates), color='blue', s=20, alpha=0.5, zorder=5)

extreme_t_dates = timeline[timeline['extreme_temp']]['date']
ax2.scatter(extreme_t_dates, [t95]*len(extreme_t_dates), color='orange', s=20, alpha=0.5, zorder=5)

for pm_date in high_pm:
    window_end = pm_date + pd.Timedelta(days=30)
    heat_in_window = extreme_t[(extreme_t > pm_date) & (extreme_t <= window_end)]
    if len(heat_in_window) > 0:
        pm_y = pm25_daily.loc[pm_date]
        ax.annotate('', xy=(heat_in_window[0], pm_y), xytext=(pm_date, pm_y),
                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.3, lw=0.5))

ax.set_ylabel('PM2.5 (µg/m³)', color='blue')
ax2.set_ylabel('Tmax (°C)', color='orange')
ax.set_xlabel('Date')
ax.set_title('SCE-30 Visualization: Winter Pollution → Summer Heat Connections', fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(fig_dir / 'fig4_sce_connections.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fig4_sce_connections.png")

# FIGURE 5: Summary dashboard
print("[4.5] Figure 5: Summary dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

ax_text = fig.add_subplot(gs[0, :])
ax_text.axis('off')
metrics_text = f"""
AHMEDABAD 2019 - KEY FINDINGS

PM2.5:        Mean={pm25_daily.mean():.1f} µg/m³  |  Max={pm25_daily.max():.1f}  |  Days >75th: {(pm25_daily>pm75).sum()}
Temperature:  Mean={tmax_daily.mean():.1f}°C       |  Max={tmax_daily.max():.1f}°C  |  Days >95th: {(tmax_daily>t95).sum()}

TRADITIONAL HPE:  {int(hpe_days)} days  |  SCE-15: {sce_results['sce_15']['count']}  |  SCE-30: {sce_results['sce_30']['count']}  |  SCE-60: {sce_results['sce_60']['count']}

SCE-30 detects {sce_results['sce_30']['count']/max(int(hpe_days),1):.1f}× more events than traditional HPE framework
"""
ax_text.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace', va='center')

ax1 = fig.add_subplot(gs[1, 0])
ax1.bar(range(1, 13), monthly_pm.values, color='steelblue', alpha=0.7)
ax1.set_title('Monthly PM2.5')
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

ax2 = fig.add_subplot(gs[1, 1])
ax2.bar(range(1, 13), monthly_t.values, color='coral', alpha=0.7)
ax2.set_title('Monthly Tmax')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

ax3 = fig.add_subplot(gs[1, 2])
ax3.bar(['HPE', 'SCE-30'], [int(hpe_days), sce_results['sce_30']['count']], 
        color=['coral', 'steelblue'], edgecolor='black')
ax3.set_title('HPE vs SCE-30')
for i, v in enumerate([int(hpe_days), sce_results['sce_30']['count']]):
    ax3.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(pm25_daily, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax4.axvline(x=pm75, color='r', ls='--', label='75th %ile')
ax4.set_title('PM2.5 Distribution')
ax4.set_xlabel('PM2.5 (µg/m³)')

ax5 = fig.add_subplot(gs[2, 1])
ax5.hist(tmax_daily, bins=30, color='coral', alpha=0.7, edgecolor='black')
ax5.axvline(x=t95, color='r', ls='--', label='95th %ile')
ax5.set_title('Tmax Distribution')
ax5.set_xlabel('Tmax (°C)')

ax6 = fig.add_subplot(gs[2, 2])
colors_month = timeline['date'].dt.month
scatter = ax6.scatter(timeline['pm25'], timeline['tmax'], c=colors_month, cmap='jet', alpha=0.5, s=10)
ax6.axvline(x=pm75, color='r', ls='--', alpha=0.5)
ax6.axhline(y=t95, color='r', ls='--', alpha=0.5)
ax6.set_xlabel('PM2.5 (µg/m³)')
ax6.set_ylabel('Tmax (°C)')
ax6.set_title('PM2.5 vs Tmax (colored by month)')
plt.colorbar(scatter, ax=ax6, label='Month')

plt.suptitle('Ahmedabad HPE Project 2019 - Summary Dashboard', fontsize=14, fontweight='bold', y=0.98)
plt.savefig(fig_dir / 'fig5_summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: fig5_summary_dashboard.png")
# STEP 5: FINAL SUMMARY
print("\n" + "=" * 70)
print("COMPLETE - ALL OUTPUTS GENERATED")
print("=" * 70)
print(f"\nSummary saved: {out_dir / 'MASTER_SUMMARY.json'}")
print(f"Timeline saved: {out_dir / 'complete_timeline_2019.csv'}")
print(f"\nFigures generated in {fig_dir}:")
for f in sorted(fig_dir.glob('fig*.png')):
    print(f"  - {f.name}")

print("\n" + "=" * 70)
print("KEY FINDING")
print("=" * 70)
print(f"""
Traditional HPE framework:  {int(hpe_days)} days
SCE-30 framework:           {sce_results['sce_30']['count']} events
UNDERESTIMATION:            {sce_results['sce_30']['count']/max(int(hpe_days),1):.1f}x

The traditional framework misses {sce_results['sce_30']['count'] - int(hpe_days)} 
compound extreme events in Ahmedabad's inverse-seasonal climate.
""")
print("=" * 70)
