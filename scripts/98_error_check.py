#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

print('=' * 70)
print('COMPREHENSIVE ERROR CHECK - AHMEDABAD HPE PROJECT')
print('=' * 70)

base = Path('.')

# CHECK 1: FILES
print('\n' + '=' * 70)
print('CHECK 1: DATA FILES')
print('=' * 70)
required = [
    'outputs/pm25_daily_merra2_ahmedabad_2019.nc',
    'outputs/era5_daily_2019.nc',
    'outputs/ier_summary.csv',
]
for f in required:
    print(f'  {f}: {"EXISTS" if (base/f).exists() else "MISSING"}')

# CHECK 2: PM2.5
print('\n' + '=' * 70)
print('CHECK 2: PM2.5 DATA')
print('=' * 70)
ds_pm = xr.open_dataset(base / 'outputs/pm25_daily_merra2_ahmedabad_2019.nc')
pm25 = ds_pm['pm25'].to_pandas()
pm25.index = pd.to_datetime(pm25.index)
print(f'  Days: {len(pm25)}')
print(f'  Min: {pm25.min():.1f}, Max: {pm25.max():.1f}, Mean: {pm25.mean():.1f}')

print('\n  Monthly PM2.5 means:')
monthly = pm25.resample('ME').mean()
for date, val in monthly.items():
    print(f'    {date.strftime("%B")}: {val:.1f}')

winter = pm25[pm25.index.month.isin([12,1,2])].mean()
summer = pm25[pm25.index.month.isin([3,4,5])].mean()
monsoon = pm25[pm25.index.month.isin([6,7,8,9])].mean()
post_monsoon = pm25[pm25.index.month.isin([10,11])].mean()

print(f'\n  Winter (Dec-Feb): {winter:.1f}')
print(f'  Summer (Mar-May): {summer:.1f}')
print(f'  Monsoon (Jun-Sep): {monsoon:.1f}')
print(f'  Post-monsoon (Oct-Nov): {post_monsoon:.1f}')

if winter > post_monsoon > summer > monsoon:
    print('  ✅ Seasonal pattern CORRECT')
else:
    print('  ❌ Seasonal pattern WRONG - expected Winter > Post-monsoon > Summer > Monsoon')

# CHECK 3: TEMPERATURE
print('\n' + '=' * 70)
print('CHECK 3: TEMPERATURE DATA')
print('=' * 70)
ds_t = xr.open_dataset(base / 'outputs/era5_daily_2019.nc')
tmax = ds_t['tmax']
if 'latitude' in tmax.dims:
    tmax = tmax.mean(dim=['latitude', 'longitude'])
tmax = tmax.to_pandas()
tmax.index = pd.to_datetime(tmax.index)
print(f'  Days: {len(tmax)}')
print(f'  Min: {tmax.min():.1f}°C, Max: {tmax.max():.1f}°C')

print('\n  Monthly Tmax means:')
monthly_t = tmax.resample('ME').mean()
for date, val in monthly_t.items():
    print(f'    {date.strftime("%B")}: {val:.1f}°C')

peak_month = monthly_t.idxmax().month
print(f'\n  Peak month: {peak_month} (expected 4 or 5)')

# CHECK 4: HPE/SCE
print('\n' + '=' * 70)
print('CHECK 4: HPE/SCE CALCULATIONS')
print('=' * 70)
pm75 = pm25.quantile(0.75)
t95 = tmax.quantile(0.95)
print(f'  PM75: {pm75:.1f}, T95: {t95:.1f}')

hpe = ((pm25 > pm75) & (tmax > t95)).sum()
print(f'  HPE days: {int(hpe)}')

high_pm = pm25[pm25 > pm75].index
extreme_t = tmax[tmax > t95].index
sce_30 = sum(1 for d in high_pm if any((extreme_t > d) & (extreme_t <= d + pd.Timedelta(days=30))))
print(f'  SCE-30: {sce_30}')

hpe_dates = pm25.index[(pm25 > pm75) & (tmax > t95)]
print(f'  HPE dates: {[str(d.date()) for d in hpe_dates]}')

# CHECK 5: IER
print('\n' + '=' * 70)
print('CHECK 5: IER HEALTH IMPACT')
print('=' * 70)
ier = pd.read_csv(base / 'outputs/ier_summary.csv')
print(ier.to_string())
total = ier['Excess'].sum()
print(f'\n  Total excess deaths: {total:.0f}')

print('\n' + '=' * 70)
print('CHECK COMPLETE')
print('=' * 70)
