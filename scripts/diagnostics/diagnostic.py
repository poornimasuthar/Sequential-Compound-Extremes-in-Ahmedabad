import xarray as xr

pm25 = xr.open_dataset('outputs/pm25_daily_cams_ahmedabad_2019.nc')['pm25']
tmax = xr.open_dataset('outputs/era5_daily_2019.nc')['tmax']

print('PM2.5 daily stats:')
print(f'  Mean: {float(pm25.mean().values):.1f}')
print(f'  75th percentile: {float(pm25.quantile(0.75).values):.1f}')
print(f'  Max: {float(pm25.max().values):.1f}')

print('Tmax stats:')
print(f'  Mean: {float(tmax.mean().values):.1f}')
print(f'  90th percentile: {float(tmax.quantile(0.9).values):.1f}')
print(f'  Max: {float(tmax.max().values):.1f}')

# Check a few specific days
print('\nSample days (Jan 2019):')
for i in range(5):
    p = float(pm25.isel(valid_time=i).mean().values)
    t = float(tmax.isel(time=i).mean().values)
    print(f'  Day {i+1}: PM2.5={p:.1f}, Tmax={t:.1f}')
    