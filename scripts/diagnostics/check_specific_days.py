import xarray as xr

tmax = xr.open_dataset('outputs/era5_daily_2019.nc')['tmax']
pm25 = xr.open_dataset('outputs/pm25_3day_cams_ahmedabad_2019.nc')['pm25']

# Select same point
tmax_point = tmax.sel(longitude=72.75, method='nearest').sel(latitude=23.0, method='nearest')
pm25_point = pm25.sel(longitude=72.75, method='nearest').sel(latitude=23.25, method='nearest')

tmax_90th = float(tmax_point.quantile(0.9).values)
pm25_75th = float(pm25_point.quantile(0.75).values)

print(f"Tmax 90th: {tmax_90th:.1f}C")
print(f"PM2.5 75th: {pm25_75th:.1f} ug/m3")
print(f"\nDays that are HOT (Tmax > {tmax_90th:.1f}) AND POLLUTED (PM2.5 > {pm25_75th:.1f}):")

found = 0
for i in range(len(tmax_point.time)):
    t = float(tmax_point.isel(time=i).values)
    p = float(pm25_point.isel(valid_time=i).values)
    date = str(tmax_point.time.values[i])[:10]
    
    if t > tmax_90th and p > pm25_75th:
        print(f"  {date}: Tmax={t:.1f}C, PM2.5={p:.1f} *** HPE ***")
        found += 1
    elif t > tmax_90th:
        print(f"  {date}: Tmax={t:.1f}C, PM2.5={p:.1f} (hot only)")
    elif p > pm25_75th:
        print(f"  {date}: Tmax={t:.1f}C, PM2.5={p:.1f} (polluted only)")

print(f"\nTotal HPE days found: {found}")

tmax.close()
pm25.close()