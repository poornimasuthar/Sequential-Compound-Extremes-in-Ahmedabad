import xarray as xr

era5 = xr.open_dataset('outputs/era5_daily_2019.nc')
cams = xr.open_dataset('outputs/pm25_daily_cams_ahmedabad_2019.nc')

print('ERA5 lat:', era5.latitude.values)
print('ERA5 lon:', era5.longitude.values)
print('CAMS lat:', cams.latitude.values)
print('CAMS lon:', cams.longitude.values)

era5.close()
cams.close()
