import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean 
import cartopy.crs as ccrs
#import seaborn as sns

# load datasets
hist = xr.load_dataset('species/climate_data/ts_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_20140616-20140616_v20190624.nc')
proj = xr.load_dataset('species/climate_data/ts_Amon_HadGEM3-GC31-LL_ssp245_r1i1p1f3_gn_20500616-20500616_v20190908.nc')
lat = hist.coords['lat']
lon = hist.coords['lon']
print(hist.lat)
print(proj.lat)
temp = proj['ts'].mean('time') - hist['ts'].mean('time')

subplot_kws=dict(projection=ccrs.PlateCarree(), facecolor='white')
plt.figure(dpi=(150))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.title('')
temp.plot(cmap='bwr', subplot_kws=subplot_kws, transform=ccrs.PlateCarree())
plt.show()