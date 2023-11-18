import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean 
import cartopy.crs as ccrs
#import seaborn as sns

# load datasets
hist_30_year = xr.load_dataset('species/climate_data/ts_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_19840116-20141216_v20190624.nc')
proj = xr.load_dataset('species/climate_data/ts_Amon_HadGEM3-GC31-LL_ssp585_r1i1p1f3_gn_20500116-20501216_v20200114.nc')

baseline = hist_30_year['ts'].mean(dim='time') - 273.15
projected = proj['ts'].mean(dim='time') - 273.15
difference = projected - baseline
# remove any differences less than zero
difference = difference.where(difference > 0, 0)
difference.attrs["long_name"] = "Temperature Anomaly"
difference.attrs["units"] = r"$^\circ$C"

subplot_kws=dict(projection=ccrs.PlateCarree(), facecolor='white')
plt.figure(dpi=(150))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.title('')
p = difference.plot(cmap='OrRd', subplot_kws=subplot_kws, transform=ccrs.PlateCarree(), add_labels=False,add_colorbar=False)
plt.colorbar(p, ticks=[0,4,8,12,16,20], shrink=0.5, label=r'Temperature Anomaly ($^\circ$C)')
#plt.show()

# get a "score" by normalising by the largest increase in temp
score = difference / difference.max()
# 
lats = score.lat.to_numpy()
lons = score.lon.to_numpy()
num_coords = len(lats) * len(lons)
coords = np.zeros((num_coords,2))
scores = np.zeros(num_coords)
i = 0
for lat in  lats:
    for lon in lons:
        coords[i,0] = lat
        coords[i,1] = lon
        scores[i] = score.sel(lat=lat,lon=lon)
        i += 1
np.save('scores', scores)
np.save('scores_coords', coords)