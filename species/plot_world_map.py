# Displays where in the world a random species has been observed, overlaid on world map

import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import matplotlib.pyplot as plt

data = np.load("species_train.npz")

train_locs = data['train_locs']
train_ids = data['train_ids']
species = data['taxon_ids']
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))

# Choosing random species
sp = np.random.choice(species)
train_inds_pos = np.where(train_ids == sp)[0]

geometry = [Point(xy) for xy in zip(train_locs[train_inds_pos, 1], train_locs[train_inds_pos, 0])] # gets list of (lat,lon) pairs
gdf = GeoDataFrame(geometry=geometry) # creates geopandas dataframe of these pairs

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) # world map included with geopandas, could download other maps
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='k', markersize=5)
plt.title(str(sp) + ' - ' + species_names[sp])
plt.show()