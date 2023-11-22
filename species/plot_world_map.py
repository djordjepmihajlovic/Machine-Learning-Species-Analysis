# Displays where in the world a random species has been observed, overlaid on world map

import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

data = np.load("species_train.npz")

data_test = np.load("species_test.npz", allow_pickle=True)
test_locs = data_test['test_locs']  
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))   

train_locs = data['train_locs']
train_ids = data['train_ids']
species = data['taxon_ids']
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))

# Choosing random species
sp = np.random.choice(species)
sp = 12716
train_inds_pos = np.where(train_ids == sp)[0]
test_inds_pos = test_pos_inds[sp] 

geometry = [Point(xy) for xy in zip(train_locs[train_inds_pos, 1], train_locs[train_inds_pos, 0])] # gets list of (lat,lon) pairs
geometry = [Point(xy) for xy in zip(test_locs[test_inds_pos, 1], test_locs[test_inds_pos, 0])]
gdf = GeoDataFrame(geometry=geometry) # creates geopandas dataframe of these pairs

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) # world map included with geopandas, could download other maps
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='k', markersize=5)
# plt.title(str(sp) + ' - ' + species_names[sp])

norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(norm=norm, cmap='plasma')
sm.set_array([])  # You need to set an array, even if it's empty

# Create a colorbar with a label
# cbar = plt.colorbar(sm)
# cbar.set_ticks([0.2])
# cbar.set_label('Vulnerability')
# cbar.ax.set_yticklabels([0.2], rotation=0, size=14)
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()

plt.show()