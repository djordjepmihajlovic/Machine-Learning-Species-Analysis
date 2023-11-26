import matplotlib.pyplot as plt
import numpy as np 
import random
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point

# load train data
data = np.load('../../data/species_train.npz')
ids = data['train_ids']
classes = np.unique(ids)
coords = np.array(list(zip(data['train_locs'][:,0], data['train_locs'][:,1]))) 
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))
#id = 46217 # chipmunk
#id = 29351 # natrix
id = 12716

ind = np.where(ids == id)
x = coords[ind]
N = len(x)

# mle for mean vector
mu = np.array([np.sum(x[:,0]), np.sum(x[:,1])])/N

# mle for cov matrix
sig = np.array([[0.0,0.0],[0.0,0.0]])
for i in range(N):
    sig += np.outer((x[i]-mu), (x[i]-mu))
sig = sig/N

n_gridpoints = 100
lats = np.linspace(-90, 90, n_gridpoints)
longs = np.linspace(-180, 180, n_gridpoints)
pvals = np.zeros((n_gridpoints, n_gridpoints))

for i in range(n_gridpoints):
    for j in range(n_gridpoints):
        pvals[i,j] = (1/(2* np.pi * np.linalg.det(sig))) * np.exp(-0.5 * np.dot(([lats[i],longs[j]]-mu), np.matmul(np.linalg.inv(sig), ([lats[i],longs[j]]-mu))))

X, Y = np.meshgrid(longs, lats)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
ax = world.plot(figsize=(10, 6))
ax.set_title(str(12716) + ' - ' + str(species_names[id]))
cs = ax.contourf(X, Y, pvals, levels = np.linspace(10**(-10), np.max(pvals), 10), alpha = 0.5, cmap = 'plasma')
ax.clabel(cs, inline = True)
plt.show() 