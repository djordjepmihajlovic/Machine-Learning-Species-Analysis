import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.neighbors import KNeighborsClassifier

# scan over all coordinates
# compute 5 nearest neighbors
# return ratio of nn which are 12716 over total neighbours = probability

# load training data    
data = np.load('species/species_train.npz')
#data = np.load('species_train.npz')
train_locs = data['train_locs']
train_ids = data['train_ids']
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

# test data
data_test = np.load('species/species_test.npz', allow_pickle=True) 
test_locs = data_test['test_locs']
num_locs = len(test_locs)
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))

# k nearest neighbours classifier, optimal k found by examining F1 scores
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_locs, train_ids)

id = 12716 # turdus merula
id_index = np.where(knn.classes_ == id)[0][0]

n_gridpoints = 500
lats = np.linspace(-90, 90, n_gridpoints)
longs = np.linspace(-180, 180, n_gridpoints)
pvals = np.zeros((n_gridpoints, n_gridpoints))

for i in range(n_gridpoints):
    for j in range(n_gridpoints):
        pvals[i,j] = knn.predict_proba(np.array([lats[i], longs[j]]).reshape(1,-1))[0, id_index]

X, Y = np.meshgrid(longs, lats)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
ax = world.plot(figsize=(10, 6))
ax.set_xticks([])
ax.set_yticks([])
cs = ax.contourf(X, Y, pvals, levels = [0.33, 0.66, 1], alpha = 0.5, cmap = 'plasma')
#ax.clabel(cs, inline = True)
plt.show() 
#plt.savefig('knn_plot.png')