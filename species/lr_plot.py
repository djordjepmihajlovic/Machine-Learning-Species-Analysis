import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.linear_model import LogisticRegression
import pickle

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

# logistic regression classifier
"""
lr = LogisticRegression()
lr.fit(train_locs, train_ids)
with open('lr_model.pkl','wb') as f:
    pickle.dump(lr,f)
"""
with open('species/lr_model.pkl', 'rb') as f:
    lr = pickle.load(f)


id = 12716 # turdus merula
id_index = np.where(lr.classes_ == id)[0][0]

n_gridpoints = 500
lats = np.linspace(-90, 90, n_gridpoints)
longs = np.linspace(-180, 180, n_gridpoints)
pvals = np.zeros((n_gridpoints, n_gridpoints))

for i in range(n_gridpoints):
    for j in range(n_gridpoints):
        pvals[i,j] = lr.predict_proba(np.array([lats[i], longs[j]]).reshape(1,-1))[0, id_index]


X, Y = np.meshgrid(longs, lats)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
ax = world.plot(figsize=(10, 6))
ax.set_xticks([])
ax.set_yticks([])
cs = ax.contourf(X, Y, pvals, vmin=0.006, vmax = 0.05, levels = [0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],alpha = 0.5, cmap = 'plasma')
ax.clabel(cs, inline = True)
plt.show() 
#plt.savefig('lr_plot.png')