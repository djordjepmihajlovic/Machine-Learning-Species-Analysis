import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.linear_model import LogisticRegression
import pickle

#Load data
data = np.load('../../data/species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

range_list = range(len(species)) #Range from 0-499
spec_dict = dict(zip(species, range_list)) #Dictionary matches species id with index in species
train_ids_v2 = [] #List of train ids, now they go from 0 to 499
for indx in train_ids:
    x = spec_dict.get(indx)
    train_ids_v2.append(x)
train_ids_v3 = np.array(train_ids_v2)

#Balance data

mean_train = 544
species_count = np.bincount(train_ids) 
sp_list_a = [] 
sp_list_b = [] 

i = 0
for n in species_count:
    if n > mean_train: 
        sp_list_a.append(i) 
    elif n != 0:
        sp_list_b.append(i)
    i = i + 1

train_inds_pos_a = [] 
train_inds_pos_b= [] 
wanted_indices = [] 

for species_id in sp_list_a:
    train_inds_pos_a.append(np.where(train_ids == species_id)[0])

for species_id in sp_list_b:
    train_inds_pos_b.append(np.where(train_ids == species_id)[0])

for sp_indices in train_inds_pos_a:
    sp_choice = np.random.choice(sp_indices, mean_train, replace = False) #
    wanted_indices.append(sp_choice)

for sp_indices in train_inds_pos_b:
    wanted_indices.append(sp_indices)

flat_wanted_indices = [item for sublist in wanted_indices for item in sublist]
new_train_locs = train_locs[flat_wanted_indices]
new_train_ids = train_ids_v3[flat_wanted_indices]

# test data
data_test = np.load('../../data/species_test.npz', allow_pickle=True) 
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
with open('lr_model.pkl', 'rb') as f:
    lr = pickle.load(f)


id = 12716 # turdus merula
id_index = spec_dict[id]

n_gridpoints = 1000
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