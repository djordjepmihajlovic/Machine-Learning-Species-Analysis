from sklearn import tree
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

data = np.load('species_train.npz')
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

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 
with open('reverse_dict.pkl', 'rb') as file:
    reverse_test_pos_inds = pickle.load(file)

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

tcf = tree.DecisionTreeClassifier()#min_samples_leaf= 20)

tcf.fit(new_train_locs, new_train_ids)

predictions = tcf.predict(test_locs)

predictions_p = tcf.predict_proba(test_locs)

test_ids = [] #Uses the new reverse dictionary to create set ids to each of the test locations
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    test_ids.append(test_id)

location_index = 15689

print('proabilities in random location =', predictions_p[location_index])
print('length of prediction array', len(predictions_p[location_index]))
prob_indeces = []
for index in range(len(predictions_p[location_index])):
    if predictions_p[location_index][index] > 0:
        prob_indeces.append(index)
    else:
        continue
print('indeces with probability above 0 in loc =', prob_indeces)
print('predicition in same random location =', predictions[location_index])
print('index of prediction', np.where(species == predictions[location_index])[0])
print('actual species in location', test_ids[location_index])
real_indeces = []
for id in test_ids[location_index]:
    real_indeces.append(np.where(species == id)[0])
print('indices of species in location:', real_indeces)

"""

# Create a mapping of id to index in the ids list
id_to_index = {id: i for i, id in enumerate(species)}

# Initialize a 2D array with zeros
final_array = np.zeros((len(new_train_locs), len(species)), dtype=int)

for i, species_id in enumerate(new_train_ids):
    index = id_to_index.get(species_id)
    if index is not None:
        final_array[i, index] = species_id
"""
"""
id_to_index = {id: i for i, id in enumerate(species)}

# Initialize a list of lists with zeros
final_list = [[0] * len(species) for _ in range(len(new_train_locs))]

for i, species_id in enumerate(new_train_ids):
    index = id_to_index.get(species_id)
    if index is not None:
        final_list[i][index] = species_id

final_list2 = mlb.fit_transform(final_list)
#print(result_array.shape)
"""
"""
#print(tree_classifier.classes_)

#predictions = tree_classifier.predict(test_locs)

location_index = 0
location_features = test_locs[location_index].reshape(1, -1)  # Reshape to match the expected input shape

# Get the probability estimates for each class
#probabilities = tree_classifier.predict_proba(location_features)
test_locs_final = []
test_ids = [] #Uses the new reverse dictionary to create set ids to each of the test locations
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    if test_id[0] == 0:
        continue
    test_ids.append(test_id)
    test_locs_final.append(test_locs[index])

print(len(test_locs_final))
print(len(test_ids))
"""
"""
# Initialize a list of lists with zeros
converted_list_of_lists = [[0] * len(species) for _ in range(len(test_ids))]

# Create a mapping of species to their indices
species_to_index = {id: i for i, id in enumerate(species)}

# Fill in the converted list of lists
for location_index, id_list in enumerate(test_ids):
    for species_id in id_list:
        index = species_to_index.get(species_id)
        if index is not None:
            converted_list_of_lists[location_index][index] = species_id
"""

#print(test_ids)
#print(final_list)
"""

non_zero_locations = []

# Initialize a list of lists with zeros
converted_list_of_lists = [[0] * len(species) for _ in range(len(test_ids))]

# Create a mapping of species to their indices
species_to_index = {id: i for i, id in enumerate(species)}

# Fill in the converted list of lists
for location_index, id_list in enumerate(test_ids):
    has_non_zero = False
    for species_id in id_list:
        index = species_to_index.get(species_id)
        if index is not None:
            converted_list_of_lists[location_index][index] = species_id
            has_non_zero = True

    # If at least one non-zero ID, include the location
    if has_non_zero:
        non_zero_locations.append(test_locs[location_index])
"""


