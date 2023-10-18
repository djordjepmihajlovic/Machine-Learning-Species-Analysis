"""
Implementation of the Decision tree model
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 

reverse_test_pos_inds = {}

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index] = species_id

tree_classifier = DecisionTreeClassifier()

tree_classifier.fit(train_locs, train_ids)

predictions = tree_classifier.predict(test_locs)

sample_indices = random.sample(range(len(test_locs)), k=10)

i = 0
for index in sample_indices:
    i += 1
    real_species_id = reverse_test_pos_inds.get(index)
    predicted_species_id = predictions[index]
    print(i, '. predicted species name:', species_names[predicted_species_id])
    print(i, '. real species name:', species_names[real_species_id])

#Results are not great, implementing a resampling or similar to get rid of the data
#imbalance I think might help, as one of the species is being predicted multiple times
