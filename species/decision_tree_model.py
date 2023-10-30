"""
Implementation of the Decision tree model
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

#Load data
data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 

reverse_test_pos_inds = {} #Reversing dictionary so that you can check species at given location.

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index] = species_id

#Decision Tree model
tree_classifier = DecisionTreeClassifier()

#Fitting
tree_classifier.fit(train_locs, train_ids)

#Predictions
predictions = tree_classifier.predict(test_locs)

# Create a list to store valid test_ids and test_locs
valid_test_ids = []
valid_test_locs = []

# Iterate through the indices and append valid test_ids/test_locs
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    if test_id is not None:  # Check if the value is not None (i.e., valid)
        valid_test_ids.append(test_id)
        valid_test_locs.append(test_locs[index])



# Convert to a numpy array
valid_test_ids = np.array(valid_test_ids)
valid_test_locs = np.array(valid_test_locs)

print('Decision tree classification Accuracy: ' + str(tree_classifier.score(valid_test_locs, valid_test_ids)))

#Accuracy of 10-15% which is terrible, I think species with large extent predominate
# Possible improvements: Look at more specific regions? What to do about data imbalance? (2000 vs 50 viewings)

#Printing 10 random locations, the species in them and the prediction.
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
