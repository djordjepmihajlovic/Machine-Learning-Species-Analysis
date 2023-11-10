"""
Changing decision tree to random forest, checking what I get for probabilities.
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib
import pickle


#Load data
data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

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
new_train_ids = train_ids[flat_wanted_indices]

#Load test data plus reverse dictionary

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 
with open('reverse_dict.pkl', 'rb') as file:
    reverse_test_pos_inds = pickle.load(file)


rdf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')

rdf.fit(new_train_locs, new_train_ids)

predictions = rdf.predict(test_locs)

#class_probabilities = rdf.predict_proba([X_test[27]])[0]

#print(class_probabilities)

"""

reverse_test_pos_inds = {} #Reversing dictionary so that you can check species at given location.

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index] = [] #Creates list at each index
        

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index].append(species_id) #Appends species id to lists created earlier


indices = range(len(test_locs))
indices_0 = [] #List for indices/locations with no species. Uses following loop to fill list.

for index in indices:
    if index not in reverse_test_pos_inds.keys():
        indices_0.append(index)
        
test_pos_inds[0] = indices_0 #Modifies test_pos_inds dictionary to include indices with no species and sets them to key "0"

reverse_test_pos_inds = {} #Reversing dictionary again so that you have the locations with no species referencing a 0.

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index] = []
        

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index].append(species_id)

"""

test_ids = [] #Uses the new reverse dictionary to create set ids to each of the test locations
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    test_ids.append(test_id)



id = 12716
tp = 0
tn = 0
fn = 0
fp = 0
for i in range(len(test_locs)):
    if id in test_ids[i] and predictions[i] == id:
        tp += 1
    elif id in test_ids[i] and predictions[i] != id:
        fn += 1
    elif id not in test_ids[i] and predictions[i] == id:
        fp += 1
    elif id not in test_ids[i] and predictions[i] != id:
        tn += 1
        
print('True positive Turdus Merulus:', tp)
print('True negative Turdus Merulus:', tn)
print('False positive Turdus Merulus:', fp)
print('False negative Turdus Merulus:', fn)

tp = 0
tn = 0
fn = 0
fp = 0

for id in species:
    for i in range(len(test_locs)):
        if id in test_ids[i] and predictions[i] == id:
            tp += 1
        elif id in test_ids[i] and predictions[i] != id:
            fn += 1
        elif id not in test_ids[i] and predictions[i] == id:
            fp += 1
        elif id not in test_ids[i] and predictions[i] != id:
            tn += 1

print('Total True positive:', tp)
print('Total True negative:', tn)
print('Total False positive:', fp)
print('Total False negative:', fn)
