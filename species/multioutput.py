"""
MultiOutput for decision tree
"""

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import tree
import random
import matplotlib.pyplot as plt
import pickle

#TRAIN DATA
data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

#TRAIN DATA BALANCING

species_count = np.bincount(train_ids)
sp_list_a = [] 
sp_list_b = []
train_inds_pos_a = []
train_inds_pos_b= []
wanted_indices = []

i = 0
for n in species_count:
    if n > 500: ##############Number here
        sp_list_a.append(i) 
    elif n != 0:
        sp_list_b.append(i)
    i = i + 1

for species_id in sp_list_a:
    train_inds_pos_a.append(np.where(train_ids == species_id)[0])

for species_id in sp_list_b:
    train_inds_pos_b.append(np.where(train_ids == species_id)[0])

for sp_indices in train_inds_pos_a:
    sp_choice = np.random.choice(sp_indices, 500, replace = False) #Number here
    wanted_indices.append(sp_choice)

for sp_indices in train_inds_pos_b:
    wanted_indices.append(sp_indices)

flat_wanted_indices = [item for sublist in wanted_indices for item in sublist]
new_train_locs = train_locs[flat_wanted_indices]
new_train_ids = train_ids[flat_wanted_indices]


#TEST DATA + REVERSE DICT

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 
with open('reverse_dict.pkl', 'rb') as file:
    reverse_test_pos_inds = pickle.load(file)



tree_classifier = tree.DecisionTreeClassifier()

multioutput_tree_classifier = MultiOutputClassifier(tree_classifier)

multioutput_tree_classifier.fit(new_train_locs, new_train_ids)

class_probabilities_multioutput = multioutput_tree_classifier.predict_proba([test_locs[0]])[0]

print(class_probabilities_multioutput)

"""
Sadly doesnt work for single label train data ... :(
"""


