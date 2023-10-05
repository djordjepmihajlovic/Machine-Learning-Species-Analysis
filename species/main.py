# copy pasted from explore_species_data.py; use this file to edit and test so explore_species_data.py is clean

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = np.load('species_train.npz')
train_locs = data['train_locs']
train_ids = data['train_ids']
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

# loading test data 
data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']   
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))    

# data stats
print('Train Stats:')
print('Number of species in train set:           ', len(species))
print('Number of train locations:                ', train_locs.shape[0])
_, species_counts = np.unique(train_ids, return_counts=True)
print('Average number of locations per species:  ', species_counts.mean())
print('Minimum number of locations for a species:', species_counts.min())
print('Maximum number of locations for a species:', species_counts.max())

print(train_locs[0])


