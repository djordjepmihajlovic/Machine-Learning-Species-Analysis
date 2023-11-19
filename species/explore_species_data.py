"""
Script demonstrating simple data loading and visualization.

Data Format: 
There are two files 'species_train.npz', and 'species_test.npz'
For the train data, we have the geographical coordinates where different 
species have been observed. This data has been collected by citizen scientists 
so it is noisy. 
For the test data, we have a set of locations for all species from the training, 
set and for each location we know if a species is present there or not. 

You can find out information about each species by appending the taxon_id to this 
URL, e.g. for 22956: 'Leptodactylus mystacinus', the URL is: 
https://www.inaturalist.org/taxa/22956
note some species might not be on the website anymore

Possible questions to explore: 
 - train a separate model to predict what locations a species of interest is present 
 - train a single model instead of per species model 
 - how to deal with "positive only data"
 - dealing with noisy/biased training data
 - using other input features e.g. climate data from  WorldClim Bioclimatic 
   variables  https://www.worldclim.org/data/worldclim21.html
 - how to evaluate e.g. what is a good metric to use?

Additional ideas:
  - 
 
Data sources:
 -  train data is from iNaturalist -  www.inaturalist.org
 -  test data is IUCN - https://www.iucnredlist.org/resources/spatial-data-download
"""


import numpy as np
import matplotlib.pyplot as plt

# loading training data    
data = np.load('species_train.npz')
train_locs = data['train_locs']  # 2D array, rows are number of datapoints and 
                                 # columns are "latitude" and "longitude"
train_ids = data['train_ids']    # 1D array, entries are the ID of the species 
                                 # that is present at the corresponding location in train_locs
species = data['taxon_ids']      # list of species IDe. Note these do not necessarily start at 0 (or 1)
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))  # latin names of species 

# loading test data 
data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']    # 2D array, rows are number of datapoints 
                                      # and columns are "latitude" and "longitude"
# data_test['test_pos_inds'] is a list of lists, where each list corresponds to 
# the indices in test_locs where a given species is present, it can be assumed 
# that they are not present in the other locations 
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))     

# data stats
print('Train Stats:')
print('Number of species in train set:           ', len(species)) #500
print('Number of train locations:                ', train_locs.shape[0]) #272037
_, species_counts = np.unique(train_ids, return_counts=True)
print('Average number of locations per species:  ', species_counts.mean()) #544
print('Minimum number of locations for a species:', species_counts.min()) #50
print('Maximum number of locations for a species:', species_counts.max()) #2000

print('Number of test locations:'                 , test_locs.shape) #288122 locations


total_elements = sum(len(sublist) for sublist in data_test['test_pos_inds'])

# Step 2: Calculate the total number of sublists
total_sublists = len(data_test['test_pos_inds'])

# Step 3: Calculate the average
average_elements_per_list = total_elements / total_sublists

print("Average elements per list:", average_elements_per_list)


# plot train and test data for a random species
plt.close('all')
plt.figure(0)

sp = np.random.choice(species)
sp = 7729
print('\nDisplaying random species:')
print(str(sp) + ' - ' + species_names[sp]) 

# get test locations and plot
# test_inds_pos is the locations where the selected species is present
# test_inds_neg is the locations where the selected species is not present
test_inds_pos = test_pos_inds[sp]  
test_inds_neg = np.setdiff1d(np.arange(test_locs.shape[0]), test_pos_inds[sp])
plt.plot(test_locs[test_inds_pos, 1], test_locs[test_inds_pos, 0], 'b.', label='test')

# get train locations and plot
train_inds_pos = np.where(train_ids == sp)[0]
plt.plot(train_locs[train_inds_pos, 1], train_locs[train_inds_pos, 0], 'rx', label='train')

plt.title(str(sp) + ' - ' + species_names[sp])
plt.grid(True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.ylabel('latitude')
plt.xlabel('longitude')
plt.legend()
plt.show()


TM_total = len(test_pos_inds[12716]) #Number of indeces for Turdus Merula
print('TM total =', TM_total) # I got 9197



