from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import csv

#Load data
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

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 
with open('reverse_dict.pkl', 'rb') as file:
    reverse_test_pos_inds = pickle.load(file)    

test_ids = [] #Uses the new reverse dictionary to create set ids to each of the test locations
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    test_ids.append(test_id)

rdf = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 15) #, class_weight="balanced_subsample")

#rdf.fit(new_train_locs, new_train_ids)
rdf.fit(train_locs, train_ids_v3)

predictions_p = rdf.predict_proba(test_locs)


NA_species = [890, 10243, 11586, 42223, 15035]
EU_species = [13851, 472766, 67819, 117054, 201178]
OC_species = [508981, 20504, 12526, 40908, 1692]
AF_species = [12832, 14104, 2203, 3813, 4309]
SA_species = [16006, 5612, 14881, 10079, 20535]
AS_species = [12821, 8277, 8079, 204523, 113754]
AN_species = [54549]
most_sparse = [4345, 44570, 42961, 32861, 2071]
most_dense =  [38992, 29976, 8076, 145310, 4569]
larg_dist = [4208, 12716, 145300, 4636, 4146]
small_dist = [35990, 64387, 73903, 6364, 27696]

thr = 0.05


#all_lists = [NA_species, EU_species, OC_species, AF_species, SA_species, AS_species, most_sparse, most_dense, larg_dist, small_dist]
all_lists = [AN_species]

csv_filename = 'totalcf_data.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')

    k = 0
    for list in all_lists:
        true_p = 0#np.zeros((5, 20))
        true_n = 0#np.zeros((5, 20))
        false_p = 0#np.zeros((5, 20))
        false_n = 0#np.zeros((5, 20))

        j = 0

        for id in list:
            id_inx = np.where(species == id)
            for i in range(len(test_locs)):
                if id in test_ids[i] and predictions_p[i][id_inx[0]] >= thr:
                    true_p+= 1
                elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                    false_n += 1
                elif id not in test_ids[i] and predictions_p[i][id_inx[0]] >= thr:
                    false_p += 1
                elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                    true_n += 1
            j += 1
            print(f"Species {j} done.")
        
        csv_writer.writerow([f'Iteration {k}'])
        csv_writer.writerows([[true_p]])
        csv_writer.writerows([[true_n]])
        csv_writer.writerows([[false_p]])
        csv_writer.writerows([[false_n]])
        k +=1


    




