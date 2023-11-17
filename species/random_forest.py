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
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import seaborn as sns
import pandas as pd
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

#Balance data
"""
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
"""
#Load test data plus reverse dictionary

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 
with open('reverse_dict.pkl', 'rb') as file:
    reverse_test_pos_inds = pickle.load(file)

rdf = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 15, class_weight="balanced_subsample")
#############################################################################################################################################
# Should keep working on best parameters for max_depth and eventually do it with 100 estimators.
# Using max depth = 18, from graph I think 15 could be argued as good accuracy without over fitting too much
#############################################################################################################################################
#rdf.fit(new_train_locs, new_train_ids)
rdf.fit(train_locs, train_ids_v3)

#predictions = rdf.predict(test_locs)

predictions_p = rdf.predict_proba(test_locs)

test_ids = [] #Uses the new reverse dictionary to create set ids to each of the test locations
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    test_ids.append(test_id)



#NA_species = [890, 10243, 11586, 42223, 15035]
#EU_species = [13851, 472766, 67819, 117054, 201178]
#OC_species = [508981, 20504, 12526, 40908, 1692]
#AF_species = [12832, 14104, 2203, 3813, 4309]
#SA_species = [16006, 5612, 14881, 10079, 20535]
#AS_species = [12821, 8277, 8079, 204523, 113754]
#AN_species = [54549]
most_sparse = [4345, 44570, 42961, 32861, 2071]
most_dense =  [38992, 29976, 8076, 145310, 4569]
larg_dist = [4208, 12716, 145300, 4636, 4146]
small_dist = [35990, 64387, 73903, 6364, 27696]
#Plus total!
all_lists = [most_sparse, most_dense, larg_dist, small_dist]
rng = 0.05
csv_filename1 = 'cf_ditr_data.csv'
with open(csv_filename1, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')

    k = 1
    for list in all_lists:
        true_p = np.zeros((5, 20))
        true_n = np.zeros((5, 20))
        false_p = np.zeros((5, 20))
        false_n = np.zeros((5, 20))
        total = np.zeros((5, 20))

        j = 0
        for id in most_sparse: #####
            id_inx = np.where(species == id)
            for i in range(len(test_locs)):
                for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
                    if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                        true_p[j][idx] += 1
                    elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                        false_n[j][idx] += 1
                    elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                        false_p[j][idx] += 1
                    elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                        true_n[j][idx] += 1
            j += 1
            print(f"Species {j} done.")

        true_p_rate = true_p/(true_p + false_n)
        false_p_rate = false_p/(true_n + false_p)
        precision = true_p/(true_p+false_p)
        recall = true_p_rate

        AUC = []
        for i in range(len(most_sparse)):
            AUC.append(np.abs(np.trapz(y=true_p_rate[i].tolist(), x=false_p_rate[i].tolist())))
        mean_AUC_ROC = sum(AUC)/len(AUC)

        PR_AUC = []
        for i in range(len(most_sparse)):
            PR_AUC.append(np.abs(np.trapz(y=precision[i].tolist(), x=recall[i].tolist())))
        mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

        csv_writer.writerow([f'Iteration {k}'])
        csv_writer.writerow([mean_AUC_ROC])
        csv_writer.writerow([mean_AUC_PR])

csv_filename2 = 'cf_tot_data.csv'
with open(csv_filename1, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')

    true_p = np.zeros((500, 20))
    true_n = np.zeros((500, 20))
    false_p = np.zeros((500, 20))
    false_n = np.zeros((500, 20))
    total = np.zeros((500, 20))

    j = 0
    for id in species: #####
        id_inx = np.where(species == id)
        for i in range(len(test_locs)):
            for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
                if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                    true_p[j][idx] += 1
                elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                    false_n[j][idx] += 1
                elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                    false_p[j][idx] += 1
                elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                    true_n[j][idx] += 1
        j += 1
        print(f"Species {j} done.")
        
    true_p_rate = true_p/(true_p + false_n)
    false_p_rate = false_p/(true_n + false_p)
    precision = true_p/(true_p+false_p)
    recall = true_p_rate

    AUC = []
    for i in range(len(species)):
        AUC.append(np.abs(np.trapz(y=true_p_rate[i].tolist(), x=false_p_rate[i].tolist())))
    mean_AUC_ROC = sum(AUC)/len(AUC) 

    PR_AUC = []
    for i in range(len(species)):
        PR_AUC.append(np.abs(np.trapz(y=precision[i].tolist(), x=recall[i].tolist())))
    mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)
    csv_writer.writerow([f'Iteration 1'])
    csv_writer.writerow([mean_AUC_ROC])
    csv_writer.writerow([mean_AUC_PR])


"""
true_p = np.zeros((5, 20))
true_n = np.zeros((5, 20))
false_p = np.zeros((5, 20))
false_n = np.zeros((5, 20))
total = np.zeros((5, 20))

j = 0
for id in EU_species: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

prec = precision[0].tolist()
rec = recall[0].tolist()
tpr = true_p_rate[0].tolist()
fpr = false_p_rate[0].tolist()

AUC = []

AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

EU_values = [mean_AUC_PR, mean_AUC_ROC]

true_p = np.zeros((5, 20))
true_n = np.zeros((5, 20))
false_p = np.zeros((5, 20))
false_n = np.zeros((5, 20))
total = np.zeros((5, 20))

j = 0
for id in NA_species: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

prec = precision[0].tolist()
rec = recall[0].tolist()
tpr = true_p_rate[0].tolist()
fpr = false_p_rate[0].tolist()


AUC = []

AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

NA_values = [mean_AUC_PR, mean_AUC_ROC]


true_p = np.zeros((5, 20))
true_n = np.zeros((5, 20))
false_p = np.zeros((5, 20))
false_n = np.zeros((5, 20))
total = np.zeros((5, 20))

j = 0
for id in OC_species: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

prec = precision[0].tolist()
rec = recall[0].tolist()
tpr = true_p_rate[0].tolist()
fpr = false_p_rate[0].tolist()


AUC = []

AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

OC_values = [mean_AUC_PR, mean_AUC_ROC]


true_p = np.zeros((5, 20))
true_n = np.zeros((5, 20))
false_p = np.zeros((5, 20))
false_n = np.zeros((5, 20))
total = np.zeros((5, 20))

j = 0
for id in SA_species: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

prec = precision[0].tolist()
rec = recall[0].tolist()
tpr = true_p_rate[0].tolist()
fpr = false_p_rate[0].tolist()


AUC = []

AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

SA_values = [mean_AUC_PR, mean_AUC_ROC]



true_p = np.zeros((5, 20))
true_n = np.zeros((5, 20))
false_p = np.zeros((5, 20))
false_n = np.zeros((5, 20))
total = np.zeros((5, 20))

j = 0
for id in AF_species: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

prec = precision[0].tolist()
rec = recall[0].tolist()
tpr = true_p_rate[0].tolist()
fpr = false_p_rate[0].tolist()


AUC = []

AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

AF_values = [mean_AUC_PR, mean_AUC_ROC]




true_p = np.zeros((1, 20))
true_n = np.zeros((1, 20))
false_p = np.zeros((1, 20))
false_n = np.zeros((1, 20))
total = np.zeros((1, 20))

j = 0
for id in AN_species: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

prec = precision[0].tolist()
rec = recall[0].tolist()
tpr = true_p_rate[0].tolist()
fpr = false_p_rate[0].tolist()


AUC = []

AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
#AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
#AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
#AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
#AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
#PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
#PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
#PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
#PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

AN_values = [mean_AUC_PR, mean_AUC_ROC]


true_p = np.zeros((5, 20))
true_n = np.zeros((5, 20))
false_p = np.zeros((5, 20))
false_n = np.zeros((5, 20))
total = np.zeros((5, 20))

j = 0
for id in AS_species: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

prec = precision[0].tolist()
rec = recall[0].tolist()
tpr = true_p_rate[0].tolist()
fpr = false_p_rate[0].tolist()



AUC = []

AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

AS_species = [mean_AUC_PR, mean_AUC_ROC]

data = {
    'Continent': ['North America', 'South America',
                  'Europe',  'Asia', 'Africa',
                  'Oceania', 'Antarctica'],
    'PR': [0.18768909895697442, 0.3396720484455499, 0.30884048357588123, 0.14939715128808032,
            0.08060646746497149, 0.17634565643525182, 0.14852307375535972],
    'ROC': [0.902208149345934, 0.8895024854903453, 0.8354512160672968, 0.9112733952498072,
            0.9146056483588995, 0.9426947499778434, 0.9172889823508424]
}


# Create a DataFrame
df = pd.DataFrame(data)

# Set the color palette for PR and ROC bars
colors = ["red", "blue"]

sns.barplot(x='Continent', y='PR', data=df, color=colors[0], label= 'PR')
sns.barplot(x='Continent', y='ROC', data=df, color=colors[1], alpha=0.5, label= 'ROC')

# Customize the plot
plt.title('PR and ROC values by Continent')
plt.xlabel('Continent')
plt.ylabel('Values')
plt.legend()#title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()
"""
"""
data = {
    'Continent': ['North America', 'South America',
                  'Europe',  'Asia', 'Africa',
                  'Oceania', 'Antarctica'],
    'PR': [0.1606, 0.3276, 0.3185, 0.1753,
            0.087, 0.1578, 0.127],
    'ROC': [0.91, 0.9067, 0.82, 0.925,
           0.9069, 0.9444, 0.9289]
}


# Create a DataFrame
df = pd.DataFrame(data)

# Set the color palette for PR and ROC bars
colors = ["red", "blue"]

sns.barplot(x='Continent', y='PR', data=df, color=colors[0], label= 'PR')
sns.barplot(x='Continent', y='ROC', data=df, color=colors[1], alpha=0.5, label= 'ROC')

# Customize the plot
plt.title('PR and ROC values by Continent')
plt.xlabel('Continent')
plt.ylabel('Values')
plt.legend()#title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()

"""

"""
sns.barplot(x=most_dense, y=AUC, color='b')########
plt.xlabel('Species')
plt.ylabel('AUC-ROC')
plt.show()

sns.barplot(x=most_dense, y=PR_AUC, color='r')########
plt.xlabel('Species')
plt.ylabel('AUC-Precision-Recall')
plt.ylim(0, 1)
plt.show()

sns.barplot(x=['mean_AUC_ROC', 'mean_AUC_PR'], y=[mean_AUC_ROC, mean_AUC_PR], color='r')
plt.xlabel('Most Dense Averages')###############
plt.ylabel('AUC')
plt.ylim(0, 1)
plt.show()
"""
"""
true_p = np.zeros((5, 20))
true_n = np.zeros((5, 20))
false_p = np.zeros((5, 20))
false_n = np.zeros((5, 20))
total = np.zeros((5, 20))

j = 0
for id in larg_dist: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

AUC = []

AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

sns.barplot(x=larg_dist, y=AUC, color='b')########
plt.xlabel('Species')
plt.ylabel('AUC-ROC')
plt.show()

sns.barplot(x=larg_dist, y=PR_AUC, color='r')########
plt.xlabel('Species')
plt.ylabel('AUC-Precision-Recall')
plt.ylim(0, 1)
plt.show()

sns.barplot(x=['mean_AUC_ROC', 'mean_AUC_PR'], y=[mean_AUC_ROC, mean_AUC_PR], color='r')
plt.xlabel('Largest Distance Averages')###############
plt.ylabel('AUC')
plt.ylim(0, 1)
plt.show()

true_p = np.zeros((5, 20))
true_n = np.zeros((5, 20))
false_p = np.zeros((5, 20))
false_n = np.zeros((5, 20))
total = np.zeros((5, 20))

j = 0
for id in small_dist: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

AUC = []

AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

sns.barplot(x=small_dist, y=AUC, color='b')########
plt.xlabel('Species')
plt.ylabel('AUC-ROC')
plt.show()

sns.barplot(x=small_dist, y=PR_AUC, color='r')########
plt.xlabel('Species')
plt.ylabel('AUC-Precision-Recall')
plt.ylim(0, 1)
plt.show()

sns.barplot(x=['mean_AUC_ROC', 'mean_AUC_PR'], y=[mean_AUC_ROC, mean_AUC_PR], color='r')
plt.xlabel('Smallest Distance Averages')###############
plt.ylabel('AUC')
plt.ylim(0, 1)
plt.show()

true_p = np.zeros((5, 20))
true_n = np.zeros((5, 20))
false_p = np.zeros((5, 20))
false_n = np.zeros((5, 20))
total = np.zeros((5, 20))

j = 0
for id in most_sparse: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

AUC = []

AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

sns.barplot(x=most_sparse, y=AUC, color='b')########
plt.xlabel('Species')
plt.ylabel('AUC-ROC')
plt.show()

sns.barplot(x=most_sparse, y=PR_AUC, color='r')########
plt.xlabel('Species')
plt.ylabel('AUC-Precision-Recall')
plt.ylim(0, 1)
plt.show()

sns.barplot(x=['mean_AUC_ROC', 'mean_AUC_PR'], y=[mean_AUC_ROC, mean_AUC_PR], color='r')
plt.xlabel('Most Sparse averages')###############
plt.ylabel('AUC')
plt.ylim(0, 1)
plt.show()

"""
"""
true_p = np.zeros((500, 20))
true_n = np.zeros((500, 20))
false_p = np.zeros((500, 20))
false_n = np.zeros((500, 20))
total = np.zeros((500, 20))

j = 0
for id in species: #####
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                true_p[j][idx] += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                false_n[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > thr:
                false_p[j][idx] += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                true_n[j][idx] += 1
    j += 1
    print(f"Species {j} done.")
    


true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)
precision = true_p/(true_p+false_p)
recall = true_p_rate

AUC = []

for i in range(len(species)):
    AUC.append(np.abs(np.trapz(y=true_p_rate[i].tolist(), x=false_p_rate[i].tolist())))

#AUC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
#AUC.append(np.abs(np.trapz(y=true_p_rate[1].tolist(), x=false_p_rate[1].tolist())))
#AUC.append(np.abs(np.trapz(y=true_p_rate[2].tolist(), x=false_p_rate[2].tolist())))
#AUC.append(np.abs(np.trapz(y=true_p_rate[3].tolist(), x=false_p_rate[3].tolist())))
#AUC.append(np.abs(np.trapz(y=true_p_rate[4].tolist(), x=false_p_rate[4].tolist())))

print('ROC AUC values are: ', AUC)

mean_AUC_ROC = sum(AUC)/len(AUC)

print('Mean ROC AUC is', mean_AUC_ROC)

PR_AUC = []

for i in range(len(species)):
    PR_AUC.append(np.abs(np.trapz(y=precision[i].tolist(), x=recall[i].tolist())))

#PR_AUC.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))
#PR_AUC.append(np.abs(np.trapz(y=precision[1].tolist(), x=recall[1].tolist())))
#PR_AUC.append(np.abs(np.trapz(y=precision[2].tolist(), x=recall[2].tolist())))
#PR_AUC.append(np.abs(np.trapz(y=precision[3].tolist(), x=recall[3].tolist())))
#PR_AUC.append(np.abs(np.trapz(y=precision[4].tolist(), x=recall[4].tolist())))

print('PR AUC values are: ', PR_AUC)

mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)

print('Mean PR AUC is', mean_AUC_PR)

sns.barplot(x=['mean_AUC_ROC', 'mean_AUC_PR'], y=[mean_AUC_ROC, mean_AUC_PR], color='r')
plt.xlabel('TOTAL AVERAGES')###############
plt.ylabel('AUC')
plt.ylim(0, 1)
plt.show()
"""
"""
#prec1 = precision[2].tolist()
#rec1 = recall[2].tolist()
#prec2 = precision[3].tolist()
#rec2 = recall[3].tolist()

#test_locs_id = test_pos_inds.get(35990)
#print(test_locs_id)
#print(prec1)
#print(true_p_rate[2].tolist())
#print(rec1)

#plt.plot(rec1, prec1)
#plt.show()

#plt.plot(rec2, prec2)
#plt.show()


id = 12716
tp = 0
tn = 0
fn = 0
fp = 0
for i in range(len(test_locs)):
    if id in test_ids[i] and species[predictions[i]] == id:
        tp += 1
    elif id in test_ids[i] and species[predictions[i]] != id:
        fn += 1
    elif id not in test_ids[i] and species[predictions[i]] == id:
        fp += 1
    elif id not in test_ids[i] and species[predictions[i]] != id:
        tn += 1
        
print('True positive Turdus Merulus:', tp)
print('True negative Turdus Merulus:', tn)
print('False positive Turdus Merulus:', fp)
print('False negative Turdus Merulus:', fn)

###################### 

id = 12716
id_inx = np.where(species == id)
tp = 0
tn = 0
fn = 0
fp = 0
"""
"""
print('proabilities in random location =', predictions_p[156789])
print('length of prediction array', len(predictions_p[156789]))
prob_indeces = []
for index in range(len(predictions_p[156789])):
    if predictions_p[156789][index] > 0:
        prob_indeces.append(index)
    else:
        continue
print('indeces with probability above 0 in loc =', prob_indeces)
print('predicition in same random location =', predictions[156789])
print('id of prediction', species[predictions[156789]])
#print('index of prediction', np.where(species == predictions[156789])[0])
print('index of prediction', np.where(range_list == predictions[156789])[0])
print('actual species in location', test_ids[156789])
real_indeces = []
for id in test_ids[156789]:
    real_indeces.append(np.where(species == id)[0])
print('indices of species in location:', real_indeces)

id = 12716
"""
"""
threshold = 0.001
for i in range(len(test_locs)):
    if id in test_ids[i] and predictions_p[i][id_inx[0]] > threshold:
        tp += 1
    elif id in test_ids[i] and predictions_p[i][id_inx[0]] < threshold:
        fn += 1
    elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > threshold:
        fp += 1
    elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < threshold:
        tn += 1
        
print('True positive Turdus Merulus w/ prob:', tp)
print('True negative Turdus Merulus w/ prob:', tn)
print('False positive Turdus Merulus w/ prob:', fp)
print('False negative Turdus Merulus w/ prob:', fn)

"""
"""
tp = 0
tn = 0
fn = 0
fp = 0

for id in species:
    id_inx = np.where(species == id)
    for i in range(len(test_locs)):
        if id in test_ids[i] and predictions_p[i][id_inx[0]] > 0.025:
            tp += 1
        elif id in test_ids[i] and predictions_p[i][id_inx[0]] < 0.025:
            fn += 1
        elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > 0.025:
            fp += 1
        elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < 0.025:
            tn += 1

print('Total True positive w/ probs:', tp)
print('Total True negative w/ probs:', tn)
print('Total False positive w/ probs:', fp)
print('Total False negative w/ probs:', fn)
"""
"""
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

"""



"""

id = 12716 # turdus merula
index_TM = spec_dict.get(id)
id_index = np.where(rdf.classes_ == index_TM)[0][0] ### OBVIAMENTE ESTO NO FUNCIONA...
print(id_index)

n_gridpoints = 500
lats = np.linspace(-90, 90, n_gridpoints)
longs = np.linspace(-180, 180, n_gridpoints)
#pvals = np.zeros((n_gridpoints, n_gridpoints))

for i in range(n_gridpoints):
    for j in range(n_gridpoints):
        pvals[i,j] = rdf.predict_proba(np.array([lats[i], longs[j]]).reshape(1,-1))[0, id_index]

#file_path = 'pvals.npy'

# Save the pvals array to the specified file
#np.save(file_path, pvals)

pvals = np.load('pvals.npy')
#print(pvals.max())
#print(pvals.min())
#sns.set_theme()
X, Y = np.meshgrid(longs, lats)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
ax = world.plot(figsize=(10, 6))
ax.set_xticks([])
ax.set_yticks([])
cs = ax.contourf(X, Y, pvals, levels = np.linspace(0.01, 0.2, 10), alpha = 0.5, cmap = 'plasma')
#ax.clabel(cs, inline = True)
plt.show() 
"""

"""
sp = 12716
test_inds_pos_TM = np.where(predictions == sp)[0]

geometry = [Point(xy) for xy in zip(test_locs[test_inds_pos_TM, 1], test_locs[test_inds_pos_TM, 0])] # gets list of (lat,lon) pairs
gdf = GeoDataFrame(geometry=geometry) # creates geopandas dataframe of these pairs

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) # world map included with geopandas, could download other maps
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='k', markersize=5)
plt.title(str(sp) + ' - ' + species_names[sp])
plt.show()
"""

"""
threshold = 0.001
TPR_list = []
FPR_list = []
Precision_list = []

for id in sp_iden:
    id_inx = np.where(species == id)
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(test_locs)):
        if id in test_ids[i] and predictions_p[i][id_inx[0]] > threshold:
            tp += 1
        elif id in test_ids[i] and predictions_p[i][id_inx[0]] < threshold:
            fn += 1
        elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > threshold:
            fp += 1
        elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < threshold:
            tn += 1
    tpr = tp/(tp+fn)
    TPR_list.append(tpr)
    fpr = fp/(fp+tn)
    FPR_list.append(fpr)
    precision = tp/(tp+fp)
    Precision_list.append(precision)

print(TPR_list)
print(FPR_list)
print(Precision_list)
"""

