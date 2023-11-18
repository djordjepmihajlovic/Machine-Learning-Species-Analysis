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

most_sparse = [4345, 44570, 42961, 32861, 2071]

all_lists = [most_sparse]

rng = 0.05
csv_filename1 = 'cf_try_data.csv'
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