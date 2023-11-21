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
from numpy import genfromtxt

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

###### NEW TRAIN DATA IN SPECIES_TRAIN_7_FEATURES
"""
features_train = genfromtxt('species_train_8_features.csv', delimiter=',')
features_test = genfromtxt('species_test_8_features.csv', delimiter=',')

list_remove = [] # making a list of indexes to remove

for idx, i in enumerate(features_train):
    if i[2] == i[3] == i[4] == i[5] == i[6] == i[7] == 0:
        list_remove.append(idx)

# removing ocean data
features_train = np.array([j for i, j in enumerate(features_train) if i not in list_remove])

#removing ocean data
train_ids_v3 = np.array([j for i, j in enumerate(train_ids_v3) if i not in list_remove])

print(len(features_train))
print(len(train_ids_v3))
print('done...')
"""
#Load test data plus reverse dictionary

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 
with open('reverse_dict.pkl', 'rb') as file:
    reverse_test_pos_inds = pickle.load(file)

test_ids = [] #Uses the new reverse dictionary to create set ids to each of the test locations
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    test_ids.append(test_id)


rdf = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 15, class_weight="balanced_subsample", random_state= 42)
#############################################################################################################################################
# Should keep working on best parameters for max_depth and eventually do it with 100 estimators.
# Using max depth = 18, from graph I think 15 could be argued as good accuracy without over fitting too much
#############################################################################################################################################
#rdf.fit(new_train_locs, new_train_ids)
#rdf.fit(features_train, train_ids_v3)
rdf.fit(train_locs, train_ids_v3)

#predictions = rdf.predict(test_locs)

predictions_p = rdf.predict_proba(test_locs)

most_sparse = [4345, 44570, 42961, 32861, 2071]
most_dense =  [38992, 29976, 8076, 145310, 4569]
larg_dist = [4208, 12716, 145300, 4636, 4146]
small_dist = [35990, 64387, 73903, 6364, 27696]

all_lists = [most_sparse, most_dense, larg_dist, small_dist]
#all_lists = [larg_dist]

rng = 0.05
csv_filename1 = 'old_cf_f2.csv'
with open(csv_filename1, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')

    k = 1
    for list in all_lists:
        true_p = np.zeros((len(list), 20))
        true_n = np.zeros((len(list), 20))
        false_p = np.zeros((len(list), 20))
        false_n = np.zeros((len(list), 20))
        total = np.zeros((len(list), 20))

        j = 0
        for id in list: #####
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

        prec = []
        for i in range(len(list)):
            prec.append(precision[i][-1])
        mean_prec = sum(prec)/len(prec)
        
        rec = []
        for i in range(len(list)):
            rec.append(recall[i][-1])
        mean_rec = sum(rec)/len(rec)

        f1 = 2*mean_prec*mean_rec/(mean_prec + mean_rec)
        f2 = 5*mean_prec*mean_rec/(4*mean_prec + mean_rec)
        """
        AUC = []
        for i in range(len(list)):
            AUC.append(np.abs(np.trapz(y=true_p_rate[i].tolist(), x=false_p_rate[i].tolist())))
        mean_AUC_ROC = sum(AUC)/len(AUC)

        PR_AUC = []
        for i in range(len(list)):
            PR_AUC.append(np.abs(np.trapz(y=precision[i].tolist(), x=recall[i].tolist())))
        mean_AUC_PR = sum(PR_AUC)/len(PR_AUC)
        """
        csv_writer.writerow([f'Iteration {k}'])
        #csv_writer.writerow([mean_AUC_ROC])
        #csv_writer.writerow([mean_AUC_PR])
        csv_writer.writerow([f1])
        csv_writer.writerow([f2])
        k += 1



"""
id = 12716 # turdus merula
index_TM = spec_dict.get(id)
id_index = np.where(rdf.classes_ == index_TM)[0][0] ### OBVIAMENTE ESTO NO FUNCIONA...
print(id_index)

n_gridpoints = 500
lats = np.linspace(-90, 90, n_gridpoints)
longs = np.linspace(-180, 180, n_gridpoints)
pvals = np.zeros((n_gridpoints, n_gridpoints))

for i in range(n_gridpoints):
    for j in range(n_gridpoints):
        pvals[i,j] = rdf.predict_proba(np.array([lats[i], longs[j]]).reshape(1,-1))[0, id_index]

file_path = 'pvals_newfeatures.npy'

# Save the pvals array to the specified file
np.save(file_path, pvals)

#pvals = np.load('pvals.npy')
#print(pvals.max())
#print(pvals.min())
#sns.set_theme()
"""
"""
X, Y = np.meshgrid(longs, lats)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
ax = world.plot(figsize=(10, 6))
ax.set_xticks([])
ax.set_yticks([])
cs = ax.contourf(X, Y, pvals, levels = np.linspace(0.01, 0.2, 10), alpha = 0.5, cmap = 'plasma')
#ax.clabel(cs, inline = True)
plt.show()
"""