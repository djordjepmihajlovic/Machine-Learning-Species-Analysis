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
data = np.load('../../data/species_train.npz')
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

data_test = np.load('../../data/species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 
with open('reverse_dict.pkl', 'rb') as file:
    reverse_test_pos_inds = pickle.load(file)

rdf = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 10, class_weight="balanced_subsample")

rdf.fit(train_locs, train_ids_v3)

#predictions = rdf.predict(test_locs)

predictions_p = rdf.predict_proba(test_locs)

test_ids = [] #Uses the new reverse dictionary to create set ids to each of the test locations
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    test_ids.append(test_id)


most_sparse = [4345, 44570, 42961, 32861, 2071]
most_dense =  [38992, 29976, 8076, 145310, 4569]
larg_dist = [4208, 12716, 145300, 4636, 4146]
small_dist = [35990, 64387, 73903, 6364, 27696]
#Plus total!
all_lists = [most_sparse, most_dense, larg_dist, small_dist]
rng = 0.05
csv_filename1 = 'cf_ditr_data2.csv'
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
        for id in list: #####
            id_inx = np.where(species == id)
            for i in range(len(test_locs)):
                for idx, thr in enumerate(np.linspace(0.0, rng, 20)): #Maybe use 4 or 5 to start?
                    if id in test_ids[i] and predictions_p[i][id_inx[0]] >= thr:
                        true_p[j][idx] += 1
                    elif id in test_ids[i] and predictions_p[i][id_inx[0]] < thr:
                        false_n[j][idx] += 1
                    elif id not in test_ids[i] and predictions_p[i][id_inx[0]] >= thr:
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
        k +=1

csv_filename2 = 'cf_tot_data.csv'
with open(csv_filename2, 'w', newline='') as csvfile:
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

    prec = []
    for i in range(len(list)):
        prec.append(precision[i][-1])
    mean_prec = sum(prec)/len(prec)
        
    rec = []
    for i in range(len(list)):
        rec.append(recall[i][-1])
    mean_rec = sum(rec)/len(rec)

    f2 = 5*mean_prec*mean_rec/(4*mean_prec + mean_rec)


    csv_writer.writerow([f'Iteration 1'])
    csv_writer.writerow([mean_AUC_ROC])
    csv_writer.writerow([mean_AUC_PR])
    csv_writer.writerow([f2])


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



