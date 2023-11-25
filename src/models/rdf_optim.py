import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# necessary to get rid of annoying scipy warning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
   
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
"""
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
new_train_ids = train_ids_v3[flat_wanted_indices]
"""
# test data
data_test = np.load('species_test.npz', allow_pickle=True) 
test_locs = data_test['test_locs']
test_ids = data_test['taxon_ids']
test_species = np.unique(test_ids)
num_locs = len(test_locs)
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))

def get_f1_score(id, threshold, probs):
    # array with 1s if species is present at that index in test_locs, 0 otherwise
    pos_inds = test_pos_inds[id]
    true = np.zeros(num_locs, dtype = int)
    true[pos_inds] = 1 
    #id_index = np.where(rdf.classes_ == id)[0][0]
    index_sp = spec_dict.get(id)
    id_index = np.where(rdf.classes_ == index_sp)[0][0]

    # predicted probabilities
    pred = np.zeros(num_locs)
    probs_bin = (probs >= threshold).astype(int)
    pred = probs_bin[:,id_index]

    f1 = f1_score(true, pred)
    return f1


depths = np.array([5,10,25,50,75,100,150,250,500])
mean_vals = np.zeros(len(depths))

t = 1
for depth in depths:

    rdf = RandomForestClassifier(n_estimators=10, max_depth = depth, class_weight="balanced_subsample")
    rdf.fit(train_locs, train_ids_v3)
    # get probability values for each id and each test loc
    probs = rdf.predict_proba(test_locs) 
    f1 = np.zeros(len(test_species))
    for i in range(len(test_species)):
        f1[i] = get_f1_score(test_species[i], 0.05, probs)
    mean_vals[np.where(depths == depth)[0][0]] = np.mean(f1)
    #print(np.mean(f1))
    print("depth", t)
    t += 1

## COULD KEEP VALUES IN A FILE AND RUN IT IN SCHOOL COMP, THEN PLOT IT HERE...
#print(np.max(mean_vals))
#print(k_vals[np.where(mean_vals == np.max(mean_vals))[0][0]])
plt.xlabel(r'$Max Depth$', fontsize = 10)
plt.ylabel('Mean F1 score', fontsize = 10)
plt.plot(depths, mean_vals, marker='o', color = 'k')
plt.savefig('knn_f1_scores.png')
plt.show()