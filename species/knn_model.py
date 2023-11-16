import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# necessary to get rid of annoying scipy warning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#Load data
data = np.load('species/species_train.npz')
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

# test data
data_test = np.load('species/species_test.npz', allow_pickle=True) 
test_locs = data_test['test_locs']
num_locs = len(test_locs)
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))

# k nearest neighbours classifier, optimal k found by examining F1 scores
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(new_train_locs, new_train_ids)
# get probability values for each id and each test loc
probs = knn.predict_proba(test_locs)

def errors(id, threshold):
    # array with 1s if species is present at that index in test_locs, 0 otherwise
    pos_inds = test_pos_inds[id]
    true = np.zeros(num_locs, dtype = int)
    true[pos_inds] = 1 
    id_index = spec_dict[id]

    # true postives, false positives, false negatives, true negatives
    tp, fp, tn, fn = (np.zeros(num_locs) for i in range (4))

    # predicted probabilities
    pred = np.zeros(num_locs)
    probs_bin = (probs >= threshold).astype(int)
    pred = probs_bin[:,id_index]

    # find tp, tn, fn, fp
    t = true[true == pred]
    f = true[true != pred]
    tp = len(t[t == 1])
    tn = len(t[t == 0])
    fn = len(f[f == 1])
    fp = len(f[f == 0])

    if(tp == 0 and fp == 0):
        print('Divide by zero found for threshold '+str(threshold))
        print('tp: '+str(tp))
        print('tn: '+str(tn))
        print('fp: '+str(fp))
        print('fn: '+str(fn))
    # compute precision, recall = tpr, fpr
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fpr = fp/(fp+tn)
    return prec, rec, fpr

# calculates precision, recall, false positive rate
def accuracy_measures(id, num_thresholds):
    thresholds = np.linspace(0, 0.5, num_thresholds)
    precision_vals, recall_vals, false_postive_rate_vals = (np.zeros(num_thresholds) for i in range(3))

    i = 0
    for t in thresholds:
        acc = errors(id, t)
        precision_vals[i] = acc[0]
        recall_vals[i] = acc[1]
        false_postive_rate_vals[i] = acc[2]
        i += 1
    return precision_vals, recall_vals, false_postive_rate_vals

# gets auc for precision-recall curve and roc
def auc(id, num_thresholds):
    prec, rec, fpr = accuracy_measures(id, num_thresholds)
    pr_inds = np.argsort(rec)
    roc_inds = np.argsort(fpr)

    prec = prec[pr_inds]
    tpr = rec[roc_inds]
    rec = rec[pr_inds]
    fpr = fpr[roc_inds]
     
    auc_pr = np.trapz(prec, rec)
    auc_roc = np.trapz(tpr, fpr)
    return auc_pr, auc_roc

# ids under consideration
sparsest = [4345, 44570, 42961, 32861, 2071]
densest = [38992, 29976, 8076, 145310, 4569]
largest = [4208, 12716, 145300, 4636, 4146]
smallest = [35990, 64387, 73903, 6364, 27696]
datasets = [sparsest, densest, largest, smallest]
names = ['sparsest', 'densest', 'largest', 'smallest']

j = 0
for data in datasets:
    pr = np.zeros(5)
    roc = np.zeros(5)

    for i in range(5):
        pr[i] = auc(data[i], 20)[0]
        roc[i] = auc(data[i], 20)[1]

    print(names[j])
    print('PR: ' + np.array2string(pr))
    print('PR mean: ' + str(np.mean(pr)))
    print('ROC: ' + np.array2string(roc))
    print('ROC mean: ' + str(np.mean(roc)))
    print('*****')
    j += 1