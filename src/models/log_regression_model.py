import numpy as np 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle

#Load data
data = np.load('../../data/species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

"""
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
"""
# test data
data_test = np.load('../../data/species_test.npz', allow_pickle=True) 
test_locs = data_test['test_locs']
test_ids = data_test['taxon_ids']
test_species = np.unique(test_ids)
num_locs = len(test_locs)
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))

# logistic regression classifier
"""
lr = LogisticRegression(class_weight='balanced')

lr.fit(train_locs, train_ids)
with open('lr_model_balanced.pkl','wb') as f:
    pickle.dump(lr,f)
"""
with open('lr_model_balanced.pkl', 'rb') as f:
    lr = pickle.load(f)

# get probability values for each id and each test loc
probs = lr.predict_proba(test_locs)

num_thresholds = 20

def errors(id, threshold):
    # array with 1s if species is present at that index in test_locs, 0 otherwise
    pos_inds = test_pos_inds[id]
    true = np.zeros(num_locs, dtype = int)
    true[pos_inds] = 1 
    id_index = np.where(lr.classes_ == id)[0][0]

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

    return tp, tn, fp, fn

def pr_auc(id, threshold):
    thresholds = np.linspace(0, threshold, num_thresholds)
    prec = np.zeros(num_thresholds)
    rec = np.zeros(num_thresholds)

    i = 0
    for t in thresholds:
        tp, tn, fp, fn = errors(id, t)
        prec[i] = tp/(tp+fp)
        rec[i] = tp/(tp+fn)
        i += 1
    
    prec = prec[np.argsort(rec)]
    rec = np.sort(rec)

    pr_auc = np.trapz(prec, rec)
    return pr_auc

def roc_auc(id, threshold):
    thresholds = np.linspace(0, threshold, num_thresholds)
    tpr = np.zeros(num_thresholds)
    fpr = np.zeros(num_thresholds)

    i = 0
    for t in thresholds:
        tp, tn, fp, fn = errors(id, t)
        fpr[i] = fp/(tn+fp)
        tpr[i] = tp/(tp+fn)
        i += 1
    
    tpr = tpr[np.argsort(fpr)]
    fpr = np.sort(fpr)

    roc_auc = np.trapz(tpr, fpr)
    return roc_auc

def fscore(id, threshold):
    tp, tn, fp, fn = errors(id, threshold)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fscore = (2 * prec * rec)/(prec + rec)
    return fscore

def kappa(id, threshold):
    tp, tn, fp, fn = errors(id, threshold)
    t = tp + tn + fp + fn
    kappa = ((tp+fp)*(tp+fn) + (tn+fn)*(tn+fp))/(t**2)
    return kappa

def get_measures(id, threshold):
    pr = pr_auc(id, threshold)
    roc = roc_auc(id, threshold)
    f = fscore(id, threshold)
    k = kappa(id, threshold)
    return pr, roc, f, k

pr, roc, f1, k = (np.zeros(len(test_species)) for i in range(4)) 
i = 0
for id in test_species:
    pr[i], roc[i], f1[i], k[i] = get_measures(id, 0.005) 
    i += 1

print('ROCAUC all species mean: ' + str(np.mean(roc)))
print('PRAUC all species mean: ' + str(np.mean(pr)))
print('F-score all species mean: ' + str(np.mean(f1)))
print('Kappa all species mean: ' + str(np.mean(k)))
print('\n')

with open('data_lr_balanced.txt', 'w') as f:
    f.write('ROCAUC all species mean: ' + str(np.mean(roc))+'\n')
    f.write('PRAUC all species mean: ' + str(np.mean(pr))+'\n')
    f.write('F-score all species mean: ' + str(np.mean(f1))+'\n')
    f.write('Kappa all species mean: ' + str(np.mean(k))+'\n')
    f.write('\n')

np.save('lr_balanced_roc', roc)
np.save('lr_balanced_pr', pr)
np.save('lr_balanced_fscore', f1)
np.save('lr_balanced_kappa', k)


# top 5 ids for different categories
sparsest = [4345, 44570, 42961, 32861, 2071]
densest = [38992, 29976, 8076, 145310, 4569]
largest = [4208, 12716, 145300, 4636, 4146]
smallest = [35990, 64387, 73903, 6364, 27696]
datasets = [sparsest, densest, largest, smallest]
names = ['sparsest', 'densest', 'largest', 'smallest']

j = 0
for data in datasets:
    inds = np.zeros(5)
    for i in range(5):
      inds[i] = np.where(lr.classes_ == data[i])[0][0]
    inds = inds.astype(int)

    top_5_pr = pr[inds]
    top_5_roc = roc[inds] 
    top_5_f = f1[inds] 
    top_5_k = k[inds] 

    print(names[j])
    print('ROC: ' + np.array2string(top_5_roc))
    print('ROC mean: ' + str(np.mean(top_5_roc)))
    print('PR: ' + np.array2string(top_5_pr))
    print('PR mean: ' + str(np.mean(top_5_pr)))
    print('F-score: ' + np.array2string(top_5_f))
    print('F-score mean: ' + str(np.mean(top_5_f)))
    print('Kappa: ' + np.array2string(top_5_k))
    print('Kappa mean: ' + str(np.mean(top_5_k)))
    print('\n')

    with open('data_lr_balanced.txt', 'a') as f:
        f.write(str(names[j])+'\n')
        f.write('ROC: ' + np.array2string(top_5_roc)+'\n')
        f.write('ROC mean: ' + str(np.mean(top_5_roc))+'\n')
        f.write('PR: ' + np.array2string(top_5_pr)+'\n')
        f.write('PR mean: ' + str(np.mean(top_5_pr))+'\n')
        f.write('F-score: ' + np.array2string(top_5_f)+'\n')
        f.write('F-score mean: ' + str(np.mean(top_5_f))+'\n')
        f.write('Kappa: ' + np.array2string(top_5_k)+'\n')
        f.write('Kappa mean: ' + str(np.mean(top_5_k))+'\n')
        f.write('\n')

    j += 1