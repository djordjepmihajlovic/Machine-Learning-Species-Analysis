import numpy as np
from numpy import genfromtxt
from sklearn.linear_model import LogisticRegression
import pickle

# train data
data_train = np.load('../../data/species_train.npz', mmap_mode="r")
train_ids = data_train['train_ids']

species_features_train = genfromtxt('../../data/species_train_8_features.csv', delimiter=',') # new train data
list_remove = [] # making a list of indexes to remove

for idx, i in enumerate(species_features_train):
    if i[2] == i[3] == i[4] == i[5] == i[6] == i[7] == 0:
        list_remove.append(idx)

# removing ocean data from train data
species_features_train = np.array([j for i, j in enumerate(species_features_train) if i not in list_remove])
species_labels_train = np.array([j for i, j in enumerate(train_ids) if i not in list_remove])

# test data
data_test = np.load('../../data/species_test.npz', allow_pickle=True)
test_ids = data_test['taxon_ids']
species_features_test = np.genfromtxt('../../data/species_test_8_features.csv', delimiter=',') # new X 
test_pos_inds = dict(zip(test_ids, data_test['test_pos_inds']))
test_species = np.unique(test_ids)
num_locs = len(species_features_test)

# logistic regression classifier

lr = LogisticRegression(class_weight='balanced')
lr.fit(species_features_train, species_labels_train)
with open('lr_model_8f.pkl','wb') as f:
    pickle.dump(lr,f)
"""
with open('lr_model_balanced.pkl', 'rb') as f:
    lr = pickle.load(f)
"""
# get probability values for each id and each test loc
probs = lr.predict_proba(species_features_test)

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

def fscore(id, threshold, beta):
    tp, tn, fp, fn = errors(id, threshold)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fscore = (1+beta**2)*((prec * rec)/((beta**2 * prec) + rec))
    return fscore

def kappa(id, threshold):
    tp, tn, fp, fn = errors(id, threshold)
    t = tp + tn + fp + fn
    kappa = ((tp+fp)*(tp+fn) + (tn+fn)*(tn+fp))/(t**2)
    return kappa

def get_measures(id, threshold):
    pr = pr_auc(id, threshold)
    roc = roc_auc(id, threshold)
    f1 = fscore(id, threshold, 1)
    f2 = fscore(id, threshold, 2)
    k = kappa(id, threshold)
    return pr, roc, f1, f2, k
"""
pr, roc, f1, k = (np.zeros(len(test_species)) for i in range(4)) 
i = 0
for id in test_species:
    pr[i], roc[i], f1[i], k[i] = get_measures(id, 0.05) 
    i += 1

print('ROCAUC all species mean: ' + str(np.mean(roc)))
print('PRAUC all species mean: ' + str(np.mean(pr)))
print('F-score all species mean: ' + str(np.mean(f1)))
print('Kappa all species mean: ' + str(np.mean(k)))
print('\n')

with open('data_lr_8f.txt', 'w') as f:
    f.write('ROCAUC all species mean: ' + str(np.mean(roc))+'\n')
    f.write('PRAUC all species mean: ' + str(np.mean(pr))+'\n')
    f.write('F-score all species mean: ' + str(np.mean(f1))+'\n')
    f.write('Kappa all species mean: ' + str(np.mean(k))+'\n')
    f.write('\n')

np.save('lr_roc_balanced', roc)
np.save('lr_pr_balanced', pr)
np.save('lr_fscore_balanced', f1)
np.save('lr_kappa_balanced', k)
"""
# top 5 ids for different categories
sparsest = [4345, 44570, 42961, 32861, 2071]
densest = [38992, 29976, 8076, 145310, 4569]
largest = [4208, 12716, 145300, 4636, 4146]
smallest = [35990, 64387, 73903, 6364, 27696]
datasets = [sparsest, densest, largest, smallest]
names = ['sparsest', 'densest', 'largest', 'smallest']

j = 0
for data in datasets:
    if names[j] != 'largest':
        top_5_pr, top_5_roc, top_5_f1, top_5_f2, top_5_k = (np.zeros(5) for i in range(5))
        for i in range(5):
            top_5_pr[i],top_5_roc[i],top_5_f1[i], top_5_f2[i], top_5_k[i] = get_measures(data[i], 0.05)
    else:
        top_5_pr, top_5_roc, top_5_f1, top_5_f2, top_5_k = (np.zeros(3) for i in range(5))
        for i in range(3):
            top_5_pr[i],top_5_roc[i],top_5_f1[i], top_5_f2[i], top_5_k[i] = get_measures(data[i], 0.05)
    

    print(names[j])
    print('ROC: ' + np.array2string(top_5_roc))
    print('ROC mean: ' + str(np.mean(top_5_roc)))
    print('PR: ' + np.array2string(top_5_pr))
    print('PR mean: ' + str(np.mean(top_5_pr)))
    print('F1-score: ' + np.array2string(top_5_f1))
    print('F1-score mean: ' + str(np.mean(top_5_f1)))
    print('F2-score: ' + np.array2string(top_5_f2))
    print('F2-score mean: ' + str(np.mean(top_5_f2)))
    print('Kappa: ' + np.array2string(top_5_k))
    print('Kappa mean: ' + str(np.mean(top_5_k)))
    print('\n')

    with open('data_lr_8f.txt', 'a') as f:
        f.write(str(names[j])+'\n')
        f.write('ROC: ' + np.array2string(top_5_roc)+'\n')
        f.write('ROC mean: ' + str(np.mean(top_5_roc))+'\n')
        f.write('PR: ' + np.array2string(top_5_pr)+'\n')
        f.write('PR mean: ' + str(np.mean(top_5_pr))+'\n')
        f.write('F1-score: ' + np.array2string(top_5_f1)+'\n')
        f.write('F1-score mean: ' + str(np.mean(top_5_f1))+'\n')
        f.write('F2-score: ' + np.array2string(top_5_f2)+'\n')
        f.write('F2-score mean: ' + str(np.mean(top_5_f2))+'\n')
        f.write('Kappa: ' + np.array2string(top_5_k)+'\n')
        f.write('Kappa mean: ' + str(np.mean(top_5_k))+'\n')
        f.write('\n')

    j += 1