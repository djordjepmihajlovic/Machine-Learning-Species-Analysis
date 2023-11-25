import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# necessary to get rid of annoying scipy warning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#Load data
data = np.load('../../data/species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

# test data
data_test = np.load('../../data/species_test.npz', allow_pickle=True) 
test_locs = data_test['test_locs']
test_ids = data_test['taxon_ids']
test_species = np.unique(test_ids)
num_locs = len(test_locs)
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))

# k nearest neighbours classifier, optimal k found by examining F1 scores
knn = KNeighborsClassifier(n_neighbors = 75)
knn.fit(train_locs, train_ids)
# get probability values for each id and each test loc
probs = knn.predict_proba(test_locs)
# weight the probability vector at each location
weights = np.load('../../data/weights.npy')
probs *=  weights
num_thresholds = 20

def errors(id, threshold):
    # array with 1s if species is present at that index in test_locs, 0 otherwise
    pos_inds = test_pos_inds[id]
    true = np.zeros(num_locs, dtype = int)
    true[pos_inds] = 1 
    id_index = np.where(knn.classes_ == id)[0][0]

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

pr, roc, f1, f2, k = (np.zeros(len(test_species)) for i in range(5)) 
i = 0
for id in test_species:
    pr[i], roc[i], f1[i], f2[i], k[i] = get_measures(id, 0.025) 
    i += 1

print('ROCAUC all species mean: ' + str(np.mean(roc)))
print('PRAUC all species mean: ' + str(np.mean(pr)))
print('F2-score all species mean: ' + str(np.mean(f1)))
print('F2-score all species mean: ' + str(np.mean(f2)))
print('Kappa all species mean: ' + str(np.mean(k)))
print('\n')


with open('data_knn_balanced.txt', 'w') as f:
    f.write('ROCAUC all species mean: ' + str(np.mean(roc))+'\n')
    f.write('PRAUC all species mean: ' + str(np.mean(pr))+'\n')
    f.write('F1-score all species mean: ' + str(np.mean(f1))+'\n')
    f.write('F2-score all species mean: ' + str(np.mean(f2))+'\n')
    f.write('Kappa all species mean: ' + str(np.mean(k))+'\n')
    f.write('\n')

np.save('knn_roc_balanced', roc)
np.save('knn_pr_balanced', pr)
np.save('knn_f1_balanced', f1)
np.save('knn_f2_balanced', f2)
np.save('knn_kappa_balanced', k)

pr = np.load('knn_pr_balanced.npy')
roc = np.load('knn_roc_balanced.npy')
f1 = np.load('knn_f1_balanced.npy')
f2 = np.load('knn_f2_balanced.npy')
k = np.load('knn_kappa_balanced.npy')

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
      inds[i] = np.where(knn.classes_ == data[i])[0][0]
    inds = inds.astype(int)

    top_5_pr = pr[inds]
    top_5_roc = roc[inds] 
    top_5_f1 = f1[inds] 
    top_5_f2 = f2[inds]
    top_5_k = k[inds] 

    print(names[j])
    print('ROC: ' + str(np.mean(top_5_roc)))
    print('PR: ' + str(np.mean(top_5_pr)))
    print('F1-score: ' + str(np.mean(top_5_f1)))
    print('F2-score: ' + str(np.mean(top_5_f2)))
    print('Kappa: ' + str(np.mean(top_5_k)))
    print('\n')

    with open('data_knn_balanced.txt', 'a') as f:
        f.write(str(names[j])+'\n')
        f.write('ROC: ' + str(np.mean(top_5_roc))+'\n')
        f.write('PR: ' + str(np.mean(top_5_pr))+'\n')
        f.write('F1-score: ' + str(np.mean(top_5_f1))+'\n')
        f.write('F2-score: ' + str(np.mean(top_5_f2))+'\n')
        f.write('Kappa: ' + str(np.mean(top_5_k))+'\n')
        f.write('\n')

    j += 1