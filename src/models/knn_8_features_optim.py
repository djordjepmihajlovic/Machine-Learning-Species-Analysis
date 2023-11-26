import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# necessary to get rid of annoying scipy warning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
   
# train data
data_train = np.load('../../data/species_train.npz')
train_ids = data_train['train_ids']

species_features_train = np.genfromtxt('../../data/species_train_8_features.csv', delimiter=',') # new train data
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
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))
test_species = np.unique(test_ids)
num_locs = len(species_features_test)

# adjusted weights
weights = np.load('../../data/weights.npy')

def get_f1_score(id, threshold, probs):
    # array with 1s if species is present at that index in test_locs, 0 otherwise
    pos_inds = test_pos_inds[id]
    true = np.zeros(num_locs, dtype = int)
    true[pos_inds] = 1 
    id_index = np.where(knn.classes_ == id)[0][0]

    # predicted probabilities
    pred = np.zeros(num_locs)
    probs_bin = (probs >= threshold).astype(int)
    pred = probs_bin[:,id_index]

    f1 = f1_score(true, pred)
    return f1


k_vals = np.array([5,10,25,50,75,100,150,250,500])
mean_vals = np.zeros(len(k_vals))
ids = [4345, 44570, 42961, 32861, 2071, 38992, 29976, 8076, 145310, 4569,
    4208, 12716, 145300, 35990, 64387, 73903, 6364, 27696]
for k in k_vals:
    # k nearest neighbours classifier
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(species_features_train, species_labels_train)
    # get probability values for each id and each test loc
    probs = knn.predict_proba(species_features_test) * weights
    f1 = np.zeros(len(test_species))
    for i in range(len(ids)):
        f1[i] = get_f1_score(test_species[i], 1/k, probs)
    mean_vals[np.where(k_vals == k)[0][0]] = np.mean(f1)

print(np.max(mean_vals))
print(k_vals[np.where(mean_vals == np.max(mean_vals))[0][0]])
plt.xlabel(r'$k$', fontsize = 10)
plt.ylabel('Mean F1 score (8 features)', fontsize = 10)
plt.plot(k_vals, mean_vals, marker='o', color = 'k')
plt.savefig('knn_f1_scores_8f.png')
plt.show()