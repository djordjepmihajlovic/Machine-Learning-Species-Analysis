import numpy as np 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True)
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.savefig('cm_turdus.png')
    plt.show()

# load train data
data = np.load('../../data/species_train.npz')
ids = data['train_ids']
classes = np.unique(ids)
coords = np.array(list(zip(data['train_locs'][:,0], data['train_locs'][:,1]))) 
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))

# test data
data_test = np.load('../../data/species_test.npz', allow_pickle=True) 
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))

# load mean vector and covariance matrix data
means = np.load('../../data/gauss_data/means.npy', allow_pickle=True).item()
covs = np.load('../../data/gauss_data/covs.npy', allow_pickle=True).item()

# load per class likelihood
likelihood = np.load('../../data/gauss_data/likelihood.npy', allow_pickle=True).item()

def prior_prob(id, lat, lon):
    mu = means[id]
    sig = covs[id]
    p = (1/(2* np.pi * np.linalg.det(sig))) * np.exp(-0.5 * np.dot(([lat,lon]-mu), np.matmul(np.linalg.inv(sig), ([lat,lon]-mu))))
    return p

# computes evidence term (sum of priors)
def evidence(lat, lon):
    ev = 0
    for c in classes:
        ev += prior_prob(c, lat, lon) * likelihood[c]
    return ev

# computes posterior probability p(y=id|x)
def probability(id, lat, lon, ev):
    l = likelihood(id)
    p = prior_prob(id, lat, lon)
    return (p * l)/ev

threshold = 0.5
id_test = 12716
pos_inds = test_pos_inds[id_test]
true = np.zeros(len(test_locs))
true[pos_inds] = 1 

pred = np.zeros(len(test_locs))
i = 0
l = likelihood[id_test]

for loc in test_locs:
    ev = evidence(loc[0],loc[1])
    prior = prior_prob(id_test, loc[0], loc[1])
    p = (prior * l)/ev
    if p > threshold:
        pred[i] == 1
    else: 
        pred[i] == 0
    i += 1

conf_matrix = confusion_matrix(true, pred)
plot_confusion_matrix(conf_matrix) 