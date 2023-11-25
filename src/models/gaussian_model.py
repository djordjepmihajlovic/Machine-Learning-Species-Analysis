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

# returns list of species present at (lat,lon) sorted by probability
def predict(lat, lon, ev):
    p = np.array([])
    for c in classes:
        prob = probability(c, lat, lon, ev)
        p = np.append(p, prob)
    ind = np.argsort(p)
    ranking = np.flip(classes[ind])
    probabilities = np.flip(p[ind])
    return ranking, probabilities


# coords of edinburgh city center
la = 55.953332
lo = -3.189101
ev = evidence(la, lo)
prediction = predict(la, lo, ev)

print('Most likely species at (' + str(la) + ',' + str(lo) + ') is ' + str(species_names[prediction[0][0]]) +
      ' with probability ' + str(prediction[1][0]))
print('Top 3 Species:')
print(species_names[prediction[0][0]])
print(species_names[prediction[0][1]])
print(species_names[prediction[0][2]])