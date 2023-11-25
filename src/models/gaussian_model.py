import numpy as np 
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(coords, ids, train_size = 0.999, test_size=0.001)

# computes prior probability p(x|y=id)
def prior_prob(id, lat, lon):
    ind = np.where(ids == id)
    # take 50 random samples if more than 50 examples exist
    if len(ind) > 50:
        ind = random.sample(ind, 50)

    x = coords[ind]
    x_test = np.array([lat,lon])
    N = len(x)
    if N == 0:
        return 0
    
    # mle for mean vector
    mu = np.array([np.sum(x[:,0]), np.sum(x[:,1])])/N

    # mle for cov matrix
    sig = np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(N):
        sig += np.outer((x[i,:]-mu), (x[i,:]-mu))
    sig = sig/N

    # prior modelled by 2d gaussian
    p = (1/(2* np.pi * np.linalg.det(sig))) * np.exp(-0.5 * np.dot((x_test-mu), np.matmul(np.linalg.inv(sig), (x_test-mu))))
    return p   

# computes likelihood term
def likelihood(id):
    l = len(ids[ids == id])/len(ids)
    return l

# computes evidence term (sum of priors)
def evidence(lat, lon):
    ev = 0
    for c in classes:
        ev += prior_prob(c, lat, lon) * likelihood(c)
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

"""
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

# find F1 score
y_pred = np.zeros(len(y_test))
print(len(y_test))
for i in range(len(y_test)):
    test_lat = X_test[i,0]
    test_lon = X_test[i,1]
    y_pred[i] = predict(test_lat, test_lon, evidence(test_lat, test_lon))[0][0]

print('F1 score: ' + str(f1_score(y_test, y_pred, average = 'micro')))
"""