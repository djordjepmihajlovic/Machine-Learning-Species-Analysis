import numpy as np 
import random

# load train data
data = np.load('species/species_train.npz')
ids = data['train_ids']
classes = np.unique(ids)
coords = np.array(list(zip(data['train_locs'][:,0], data['train_locs'][:,1]))) 
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))

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
    mu = np.array([np.sum(coords[0]), np.sum(coords[1])])/N

    # mle for cov matrix
    sig = np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(N):
        sig += np.outer((x[i]-mu), (x[i]-mu))
    sig = sig/N

    # prior modelled by 2d gaussian
    p = (1/(2* np.pi * np.linalg.det(sig))) * np.exp(-0.5 * np.dot((x_test-mu), np.matmul(np.linalg.inv(sig), (x_test-mu))))
    return p   

# computes evidence term (sum of priors)
def prior_sum(lat, lon):
    evidence = 0
    for c in classes:
        evidence += prior_prob(c, lat, lon)
    return evidence

# computes posterior probability p(y=id|x)
def probability(id, lat, lon, evidence):
    likelihood = len(ids[ids == id])/len(ids)
    prior = prior_prob(id, lat, lon)
    return (prior * likelihood)/evidence


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

la = 55.92
lo = -3.17
ev = prior_sum(la, lo)

prediction = predict(la, lo, ev)

print('Most likely species at (' + str(la) + ',' + str(lo) + ') is ' + str(species_names[prediction[0][0]]) +
      ' with probability ' + str(prediction[1][0]))
print('Top 3 Species:')
print(species_names[prediction[0][0]])
print(species_names[prediction[0][1]])
print(species_names[prediction[0][2]])