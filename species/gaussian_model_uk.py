import numpy as np 
import random

# load train data
data = np.load('species/species_train.npz')
ids = data['train_ids']
coords = np.array(list(zip(data['train_locs'][:,0], data['train_locs'][:,1]))) 
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))

# uk bounding box
ind1 = np.array(np.where(coords[:,1] >= -9.0))
ind2 = np.array(np.where(coords[:,1] <= 2.01))
ind3 = np.array(np.where(coords[:,0] >= 49.75))
ind4 = np.array(np.where(coords[:,0] <= 61.01))
uk_ind = np.intersect1d(ind1, np.intersect1d(ind2, np.intersect1d(ind3, ind4)))

uk_coords = coords[uk_ind]
uk_ids = ids[uk_ind]
uk_classes = np.unique(uk_ids)

# computes prior probability p(x|y=id)
def prior_prob(id, lat, lon):
    ind = np.where(uk_ids == id)
    # take 50 random samples if more than 50 examples exist
    if len(ind) > 50:
        ind = random.sample(ind, 50)

    x = uk_coords[ind]
    x_test = np.array([lat,lon])
    N = len(x)

    # if this species isn't present return a zero prior probability
    if N == 0:
        return 0
    
    # mle for mean vector
    mu = np.array([np.sum(uk_coords[0]), np.sum(uk_coords[1])])/N

    # mle for cov matrix
    sig = np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(N):
        sig += np.outer((x[i]-mu), (x[i]-mu))
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
    for c in uk_classes:
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
    for c in uk_classes:
        prob = probability(c, lat, lon, ev)
        p = np.append(p, prob)
    ind = np.argsort(p)
    ranking = np.flip(uk_classes[ind])
    probabilities = np.flip(p[ind])
    return ranking, probabilities

# coords
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
