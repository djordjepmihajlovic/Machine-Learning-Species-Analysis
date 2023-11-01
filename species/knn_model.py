import numpy as np 
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# necessary to get rid of annoying scipy warning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# load training data    
data = np.load('species/species_train.npz')
#data = np.load('species_train.npz')
train_locs = data['train_locs']
train_ids = data['train_ids']
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))   

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(train_locs, train_ids, train_size = 0.9, test_size=0.1)

# k nearest neighbours classifier
knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train, y_train)

# classification scores
print('Classification Accuracy: ' + str(knn.score(X_test, y_test)))
print('F1 score micro: ' + str(f1_score(y_test, knn.predict(X_test), average = 'micro')))
#print('F1 score macro: ' + str(f1_score(y_test, knn.predict(X_test), average = 'macro')))
#print('F1 score None: ' + str(f1_score(y_test, knn.predict(X_test), average = None)))
# average = None gives the per class F1 score, may be useful
# average = 'macro' takes the unweighted mean of all the F1 scores
# average = 'micro' seems to be the same as knn.score()

# coords of edinburgh city center
la = 55.953332
lo = -3.189101
print(species_names[knn.predict([[la,lo]])[0]])