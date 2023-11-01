"""
File to check the accuracy of decision tree using train/test split instead of using testing data.
Will allow for comparison of accuracy.
"""
from sklearn import tree
import numpy as np
import random
from sklearn.model_selection import train_test_split

#Load data
data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(train_locs, train_ids, train_size = 0.9, test_size=0.1)

tree_classifier = tree.DecisionTreeClassifier(min_samples_leaf= 2)

tree_classifier.fit(X_train, y_train)

#predictions = tree_classifier.predict(X_test)

print('Decision tree classification Accuracy: ' + str(tree_classifier.score(X_test, y_test)))

"""
I actually get a worse accuracy using the score method and the data split. However, 
also true that locations in test data have multiple species and so more chance of being 
correct and that is not taken into account in my accuracy calculation, just takes it as correct
if most likely species is in the location. Maybe I could use that information... If I predict 
a specie correctly in a location with only one specie that has a value of 1/1, however, if I 
precict a specie correctly and that specie is one of five in that location, then maybe that should 
have a value of 1/5? This is also not fully fair because I am (at the moment) only predicting one
specie, if I were to predict exactly the amount of species in that location I might get a better 
accuracy estimate. If there are 5 species, predict 5 and check how many of those coincide (2/5 maybe?).
Is this possible computationally? Or useful?
"""


