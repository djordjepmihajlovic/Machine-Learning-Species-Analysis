"""
File to check the accuracy of decision tree using train/test split instead of using testing data.
Will allow for comparison of accuracy.
"""
from sklearn import tree
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


#Load data
data = np.load('../../data/species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

range_list = range(len(species)) #Range from 0-499
spec_dict = dict(zip(species, range_list)) #Dictionary matches species id with index in species
train_ids_v2 = [] #List of train ids, now they go from 0 to 499
for indx in train_ids:
    x = spec_dict.get(indx)
    train_ids_v2.append(x)

"""
species_count = np.bincount(train_ids) 
sp_list_a = [] #
sp_list_b = [] 

i = 0
for n in species_count:
    if n > 500: ####################### number here!
        sp_list_a.append(i) 
    elif n != 0:
        sp_list_b.append(i)
    i = i + 1

train_inds_pos_a = [] 
train_inds_pos_b= [] 
wanted_indices = [] 

for species_id in sp_list_a:
    train_inds_pos_a.append(np.where(train_ids == species_id)[0])

for species_id in sp_list_b:
    train_inds_pos_b.append(np.where(train_ids == species_id)[0])

for sp_indices in train_inds_pos_a:
    sp_choice = np.random.choice(sp_indices, 500, replace = False) #################### number here!
    wanted_indices.append(sp_choice)

for sp_indices in train_inds_pos_b:
    wanted_indices.append(sp_indices)

flat_wanted_indices = [item for sublist in wanted_indices for item in sublist]
new_train_locs = train_locs[flat_wanted_indices]
new_train_ids = train_ids[flat_wanted_indices]
"""
# split into train and test data
#X_train, X_test, y_train, y_test = train_test_split(new_train_locs, new_train_ids, train_size = 0.9, test_size=0.1)
X_train, X_test, y_train, y_test = train_test_split(train_locs, train_ids2, train_size = 0.9, test_size=0.1)
"""
depths = [10, 30, 50, 60, 80]

depthscore = []

for depth in depths:

    tree_classifier = tree.DecisionTreeClassifier(min_samples_leaf= 1, max_depth= depth)#, random_state= 1000)#, class_weight= 'balanced')

    tree_classifier.fit(X_train, y_train)

    depthscore.append(tree_classifier.score(X_test, y_test))
plt.plot(depths, depthscore)
plt.xlabel('Depth value')
plt.ylabel('Accuracy')
plt.show()
"""

tcl2 = tree.DecisionTreeClassifier(min_samples_leaf= 5)

tcl2.fit(X_train, y_train)

predictions_p = tcl2.predict_proba(X_test)
predictions = tcl2.predict(X_test)

location_index = 1

print('proabilities in random location =', predictions_p[location_index])
print('length of prediction array', len(predictions_p[location_index]))
prob_indeces = []
for index in range(len(predictions_p[location_index])):
    if predictions_p[location_index][index] > 0:
        prob_indeces.append(index)
    else:
        continue
print('indeces with probability above 0 in loc =', prob_indeces)
print('predicition in same random location =', predictions[location_index])
print('index of prediction', np.where(range_list == predictions[location_index])[0])



#predictions = tree_classifier.predict(X_test)

#print('Decision tree classification Accuracy: ' + str(tree_classifier.score(X_test, y_test)))

#cm = confusion_matrix(y_test, tree_classifier.predict(X_test), normalize = 'true')
#print(cm)



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


