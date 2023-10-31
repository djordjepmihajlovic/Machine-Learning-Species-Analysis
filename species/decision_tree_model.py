"""
Implementation of the Decision tree model
"""

#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import random

#Load data
data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 


reverse_test_pos_inds = {} #Reversing dictionary so that you can check species at given location.

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index] = []
        

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index].append(species_id)


indices = range(len(test_locs))
indices_0 = []

for index in indices:
    if index not in reverse_test_pos_inds.keys():
        indices_0.append(index)
        
test_pos_inds[0] = indices_0

reverse_test_pos_inds = {} #Reversing dictionary again so that you have the none species as 0

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index] = []
        

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index].append(species_id)
        

######Decision Tree model#######
tree_classifier = tree.DecisionTreeClassifier()

######Fitting######
tree_classifier.fit(train_locs, train_ids)

######Predictions######
predictions = tree_classifier.predict(test_locs)

test_ids = []
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    test_ids.append(test_id)

#Plotting tree?????
#tree.plot_tree(tree_classifier)

######Accuracy######
#Accuracy method cant be used because each test_loc might have multiple species in it.. hence the method cant compare how 
#good the decision tree is.
#print('Decision tree classification Accuracy: ' + str(tree_classifier.score(test_locs, test_ids)))



#Printing 10 random locations, the species in them and the prediction.
sample_indices = random.sample(range(len(test_locs)),k=10)

i = 0
for index in sample_indices:
    i += 1
    real_species_ids = reverse_test_pos_inds.get(index)
    predicted_species_id = predictions[index]
    predicted_species_name = species_names[predicted_species_id]
    #print(i,'. predicted species name:', predicted_species_name)

    real_names = []
    #print('real species ids', real_species_ids)
    for id in real_species_ids:
        if id == 0:
            species_name = 'No species'
        else:
            species_name = species_names[id]
        real_names.append(species_name)

    #print(i, '. real species names:', real_names)

    if predicted_species_name in real_names:
        print(i, 'prediction is correct')
    else:
        print(i, 'prediction is wrong')

#Second try at accuracy:
j = 0
for index in range(len(test_locs)):
    real_species_ids = reverse_test_pos_inds.get(index)
    real_names = []
    #print('real species ids', real_species_ids)
    for id in real_species_ids:
        if id == 0:
            species_name = 'No species'
        else:
            species_name = species_names[id]
        real_names.append(species_name)

    predicted_species_id = predictions[index]
    predicted_species_name = species_names[predicted_species_id]
    if predicted_species_name in real_names:
        j += 1

#Accuracy of predicted values, checks whether predicted id is in the location for real.
#Took away locations with no species from the total as those will always be wrong as we
#havent trained the data with places with no species.
accuracy = j/(len(test_locs)-len(indices_0))*100 
# This accuracy only takes into account most likely specie in location, doesnt account 
#for the second/third... most likely and doesnt account for probabilities. It also "ignores"
#the fact that there is more than one specie for location just takes it as a "win" if the 
#predicted specie is there.

print(accuracy)
#Results are not great, I get around 54% accuracy
#Implementing a resampling or similar to get rid of the data imbalance I think might help

"""
# coords of edinburgh city center
la = 55.953332
lo = -3.189101
print(species_names[tree_classifier.predict([[la,lo]])[0]])
"""