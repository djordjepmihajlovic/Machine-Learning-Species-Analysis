"""
Implementation of the Decision tree model
"""

#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

#Load data
data = np.load('species_train.npz')
train_locs = data['train_locs']          
train_ids = data['train_ids']               
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 

##### The next 30 lines take care of the reverse dictionary with added 0s for empty locations ######

reverse_test_pos_inds = {} #Reversing dictionary so that you can check species at given location.

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index] = [] #Creates list at each index
        

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index].append(species_id) #Appends species id to lists created earlier


indices = range(len(test_locs))
indices_0 = [] #List for indices/locations with no species. Uses following loop to fill list.

for index in indices:
    if index not in reverse_test_pos_inds.keys():
        indices_0.append(index)
        
test_pos_inds[0] = indices_0 #Modifies test_pos_inds dictionary to include indices with no species and sets them to key "0"

reverse_test_pos_inds = {} #Reversing dictionary again so that you have the locations with no species referencing a 0.

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index] = []
        

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index].append(species_id)

############################

#1st attempt to balance dataset, I am going to reduce the number of datapoints for each species to 500
#This is not the best method, should use a sample weighing method but unsure how to do this right now.
#Finding indices to remove and then removing them from train ids and train locs
#Find ids with more than 500 locations, find the difference between n(the number of locs) and 500 d = n - 500
#Get a sample of d indices and remove them from train ids and train locs. This is the idea.

species_count = np.bincount(train_ids) # Returns an array with 1368520 entries (which is the biggest id), each entry is the number of
# locations for the id/index 



sp_list_a = [] #id list of species with more than 500 location
sp_list_b = [] #id list for species below 500 locations

i = 0
for n in species_count:
    if n >= 500: #around species_counts.mean():
        sp_list_a.append(i) # i is the id of the species/because index = id
    elif n != 0:
        sp_list_b.append(i)
    i = i + 1

train_inds_pos_a = [] #List with indices of train_ids/train_locs of the above 500 species. Created in the following loop.

for species_id in sp_list_a:
    train_inds_pos_a.append(np.where(train_ids == species_id)[0])

train_inds_pos_b= [] #List with indices of train_ids/train_locs of the below 500 species.

for species_id in sp_list_b:
    train_inds_pos_b.append(np.where(train_ids == species_id)[0])

wanted_indices = [] # Wanted indices, if more than 500 choose 500 best, if not take all datapoints


for sp_indices in train_inds_pos_a:
    sp_choice = np.random.choice(sp_indices, 500, replace = False)
    wanted_indices.append(sp_choice)

for sp_indices in train_inds_pos_b:
    wanted_indices.append(sp_indices)

flat_wanted_indices = [item for sublist in wanted_indices for item in sublist]

new_train_locs = train_locs[flat_wanted_indices] ##What I wanted, new train locs and train ids with a max of 500 per specie
new_train_ids = train_ids[flat_wanted_indices]

#Using these new locs and ids I got an improvement on the accuracy of around 2% which is not very significant but it is noteworthy
# because I have REMOVED data and it has IMPROVED accuracy... There are better methods to explore the data imbalance we should look into.

######Decision Tree model#######
tree_classifier = tree.DecisionTreeClassifier(min_samples_leaf= 5) #SHOULD LOOP THROUGH DIFFERENT LEAF NUMBERS TO CHECK BEST RESULTS!

##### Should I use a min sample leaf? Best Results so far (small sample) is using minimum of 2 per leaf, not massive change.
##### Using class_weight = "balanced" made the model a little worst actually, maybe it can be weighed properly using another method.

######Fitting######
tree_classifier.fit(new_train_locs, new_train_ids)


######Predictions######
predictions = tree_classifier.predict(test_locs)

test_ids = [] #Uses the new reverse dictionary to create set ids to each of the test locations
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    test_ids.append(test_id)

"""
IGNORE
Plotting tree?????
tree.plot_tree(tree_classifier)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(tree_classifier, max_depth = 1, filled=True)
fig.savefig("decistion_tree.png")
"""

######Accuracy######
#Score method cant be used because each test_loc might have multiple species in it. Hence the method cant compare how 
#good the decision tree is.
#print('Decision tree classification Accuracy: ' + str(tree_classifier.score(test_locs, test_ids)))
#Is there another method???

"""
Commented out because it is irrelevant, but it takes 10 random locations and checks whether or not predicted 
specie is in that location.

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
"""

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
#Results are not great, I get around 54% accuracy - Increased to around 56-57% using an "in-house" data balancing.
#Implementing a resampling or similar to get rid of the data imbalance I think might help

"""
Attempt at predicting different number of species per location depending on the number of species 
in the testing location. Unsuccesful because decision predicts one species per location basically.
Could try to modify depth / leaf size to change this? But accuracy might decrease by doing so.


probs = tree_classifier.predict_proba(test_locs)
top_n_species = np.argsort(-probs, axis=1)[:, :3]
#print(top_n_species.shape)
print(tree_classifier.predict_proba([test_locs[0]])[0])

for index in range(len(test_locs)):
    real_species_ids = reverse_test_pos_inds.get(index)
    real_names = []
    n = 0
    for id in real_species_ids:
        if id == 0:
            species_name = 'No species'
        else:
            species_name = species_names[id]
            n += 1
        real_names.append(species_name)
    
    predicted_species_ids = []
    for m in range(n):
        prediction = tree_classifier.predict_proba(test_locs[index])
        predicted_species_id = predictions[index][m]
"""
"""
Usually predicts the Turdus Viscivorus species in this location.
# coords of edinburgh city center
la = 55.953332
lo = -3.189101
print(species_names[tree_classifier.predict([[la,lo]])[0]])
"""