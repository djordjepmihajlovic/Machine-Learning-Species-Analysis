"""
Implementation of the Decision tree model
"""

#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import pickle
from sklearn.metrics import roc_curve, auc

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
train_ids_v3 = np.array(train_ids_v2)

data_test = np.load('../../data/species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds'])) 
with open('reverse_dict.pkl', 'rb') as file:
    reverse_test_pos_inds = pickle.load(file)
##### The next 30 lines take care of the reverse dictionary with added 0s for empty locations ######
"""
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

"""        

############################

#1st attempt to balance dataset, I am going to reduce the number of datapoints for each species to 500
#This is not the best method, should use a sample weighing method but unsure how to do this right now.
#Finding indices to remove and then removing them from train ids and train locs
#Find ids with more than 500 locations, find the difference between n(the number of locs) and 500 d = n - 500
#Get a sample of d indices and remove them from train ids and train locs. This is the idea.

species_count = np.bincount(train_ids) # Returns an array with 1368520 entries (which is the biggest id), each entry is the number of
# locations for the id/index 
mean = 544


sp_list_a = [] #id list of species with more than 500 location
sp_list_b = [] #id list for species below 500 locations

i = 0
for n in species_count:
    if n > mean: #around species_counts.mean(): 
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
    sp_choice = np.random.choice(sp_indices, mean, replace = False) #ALSO-500 HERE
    wanted_indices.append(sp_choice)

for sp_indices in train_inds_pos_b:
    wanted_indices.append(sp_indices)

flat_wanted_indices = [item for sublist in wanted_indices for item in sublist]

new_train_locs = train_locs[flat_wanted_indices] ##What I wanted, new train locs and train ids with a max of 500 per specie
new_train_ids = train_ids_v3[flat_wanted_indices]

test_ids = [] #Uses the new reverse dictionary to create set ids to each of the test locations
for index in range(len(test_locs)):
    test_id = reverse_test_pos_inds.get(index)
    test_ids.append(test_id)


max_depth_values = [5, 10, 15, 20, 30, 40]
error_values = []
accuracy_values = []

for max_depth in max_depth_values:
    tree_classifier = tree.DecisionTreeClassifier(max_depth = max_depth)
    tree_classifier.fit(new_train_locs, new_train_ids)
    predictions_p = tree_classifier.predict_proba(test_locs)
    """
    id = 12716
    id_inx = np.where(species == id)
    """
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    threshold = 0.0001
    for id in species:
        id_inx = np.where(species == id)
        for i in range(len(test_locs)):
            if id in test_ids[i] and predictions_p[i][id_inx[0]] > threshold:
                tp += 1
            elif id in test_ids[i] and predictions_p[i][id_inx[0]] < threshold:
                fn += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > threshold:
                fp += 1
            elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < threshold:
                tn += 1
    total = tp+fn+fp+tn
    accuracy = (tp+tn)/total
    error = (fp+fn)/total
    print('for max depth = ', max_depth) #If I do this for all species total I would get the best result i Think, could I plot it? For appendix.
    print('accuracy = ', accuracy)
    print('error =', error)
    accuracy_values.append(accuracy)
    error_values.append(error)


plt.plot(max_depth_values, accuracy_values, label='accuracy')
plt.plot(max_depth_values, error_values, label= 'errors')
plt.show()


"""
######Decision Tree model#######
tree_classifier = tree.DecisionTreeClassifier()#min_samples_leaf= 10)#, class_weight='balanced') #SHOULD LOOP THROUGH DIFFERENT LEAF NUMBERS TO CHECK BEST RESULTS!

######Fitting######
tree_classifier.fit(new_train_locs, new_train_ids)

######Predictions######
predictions = tree_classifier.predict(test_locs)

predictions_p = tree_classifier.predict_proba(test_locs)

id = 12716
id_inx = np.where(species == id)
tp = 0
tn = 0
fn = 0
fp = 0
threshold = 0.0001
for i in range(len(test_locs)):
    if id in test_ids[i] and predictions_p[i][id_inx[0]] > threshold:
        tp += 1
    elif id in test_ids[i] and predictions_p[i][id_inx[0]] < threshold:
        fn += 1
    elif id not in test_ids[i] and predictions_p[i][id_inx[0]] > threshold:
        fp += 1
    elif id not in test_ids[i] and predictions_p[i][id_inx[0]] < threshold:
        tn += 1
        
print('True positive Turdus Merulus w/ probs:', tp)
print('True negative Turdus Merulus w/ probs:', tn)
print('False positive Turdus Merulus w/ probs:', fp)
print('False negative Turdus Merulus w/ probs:', fn)
"""
"""
tp = 0
tn = 0
fn = 0
fp = 0
for i in range(len(test_locs)):
    if id in test_ids[i] and species[predictions[i]] == id:
        tp += 1
    elif id in test_ids[i] and species[predictions[i]] != id:
        fn += 1
    elif id not in test_ids[i] and species[predictions[i]] == id:
        fp += 1
    elif id not in test_ids[i] and species[predictions[i]] != id:
        tn += 1
        
print('True positive Turdus Merulus:', tp)
print('True negative Turdus Merulus:', tn)
print('False positive Turdus Merulus:', fp)
print('False negative Turdus Merulus:', fn)
"""
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

#print(tree_classifier.predict_proba([test_locs[1]])[0])
"""
"""
Attempt at predicting different number of species per location depending on the number of species 
in the testing location. Unsuccesful because decision predicts one species per location basically.
Could try to modify depth / leaf size to change this? But accuracy might decrease by doing so.


probs = tree_classifier.predict_proba(test_locs)
top_n_species = np.argsort(-probs, axis=1)[:, :3]
#print(top_n_species.shape)


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


"""
#Accuracy for a specie: 12716 Turdus Merulus

id = 12716
tp = 0
tn = 0
fn = 0
fp = 0
for i in range(len(test_locs)):
    if id in test_ids[i] and predictions[i] == id:
        tp += 1
    elif id in test_ids[i] and predictions[i] != id:
        fn += 1
    elif id not in test_ids[i] and predictions[i] == id:
        fp += 1
    elif id not in test_ids[i] and predictions[i] != id:
        tn += 1
        
print('True positive:', tp)
print('True negative:', tn)
print('False positive:', fp)
print('False negative:', fn)

tp = 0
tn = 0
fn = 0
fp = 0

for id in species:
    for i in range(len(test_locs)):
        if id in test_ids[i] and predictions[i] == id:
            tp += 1
        elif id in test_ids[i] and predictions[i] != id:
            fn += 1
        elif id not in test_ids[i] and predictions[i] == id:
            fp += 1
        elif id not in test_ids[i] and predictions[i] != id:
            tn += 1

print('Total True positive:', tp)
print('Total True negative:', tn)
print('Total False positive:', fp)
print('Total False negative:', fn)

""" 

""" 
Code for nn distribution of turdus merulus

x = np.linspace(-180, 180, 100)
y = np.linspace(-90, 90, 100)
heatmap = np.zeros((len(y), len(x)))
sp_iden =  12716 # turdus merulus
sp_idx = list(labels).index(sp_iden)

for idx, i in enumerate(x):
    for idy, j in enumerate(y):
        X = torch.tensor([j, i]).type(torch.float) # note ordering of j and i (kept confusing me yet again)
        output = net(X.view(-1, 2))
        sp_choice = output[0][sp_idx].item() # choose species of evaluation
        if sp_choice < 0.01: # if percentage prediction is < 1% of species being there then == 0 
            sp_choice = 0
        heatmap[idy, idx] = sp_choice

X, Y = np.meshgrid(x, y)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
ax = world.plot(figsize=(10, 6))
ax.set_title(str(species_names[sp_iden]))
cs = ax.contourf(X, Y, heatmap, levels = np.linspace(10**(-10), np.max(heatmap), 10), alpha = 0.5)
ax.clabel(cs, inline = True)
plt.show()
""" 

"""
Drawing predicted distribution of Turdus Merulas

sp = 12716
test_inds_pos_TM = np.where(predictions == sp)[0]

geometry = [Point(xy) for xy in zip(test_locs[test_inds_pos_TM, 1], test_locs[test_inds_pos_TM, 0])] # gets list of (lat,lon) pairs
gdf = GeoDataFrame(geometry=geometry) # creates geopandas dataframe of these pairs

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) # world map included with geopandas, could download other maps
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='k', markersize=5)
plt.title(str(sp) + ' - ' + species_names[sp])
plt.show()
"""


