# PyTorch and Numpy modules used to build network + datastructure
import numpy as np
from nn_models import *
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# geopy used to decode lat/lon to named address
from geopy.geocoders import Nominatim

# sklearn.metrics confusion matrix used for evaluation of conf matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# visualization
import matplotlib.pyplot as plt
import geopandas as gpd

# at the heart of it, this is a multi label classification problem

# set up data
data_train = np.load('species_train.npz', mmap_mode="r")
train_locs = data_train['train_locs']  # X --> features of training data as tensor for PyTorch
train_ids = data_train['train_ids']  # y --> labels of training data as tensor for PyTorch
species_names = dict(zip(data_train['taxon_ids'], data_train['taxon_names']))  # latin names of species 

tensor_train_f = torch.Tensor(train_locs) # transform to torch tensor

# preprocessing of data s.t. train_ids isn't random numbers rather list (0,1,2,3,4 -> no. unique ID's)
labels = np.unique(data_train['train_ids'])  
labels_vec = np.arange(0, len(labels))
labels_dict = dict(zip(labels, labels_vec)) # label + corresponding one-hot vector index

# there is are features (locations) passed in multiple times with different labels, need to find ALL corresponding labels 
# with associated feature for nn to work fully
train_ids_alt = []
poss_labels = [[] for i in range(len(train_locs))]
f_prev = [0, 0]

# one-hot encoding for train
train_labels = [0]*len(train_ids)
for idx, v in enumerate(train_ids): # idx is index, v is element
    base = [0]*len(labels_vec)
    point = labels_dict[v]
    code = labels_vec.tolist().index(point)
    base[code] = 1 # (creates a k-hot vector)
    train_labels[idx] = base

# train_labels is essentially just a relabelled version of train_ids 
    
# ** train dataset (split into train/test) **
train_labels = torch.Tensor(train_labels).type(torch.float) # note: this is one vector label per coord
train_set = TensorDataset(tensor_train_f,train_labels) 

train_size = int(0.9*len(train_set))
val_size = len(train_set) - train_size

train, val = torch.utils.data.random_split(train_set, [train_size, val_size])

train_loader = DataLoader(train, batch_size=100, shuffle=True)
val_loader = DataLoader(val, batch_size=100, shuffle=True)

# ** test set **

species = data_train['taxon_ids']      # list of species IDe. Note these do not necessarily start at 0 (or 1)

data_test = np.load('species_test.npz', allow_pickle=True) 
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))   

############

# returns ID's of species + associated test_loc index at which observed

#############


# this is wrong

# # one-hot encoding for test set
# test_labels = [[0]*len(labels_vec)]*len(test_locs) # make list of labels for test data
# for species_id, index in test_pos_inds.items(): # species ids in dict
#     base = [0]*len(labels_vec) # one-hot vector (hopefully same size)
#     point = labels_dict[species_id]
#     code = labels_vec.tolist().index(point)
#     base[code] = 1

#     for i in index:
#         test_labels[i] = base


# tensor_test_f = torch.Tensor(test_locs) # transform to torch tensor
# tensor_test_l = torch.Tensor(test_labels).type(torch.float) # note: this is one vector label per coord
# test_set = TensorDataset(tensor_test_f,tensor_test_l) 

# test_loader = DataLoader(test_set, batch_size=100, shuffle=True)

# # # 

net = FFNNet(input_size = 2, train_size = 32, output_size = (len(labels)))  # pulls in defined FFNN from models.py

optimizer = optim.Adam(net.parameters(), lr = 0.001) # learning rate = size of steps to take, alter as see fit (0.001 is good)
EPOCHS = 15 # defined no. of epochs, can change probably don't need too many (15 is good)

for epoch in range(EPOCHS):
    for data in train_loader:
        # data is a batch of features and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 2)) # pass through neural network
        # produces a vector, with idea being weight per guess for each label i.e. (0, 1, 0, 1) <- guessing that label is second and fourth in potential list
        loss = F.binary_cross_entropy(output, y) # CE most ideal for a multilabel classification problem 
        loss.backward() # backpropagation 
        optimizer.step() # adjust weights

        print(loss)

# ** model accuracy **

# still need a way to evaluate the accuracy reliably
# future idea is to maybe plot distribution? -> given a species where will i find it
# + given an area, what species are available

correct = 0
total = 0

# want chosen species 12176 

# sp_iden = 12176
# sp_idx = list(labels).index(sp_iden)

# with torch.no_grad():
#     for data in test_loader:
#         X, y = data
#         output = net(X.view(-1, 2))
#         for idx, i in enumerate(output):
#             positive_idx = torch.argmax(y[idx])
#             if torch.max(y[idx]) == 0:
#                 positive_idx = []
#             observations = torch.topk(i, 20)
#             if observations[0][0] <= 0.01:
#                 correct +=1
#             else:
#                 for j in observations[1]:
#                     if j in positive_idx:
#                         correct += 1
#             total +=1
            

# print("Accuracy: ", round(correct/total, 3)*100, "%")

# print(torch.argmax(y[0]), torch.topk(output[0], 30))

X = torch.tensor([-1.286389, 36.817223]) # Nairobi, Kenya

output = net(X.view(-1, 2))
observations = torch.topk(output, 10)
ch = observations[0].tolist()
sp = observations[1].tolist()
sp = sp[0]

geolocator = Nominatim(user_agent="youremail@provider")
location = geolocator.reverse("-1.286389, 36.817223")

found_sp = []
for i,v in enumerate(sp):
    found_sp.append(species_names[labels[v]])

test = 0
for i in output[0].tolist():
    test += i

print(test)
print(output[0].tolist())

print(f"Top species to be observed at {location.address}: {found_sp}, with relative liklihood {ch}")


sp_iden =  12716 # turdus merulus
sp_idx = list(labels).index(sp_iden)

# accuracy for a specific species 
true_p = 0
true_n = 0
false_p = 0
false_n = 0

# for idx, i in enumerate(labels): # iterate through all labels

for data in train_loader:
    X, y = data # note ordering of j and i (kept confusing me yet again)
    output = net(X.view(-1, 2))
    for i in range(0, len(output)):
        sp_choice = output[i][sp_idx].item() # choose species of evaluation
        value_ = y[i][sp_idx]
        if sp_choice > 0.025 and value_ == 1: # if percentage prediction is < 25% of species being there then == 0 
            true_p += 1

        elif sp_choice < 0.025 and value_ == 0:
            true_n += 1

        elif sp_choice > 0.025 and value_ == 0:
            false_p += 1

        elif sp_choice < 0.025 and value_ == 1:
            false_n += 1

        total += 1

print(idx)

print(f"True positive: {true_p}")
print(f"True negative: {true_n}")
print(f"False positive: {false_p}")
print(f"False negative: {false_n}")

x =np.linspace(-180, 180, 100)
y = np.linspace(-90, 90, 100)
heatmap = np.zeros((len(y), len(x)))

for idx, i in enumerate(x):
    for idy, j in enumerate(y):
        X = torch.tensor([j, i]).type(torch.float) # note ordering of j and i (kept confusing me yet again)
        output = net(X.view(-1, 2))
        sp_choice = output[0][sp_idx].item() # choose species of evaluation
        heatmap[idy, idx] = sp_choice

X, Y = np.meshgrid(x, y)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
ax = world.plot(figsize=(10, 6))
ax.set_title(str(species_names[sp_iden]))
cs = ax.contourf(X, Y, heatmap, levels = np.linspace(10**(-10), np.max(heatmap), 10), alpha = 0.5)
ax.clabel(cs, inline = True)
plt.show() 

# currently, model is good at placing localized species but bad at placing species with two distributions