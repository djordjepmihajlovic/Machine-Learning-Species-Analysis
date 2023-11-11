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
import seaborn as sns

# at the heart of it, this is a multi label classification problem

# set up data
data_train = np.load('species_train.npz', mmap_mode="r")
train_locs = data_train['train_locs']  # X --> features of training data as tensor for PyTorch
train_ids = data_train['train_ids']  # y --> labels of training data as tensor for PyTorch
species_names = dict(zip(data_train['taxon_ids'], data_train['taxon_names']))  # latin names of species 

tensor_train_f = torch.Tensor(train_locs) # transform to torch tensor

# preprocessing of data s.t. train_ids isn't random numbers rather list (0,1,2,3,4 -> no. unique ID's)
print(len(np.unique(data_train['train_locs'])))
labels = np.unique(data_train['train_ids'])  
labels_vec = np.arange(0, len(labels))
labels_dict = dict(zip(labels, labels_vec)) # label + corresponding one-hot vector index

# one-hot encoding for train

train_labels = [[0]*len(labels_vec)]*len(train_locs) # make list of labels for test data
train_labels = np.array(train_labels) # have to be soooo careful with list mutability vs array mutability!

for idx, v in enumerate(train_ids): # idx is index, v is element
    point = labels_dict[v]
    code = labels_vec.tolist().index(point)
    train_labels[idx][code] = 1

# train_labels is essentially just a relabelled version of train_ids 
    
# ** train dataset (split into train/test) **
train_labels = torch.Tensor(train_labels).type(torch.float) # note: this is one vector label per coord
train_set = TensorDataset(tensor_train_f,train_labels) 

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
# ** test set **

species = data_train['taxon_ids']      # list of species IDe. Note these do not necessarily start at 0 (or 1)

data_test = np.load('species_test.npz', allow_pickle=True) 
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))   

############

# returns ID's of species + associated test_loc index at which observed

#############

# one-hot encoding for test set
count = 0
test_labels = [[0]*len(labels_vec)]*len(test_locs) # make list of labels for test data

test_labels = np.array(test_labels) # have to be soooo careful with list mutability vs array mutability!

for species_id, index in test_pos_inds.items(): # species ids in dict
    point = labels_dict[species_id] # index at which species ID corresponds

    for idx, i in enumerate(index):
        test_labels[i][point] = 1

tensor_test_f = torch.Tensor(test_locs) # transform to torch tensor
tensor_test_l = torch.Tensor(test_labels).type(torch.float) # note: this is one vector label per coord
test_set = TensorDataset(tensor_test_f,tensor_test_l) 

test_loader = DataLoader(test_set, batch_size=100, shuffle=True)
# # # 

net = FFNNet(input_size = 2, train_size = 64, output_size = (len(labels)))  # pulls in defined FFNN from models.py

optimizer = optim.Adam(net.parameters(), lr = 0.00001) # learning rate = size of steps to take, alter as see fit (0.001 is good)
EPOCHS = 3 # defined no. of epochs, can change probably don't need too many (15 is good)

for epoch in range(EPOCHS):
    for data in train_loader:
        # data is a batch of features and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 2)) # pass through neural network
        # produces a vector, with idea being weight per guess for each label i.e. (0, 1, 0, 1) <- guessing that label is second and fourth in potential list
        loss = F.mse_loss(output, y) # BCE most ideal for a multilabel classification problem 
        loss.backward() # backpropagation 
        optimizer.step() # adjust weights

        print(loss)

# ** model accuracy **

# still need a way to evaluate the accuracy reliably
# future idea is to maybe plot distribution? -> given a species where will i find it
# + given an area, what species are available

correct = 0
total = 0

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
true_p = np.zeros(20)
true_n = np.zeros(20)
false_p = np.zeros(20)
false_n = np.zeros(20)
total = np.zeros(20)

# for idx, i in enumerate(labels): # iterate through all labels

for data in test_loader:
    X, y = data # note ordering of j and i (kept confusing me yet again)
    output = net(X.view(-1, 2))
    for i in range(0, len(output)):
        # for j in range(0, len(output[0])):
        j = sp_idx
        sp_choice = output[i][j].item() # choose species of evaluation
        value_ = y[i][j]

        for idx, specificity in enumerate(np.linspace(0.05, 1, 20)):

            if sp_choice >=specificity and value_ == 1: # if percentage prediction is < 25% of species being there then == 0 
                true_p[idx] += 1

            elif sp_choice < specificity and value_ == 0:
                true_n[idx] += 1

            elif sp_choice >= specificity and value_ == 0:
                false_p[idx] += 1

            elif sp_choice < specificity and value_ == 1:
                false_n[idx] += 1

            total[idx] += 1

print(f"True positive: {true_p}")
print(f"True negative: {true_n}")
print(f"False positive: {false_p}")
print(f"False negative: {false_n}")

plt.plot(false_p.tolist(), true_p.tolist())
plt.xlabel('False Positive predictions')
plt.ylabel('True Positive Predictions')

plt.show()

x =np.linspace(-180, 180, 100)
y = np.linspace(-90, 90, 100)
heatmap = np.zeros((len(y), len(x)))

for idx, i in enumerate(x):
    for idy, j in enumerate(y):
        X = torch.tensor([j, i]).type(torch.float) # note ordering of j and i (kept confusing me yet again)
        output = net(X.view(-1, 2))
        sp_choice = output[0][sp_idx].item() # choose species of evaluation

        if sp_choice < 0.025:
            heatmap[idy, idx] = 0

        else:
            heatmap[idy, idx] = sp_choice

X, Y = np.meshgrid(x, y)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
ax = world.plot(figsize=(10, 6))
ax.set_title(str(sp_iden) + ' - ' + str(species_names[sp_iden]))
cs = ax.contourf(X, Y, heatmap, levels = np.linspace(10**(-10), np.max(heatmap), 10), alpha = 0.5, cmap = 'plasma')
ax.clabel(cs, inline = True)
plt.show() 

# currently, model is good at placing localized species but bad at placing species with two distributions