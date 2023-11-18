# PyTorch and Numpy modules used to build network + datastructure
import numpy as np
from nn_models import *
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import csv
from numpy import genfromtxt

# geopy used to decode lat/lon to named address
from geopy.geocoders import Nominatim

# sklearn.metrics confusion matrix used for evaluation of conf matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# visualization
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import pandas as pd

# at the heart of it, this is a multi label classification problem

p = 'analyze'

# set up data
data_train = np.load('species_train.npz', mmap_mode="r")
train_locs = data_train['train_locs']  # original X --> features of training data as tensor for PyTorch

species_features_train = genfromtxt('species_train_5_features.csv', delimiter=',') # new X 

train_ids = data_train['train_ids']  # y --> labels of training data as tensor for PyTorch

species_names = dict(zip(data_train['taxon_ids'], data_train['taxon_names']))  # latin names of species 

tensor_train_f = torch.Tensor(species_features_train) # transform to torch tensor -> now 5 features

# preprocessing of data s.t. train_ids isn't random numbers rather list (0,1,2,3,4 -> no. unique ID's)
labels = np.unique(data_train['train_ids'])  
labels_vec = np.arange(0, len(labels))
labels_dict = dict(zip(labels, labels_vec)) # label + corresponding one-hot vector index

# one-hot encoding for train

train_labels = [[0]*len(labels_vec)]*len(train_locs) # make list of labels for test data
train_labels = np.array(train_labels) # have to be soooo careful with list mutability vs array mutability!

species_counts = [0]*500
species_counts = np.array(species_counts)

for idx, v in enumerate(train_ids): # idx is index, v is element
    point = labels_dict[v]
    code = labels_vec.tolist().index(point)
    train_labels[idx][code] = 1
    species_counts[code] += 1


species_weights = torch.tensor(len(train_locs)/(species_counts*500))

weight_data = list(zip(species_weights, labels))

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

species_features_test = genfromtxt('species_test_5_features.csv', delimiter=',') # new X 

tensor_test_f = torch.Tensor(species_features_test) # transform to torch tensor
tensor_test_l = torch.Tensor(test_labels).type(torch.float) # note: this is one vector label per coord
test_set = TensorDataset(tensor_test_f,tensor_test_l) 

test_loader = DataLoader(test_set, batch_size=100, shuffle=True)
# # # 

net = FFNNet(input_size = 5, train_size = 64, output_size = (len(labels)))  # pulls in defined FFNN from models.py

optimizer = optim.Adam(net.parameters(), lr = 0.001) # learning rate = size of steps to take, alter as see fit (0.001 is good)
EPOCHS = 5 # defined no. of epochs, can change probably don't need too many (15 is good)

for epoch in range(EPOCHS):
    for data in train_loader:
        # data is a batch of features and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 5)) # pass through neural network
        # produces a vector, with idea being weight per guess for each label i.e. (0, 1, 0, 1) <- guessing that label is second and fourth in potential list
        inter_loss = F.binary_cross_entropy(output, y) # BCE most ideal for a multilabel classification problem 
        loss = torch.mean(species_weights*inter_loss)
        loss.backward() # backpropagation 
        optimizer.step() # adjust weights

        print(loss)



species_test = [35990, 64387, 73903, 6364, 27696]
true_p = np.zeros((1, 20))
true_n = np.zeros((1, 20))
false_p = np.zeros((1, 20))
false_n = np.zeros((1, 20))
total = np.zeros((1, 20))

for idx, i in enumerate(species_test): # iterate through all labels

    for data in test_loader:
        X, y = data # note ordering of j and i (kept confusing me yet again)
        output = net(X.view(-1, 5))
        for el in range(0, len(output)):
            sp_choice = output[el][idx].item() # choose species of evaluation
            value_ = y[el][idx]

            for idxs, specificity in enumerate(np.linspace(0.0, 0.025, 20)):

                if sp_choice >=specificity and value_ == 1: # if percentage prediction is < 25% of species being there then == 0 
                    true_p[0][idxs] += 1

                elif sp_choice < specificity and value_ == 0:
                    true_n[0][idxs] += 1

                elif sp_choice >= specificity and value_ == 0:
                    false_p[0][idxs] += 1

                elif sp_choice < specificity and value_ == 1:
                    false_n[0][idxs] += 1

                total[0][idxs] += 1

    print(f"species analysis {i} done.")

true_p_rate = true_p/(true_p + false_n)
false_p_rate = false_p/(true_n + false_p)

testing = true_p+false_p

print(true_p)
print(testing)

for num, i in enumerate(testing[0]):
    if i == 0:
        testing[0][num] = 0.0001

print(testing)


precision = true_p/(testing)
recall = true_p/(true_p + false_n)

conf_mat = [[true_p[0][1]/(true_p[0][1]+false_n[0][1]), true_n[0][1]/(true_n[0][1]+false_p[0][1])], [false_p[0][1]/(true_n[0][1]+false_p[0][1]), false_n[0][1]/(false_n[0][1]+true_p[0][1])]] # ideal sensitivity
conf_label = ['True', 'False']
conf_col = ['Positive', 'Negative']

df_cm = pd.DataFrame(conf_mat, index = conf_label,
                columns = conf_col)

sns.set_theme()

sns.lineplot(x = false_p_rate[0].tolist(), y = true_p_rate[0].tolist())

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax = plt.gca()
ax.set_ylim([0, 1.05])

plt.show()

sns.lineplot(x = recall[0].tolist(), y = precision[0].tolist())
plt.xlabel('Recall')
plt.ylabel('Precision')
ax = plt.gca()
ax.set_ylim([0, 1.05])

plt.show()

AUCROC = []
AUCPR = []

AUCROC.append(np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist())))
AUCPR.append(np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist())))

print(AUCROC)
print(AUCPR)