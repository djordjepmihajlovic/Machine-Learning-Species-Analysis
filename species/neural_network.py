import numpy as np
from nn_models import FFNNet
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# at the heart of it, this is a multi label classification problem

# set up data
data_train = np.load('species_train.npz', mmap_mode="r")
train_locs = data_train['train_locs']  # X --> features of training data as tensor for PyTorch
train_ids = data_train['train_ids']  # y --> labels of training data as tensor for PyTorch

tensor_train_f = torch.Tensor(train_locs) # transform to torch tensor

# preprocessing of data s.t. train_ids isn't random numbers rather list (0,1,2,3,4 -> no. unique ID's)
labels = np.unique(data_train['train_ids'])  
labels_vec = np.arange(0, len(labels))
labels_dict = dict(zip(labels, labels_vec))

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

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)

# ** test set **

species = data_train['taxon_ids']      # list of species IDe. Note these do not necessarily start at 0 (or 1)

data_test = np.load('species_test.npz', allow_pickle=True) 
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))    

reverse_test_pos_inds = {} # reversing dictionary -> species at given location. (Thanks Miguel)

for species_id, indices in test_pos_inds.items():
    for index in indices:
        reverse_test_pos_inds[index] = species_id

# print(next(iter(reverse_test_pos_inds.items())))
# returns index + ID's of species at index

# one-hot encoding for test set
test_labels = [[0]*len(labels_vec)]*len(test_locs) # make list of labels for test data
for index, species_id in reverse_test_pos_inds.items():
    base = [0]*len(labels_vec) # one-hot vector (hopefully same size)
    point = labels_dict[species_id]
    code = labels_vec.tolist().index(point)
    base[code] = 1
    test_labels[index] = base

tensor_test_f = torch.Tensor(test_locs) # transform to torch tensor
tensor_test_l = torch.Tensor(test_labels).type(torch.float) # note: this is one vector label per coord
test_set = TensorDataset(tensor_test_f,tensor_test_l) 

test_loader = DataLoader(test_set, batch_size=100, shuffle=True)

# # # 

net = FFNNet(input_size = 2, train_size = 10, output_size = (len(labels)) )  # pulls in defined FFNN from models.py

# # # loss and optimization --> type of optimizer (Adam)

optimizer = optim.Adam(net.parameters(), lr = 0.01) # learning rate = size of steps to take, alter as see fit
EPOCHS = 3  # defined no. of epochs, can change probably don't need too many

for epoch in range(EPOCHS):
    for data in train_loader:
        # data is a batch of features and labels
        X, y = data
        net.zero_grad()  
        output = net(X.view(-1, 2)) # pass through neural network
        # produces a vector, with idea being weight per guess for each label i.e. (0, 1, 0, 1) <- guessing that label is second and fourth in potential list
        loss = F.binary_cross_entropy(output, y) # BCE most ideal for a multilabel classification problem 
        loss.backward() # backpropagation 
        optimizer.step() # adjust weights

        print(loss)

# ** model accuracy **

# still need a way to evaluate the accuracy reliably
# future idea is to maybe plot distribution? -> given a species where will i find it
# + given an area, what species are available

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        X, y = data
        output = net(X.view(-1, 2))
        for idx, i in enumerate(output):
            observations = torch.topk(i, 10)
            for j in observations[1]:
                if j == torch.argmax(y[idx]):
                    correct += 1
            total +=1

print("Accuracy: ", round(correct/total, 3))

print(y[0])
print(output[0])

print(torch.argmax(y[0]), torch.topk(output[0], 30))



