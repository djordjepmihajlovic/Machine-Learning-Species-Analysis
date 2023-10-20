import numpy as np
from nn_models import FFNNet
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# set up data
data_train = np.load('species_train.npz', mmap_mode="r")
train_locs = data_train['train_locs']  # X --> features of training data as tensor for PyTorch
train_ids = data_train['train_ids']  # y --> labels of training data as tensor for PyTorch

tensor_train_f = torch.Tensor(train_locs) # transform to torch tensor


# preprocessing of data s.t. train_ids isn't random numbers rather list (0,1,2,3,4 -> no. unique ID's)
labels = np.unique(data_train['train_ids'])  
labels_vec = np.arange(0, len(labels))
labels_dict = dict(zip(labels, labels_vec))

train_labels = [0]*len(train_ids)
for idx, v in enumerate(train_ids): # idx is index, v is element
    train_labels[idx] = labels_dict[v]
# train_labels is essentially just a relabelled version of train_ids 
    
# ** train dataset **
train_labels = torch.Tensor(train_labels).type(torch.LongTensor)
dataset = TensorDataset(tensor_train_f,train_labels) 
trainset = DataLoader(dataset, batch_size=(len(labels)), shuffle = True) 

# ** test set **
# data_test = np.load('species_test.npz', allow_pickle=True) 
# test_locs = data_test['test_locs']
# test_ids = data_test['test_pos_inds']  # actual possible labels for test data

# tensor_test_f = torch.Tensor(test_locs) # transform to torch tensor
# tensor_test_l = torch.Tensor(test_ids[0])
# testset = TensorDataset(tensor_test_f,tensor_test_l) 

net = FFNNet(input_size = 2, train_size = 2, output_size = (len(labels)) )  # pulls in defined FFNN from models.py

# # # # loss and optimization --> type of optimizer (Adam)

optimizer = optim.Adam(net.parameters(), lr = 0.001) # learning rate = size of steps to take, alter as see fit
EPOCHS = 2  # defined no. of epochs, can change probably don't need too many

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of features and labels
        X, y = data
        net.zero_grad()  
        output = net(X.view(-1, 2)) # pass through neural network
        # produces a vector, with idea being weight per guess for each label i.e. (0, 0, 0, 1) <- guessing that label is fourth in potential list
        loss = F.cross_entropy(output, y) # look at loss functions
        loss.backward() # backpropagation
        optimizer.step() # adjust weights

        print(loss)

# ** model accuracy **

# correct = 0
# total = 0

# with torch.no_grad():
#     for data in trainset:
#         X, y = data
#         output = net(X.view(-1, 2))
#         for idx, i in enumerate(output):
#             if torch.argmax(i) == y[idx]:
#                 correct += 1
#             total +=1
# print("Accuracy: ", round(correct/total, 3))

# print(torch.argmax(net(X[1].view(-1,2))[0]))



