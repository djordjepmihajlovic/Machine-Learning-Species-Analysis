# copy pasted from explore_species_data.py; use this file to edit and test so explore_species_data.py is clean

import numpy as np
import matplotlib.pyplot as plt
from models import FFNNet
import seaborn as sn


data = np.load('species_train.npz')
train_locs = data['train_locs']
train_ids = data['train_ids']
species = data['taxon_ids']      
species_names = dict(zip(data['taxon_ids'], data['taxon_names'])) 

# loading test data 
data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']   
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))    

# below is code structure for implementing model defined in models.py

# X = torch.rand((28, 28))

# output = net(X.view(-1, 28*28)) #-1?

# # loss and optimization

# optimizer = optim.Adam(net.parameters(), lr = 0.001) #learning rate = size of steps to take (try find 'lowest minima')

# EPOCHS = 3

# for epoch in range(EPOCHS):
#     for data in trainset:
#         # data is a batch of features and labels
#         X, y = data
#         net.zero_grad()  # note: pytorch!
#         output = net(X.view(-1, 28*28))
#         loss = F.nll_loss(output, y) # note: one-hot vector [0, 0, 1, 0] use mean sq
#         loss.backward() # backpropagation ***
#         optimizer.step() # adjust weights

#     print(loss)

# correct = 0
# total = 0

# with torch.no_grad():
#     for data in testset:
#         X, y = data
#         output = net(X.view(-1, 784))
#         for idx, i in enumerate(output):
#             if torch.argmax(i) == y[idx]:
#                 correct += 1
#             total +=1
# print("Accuracy: ", round(correct/total, 3))

# plt.imshow(X[1].view(28,28))
# plt.show()

# print(torch.argmax(net(X[1].view(-1,784))[0]))
