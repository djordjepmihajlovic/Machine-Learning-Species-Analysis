# PyTorch and Numpy modules used to build network + datastructure
import numpy as np
from nn_models import FFNNet
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

net = FFNNet(input_size = 2, train_size = 100, output_size = (len(labels)) )  # pulls in defined FFNN from models.py

# # # loss and optimization --> type of optimizer (Adam)

optimizer = optim.Adam(net.parameters(), lr = 0.001) # learning rate = size of steps to take, alter as see fit (0.001 is good)
EPOCHS = 20  # defined no. of epochs, can change probably don't need too many (10 is good)

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
        
    print(epoch)

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
            observations = torch.topk(i, 1)
            for j in observations[1]:
                if j == torch.argmax(y[idx]):
                    correct += 1
            total +=1

print("Accuracy: ", round(correct/total, 3))

# print(torch.argmax(y[0]), torch.topk(output[0], 30))

X = torch.tensor([-1.286389, 36.817223]) # Nairobi, Kenya
X = torch.tensor([-75.10052548639784, 123.35002912811925]) # Concordia, Antarctica
X = torch.tensor([0.5162773075444781, 25.20450202335836]) # Kisangani, Congo
X = torch.tensor([22.625950924673553, 97.30142582402296]) # Hsipaw, Myanmar
X = torch.tensor([-3.0028115414379086, -59.995268536688606]) # Manauas, Brazil
X = torch.tensor([-21.91487895822523, 48.11852675456639]) # Manaraka, Madagascar
X = torch.tensor([27.975769515926874, -15.582669871478057]) # Gran Canaria

output = net(X.view(-1, 2))
observations = torch.topk(output, 10)
ch = observations[0].tolist()
sp = observations[1].tolist()
sp = sp[0]

geolocator = Nominatim(user_agent="youremail@provider")
location = geolocator.reverse("27.975769515926874, -15.582669871478057")

found_sp = []
for i,v in enumerate(sp):
    found_sp.append(species_names[labels[v]])

print(f"Top 10, most likely species to be observed at {location.address}: {found_sp}, with relative liklihood {ch}")



x = np.linspace(-180, 180, 100)
y = np.linspace(-90, 90, 100)
heatmap = np.zeros((len(y), len(x)))
sp_iden = 12716 
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


# currently, model is good at placing localized species but bad at placing species with two distributions