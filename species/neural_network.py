# PyTorch and Numpy modules used to build network + datastructure
import numpy as np
from nn_models import *
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import csv

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
train_locs = data_train['train_locs']  # X --> features of training data as tensor for PyTorch
train_ids = data_train['train_ids']  # y --> labels of training data as tensor for PyTorch
species_names = dict(zip(data_train['taxon_ids'], data_train['taxon_names']))  # latin names of species 

tensor_train_f = torch.Tensor(train_locs) # transform to torch tensor

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


species_weights = torch.tensor(len(train_locs)/(species_counts*500)).type(torch.double)

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

tensor_test_f = torch.Tensor(test_locs) # transform to torch tensor
tensor_test_l = torch.Tensor(test_labels).type(torch.float) # note: this is one vector label per coord
test_set = TensorDataset(tensor_test_f,tensor_test_l) 

test_loader = DataLoader(test_set, batch_size=100, shuffle=True)
# # # 

net = FFNNet(input_size = 2, train_size = 64, output_size = (len(labels)))  # pulls in defined FFNN from models.py

optimizer = optim.Adam(net.parameters(), lr = 0.0001) # learning rate = size of steps to take, alter as see fit (0.001 is good)
EPOCHS = 15 # defined no. of epochs, can change probably don't need too many (15 is good)

for epoch in range(EPOCHS):
    for data in train_loader:
        # data is a batch of features and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 2))  # pass through neural network
        # produces a vector, with idea being weight per guess for each label i.e. (0, 1, 0, 1) <- guessing that label is second and fourth in potential list
        inter_loss = F.binary_cross_entropy(output, y) # BCE most ideal for a multilabel classification problem 
        loss = torch.mean(species_weights*inter_loss)
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

print(f"Top species to be observed at {location.address}: {found_sp}, with relative liklihood {ch}")

if p == "analyze":

    # 43567
    # 13851 
    # 35990 gallotia stehlini
    # 4535 anous stolidus
    # 12716 turdus merula
    sp_idx = []
    sp_iden = [[35990, 64387, 73903, 6364, 27696], [4208, 12716, 145300, 4636, 4146], [38992, 29976, 8076, 145310, 4569], [4345, 44570, 42961, 32861, 2071]]
    sp_iden = [35990, 64387, 73903, 6364, 27696]
    probs = ['Smallest distribution', 'Largest distribution', 'Densest population', 'Sparsest population']
    for num, i in enumerate(sp_iden):
        sp_idx.append(list(labels).index(i))

    # accuracy for a specific species 
    true_p = np.zeros((1, 20))
    true_n = np.zeros((1, 20))
    false_p = np.zeros((1, 20))
    false_n = np.zeros((1, 20))
    total = np.zeros((1, 20))

    for idx, i in enumerate(sp_idx): # iterate through 
    # for counter, list_data in enumerate(sp_idx):
        # for species_no, sp_idx in enumerate(list_data):
        for data in test_loader:
            X, y = data # note ordering of j and i (kept confusing me yet again)
            output = net(X.view(-1, 2)) 
            for el in range(0, len(output)):
                # for j in range(0, len(output[0])):
                sp_choice = output[el][i].item() # choose species of evaluation
                value_ = y[el][i]

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

        print(f"species analysis {idx} done.")

    true_p_rate = true_p/(true_p + false_n)
    false_p_rate = false_p/(true_n + false_p)

    testing = true_p+false_p


    for num, i in enumerate(testing[0]):
        if i == 0:
            testing[0][num] = 0.0001

    precision = true_p/(testing)
    recall = true_p/(true_p + false_n)

    testing_2 = precision + recall

    for num, i in enumerate(testing_2[0]):
        if i == 0:
            testing_2[0][num] = 0.0001

    Po = (true_n+true_p) / (true_p+true_n+false_p+false_n)
    Pe = ((true_p+false_n)*(true_p+false_p) + (false_p+true_n)*(false_n*true_n))/(true_p+true_n+false_p+false_n)**2

    F_measure = (2*precision*recall)/(precision+recall)
    F_2_measure = (5*precision*recall)/(4*testing_2)
    cohens_kappa = (Po-Pe)/(1-Pe)

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

    AUCROC = np.abs(np.trapz(y=true_p_rate[0].tolist(), x=false_p_rate[0].tolist()))
    AUCPR = np.abs(np.trapz(y=precision[0].tolist(), x=recall[0].tolist()))

    print(f"AUCROC = {AUCROC}")
    print(f"AUCPR = {AUCPR}")
    print(f"F-score = {np.mean(F_measure)}")
    print(f"Cohens Kappa = {np.mean(cohens_kappa)}")
    print(f"F-2-score = {np.mean(F_2_measure)}")

elif p == "climate":

        bio2 = plt.imread('wc2/wc2.1_10m_bio_2.tif')
        bio15 = plt.imread('wc2/wc2.1_10m_bio_15.tif')
        x_len = len(bio2)
        y_len = len(bio2[0])
        x = []
        y = []
        heatmap_15 = np.zeros((x_len, y_len))
        heatmap_2 = np.zeros((x_len, y_len))

        for j in range(0, 2160):  # conversion is 1/6...
            for i in range(0, 1080):
                heatmap_15[i][j] = bio15[i][j][0]
                heatmap_2[i][j] = bio2[i][j][0]

        k = -50000

        idx = np.argpartition(heatmap_15.ravel(), k)
        p = tuple(np.array(np.unravel_index(idx, heatmap_15.shape))[:, range(min(k, 0), max(k, 0))])
        x_rank = [(i/6)-180 for i in p[1].tolist()]
        y_rank = [(-i/6)+90 for i in p[0].tolist()]

        locations = list(zip(y_rank, x_rank))

        analysis = np.zeros((1, 500))

        for i in locations:
            dat = torch.tensor(i)
            output = net(dat.view(-1, 2))
            analysis += output.detach().numpy()

        most_affected = np.argsort(analysis) # top most affected

        most_affected_names = [labels[element] for element in most_affected]

        print(most_affected_names)


else:
    sp_iden = 12832
    sp_idx = list(labels).index(sp_iden)
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
    cs = ax.contourf(X, Y, heatmap, levels = np.linspace(10**(-10), np.max(heatmap), 10), alpha = 0.5, cmap = 'plasma')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show() 

# currently, model is good at placing localized species but bad at placing species with two distributions