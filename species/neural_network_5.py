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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# at the heart of it, this is a multi label classification problem

p = 'plot'

# set up data
data_train = np.load('species_train.npz', mmap_mode="r")
train_locs = data_train['train_locs']  # original X --> features of training data as tensor for PyTorch

species_features_train = genfromtxt('species_train_8_features.csv', delimiter=',') # new X 

list_remove = [] # making a list of indexes to remove

for idx, i in enumerate(species_features_train):
    if i[2] == i[3] == i[4] == i[5] == i[6] == i[7] == 0:
        list_remove.append(idx)


# removing ocean data
species_features_train = np.array([j for i, j in enumerate(species_features_train) if i not in list_remove])

print(len(species_features_train))

print('done...')

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

#removing ocean data
train_labels = np.array([j for i, j in enumerate(train_labels) if i not in list_remove])

print('done...')


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

species_features_test = genfromtxt('species_test_8_features.csv', delimiter=',') # new X 

list_remove = [] # making a list of indexes to remove now for test

for idx, i in enumerate(species_features_test):
    if i[2] == i[3] == i[4] == i[5] == i[6] == i[7] == 0:
        list_remove.append(idx)

# species_features_test = np.array([j for i, j in enumerate(species_features_test) if i not in list_remove])

print('done...')

tensor_test_f = torch.Tensor(species_features_test) # transform to torch tensor

# test_labels = np.array([j for i, j in enumerate(test_labels) if i not in list_remove])

print('done...')

tensor_test_l = torch.Tensor(test_labels).type(torch.float) # note: this is one vector label per coord
test_set = TensorDataset(tensor_test_f,tensor_test_l) 

test_loader = DataLoader(test_set, batch_size=100, shuffle=True)
# # # 

net = FFNNet(input_size = 8, train_size = 256, output_size = (len(labels)))  # pulls in defined FFNN from models.py

optimizer = optim.Adam(net.parameters(), lr = 0.0001) # learning rate = size of steps to take, alter as see fit (0.001 is good)
EPOCHS = 15 # defined no. of epochs, can change probably don't need too many (15 is good)

for epoch in range(EPOCHS):
    for data in train_loader:
        # data is a batch of features and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 8)) # pass through neural network
        # produces a vector, with idea being weight per guess for each label i.e. (0, 1, 0, 1) <- guessing that label is second and fourth in potential list
        inter_loss = F.binary_cross_entropy(output, y) # BCE most ideal for a multilabel classification problem 
        loss = torch.mean(species_weights*inter_loss)
        loss.backward() # backpropagation 
        optimizer.step() # adjust weights

        print(loss)

if p == "plot":

    # generate vulnerability scores

    sp_sum = torch.zeros(500)

    data_clim_coords = np.load('scores_coords.npy') # indexes [lat][lon]
    data_clim_scores = np.load('scores.npy')

    # temperature related
    bio7 = plt.imread('wc2/wc2.1_10m_bio_7.tif') # mean temp range
    bio10 = plt.imread('wc2/wc2.1_10m_bio_10.tif') # mean temp (cold quarter)
    bio11 = plt.imread('wc2/wc2.1_10m_bio_11.tif') # mean temp (warm quarter)

    # precipitation related
    bio12 = plt.imread('wc2/wc2.1_10m_bio_12.tif') # annual precip
    bio14 = plt.imread('wc2/wc2.1_10m_bio_14.tif') # precip (driest month)
    bio15 = plt.imread('wc2/wc2.1_10m_bio_15.tif') # precip seasonality 

    x_len = len(bio12)
    y_len = len(bio12[0])
    # temp heatmaps
    heatmap_temp_range = np.zeros((x_len, y_len))
    heatmap_temp_cold = np.zeros((x_len, y_len))
    heatmap_temp_warm = np.zeros((x_len, y_len))

    # precip heatmaps
    heatmap_precip = np.zeros((x_len, y_len))
    heatmap_precip_dry = np.zeros((x_len, y_len))
    heatmap_precip_season = np.zeros((x_len, y_len))

    for j in range(0, 2160):  
        for i in range(0, 1080):
            heatmap_temp_range[i][j] = bio7[i][j][0]
            heatmap_temp_cold[i][j] = bio10[i][j][0]
            heatmap_temp_warm[i][j] = bio11[i][j][0]
            heatmap_precip[i][j] = bio12[i][j][0]
            heatmap_precip_dry[i][j] = bio14[i][j][0]
            heatmap_precip_season[i][j] = bio15[i][j][0]
            
    for idx, i in enumerate(data_clim_coords):

        latitude = i[0] # -90 -> 90
        lat_conv = -6*(latitude-90) -1
        longitude = i[1] # -180 -> 180
        long_conv = 6*(longitude+180) -1
        temp_range = heatmap_temp_range[int(lat_conv)][int(long_conv)]
        temp_cold = heatmap_temp_cold[int(lat_conv)][int(long_conv)]
        temp_warm = heatmap_temp_warm[int(lat_conv)][int(long_conv)]

        precip = heatmap_precip[int(lat_conv)][int(long_conv)]
        precip_dry = heatmap_precip_dry[int(lat_conv)][int(long_conv)]
        precip_season = heatmap_precip_season[int(lat_conv)][int(long_conv)]

        if temp_range == 0.0 and temp_cold == 0.0 and temp_warm == 0.0 and precip == 0.0 and precip_dry == 0.0 and precip_season == 0.0:

            sp_sum = sp_sum
        
        else:

            X = torch.tensor([i[0], i[1], temp_range, temp_cold, temp_warm, precip, precip_dry, precip_season]).type(torch.float) # note ordering of j and i (kept confusing me yet again)
            with torch.no_grad():
                output = net(X.view(-1, 8))
            sp_sum += data_clim_scores[idx] * output[0] # climate change score * prediction


    top_5 = torch.topk(sp_sum, 5)
    bottom_5 = torch.topk(sp_sum, 5, largest = False)
    species_most_affected = [labels[i] for i in top_5[1]]
    species_least_affected = [labels[i] for i in bottom_5[1]]


    print(f"top 5 species most affected by climate change: {species_most_affected}")
    print(f"top 5 species least affected by climate change: {species_least_affected}")

    # plot species + vulnerability

    # sp_iden = 12832 # arid species 
    sp_iden = 54549 
    sp_idx = list(labels).index(sp_iden)
    x =np.linspace(-180, 180, 100)
    y = np.linspace(-90, 90, 100)
    heatmap = np.zeros((len(y), len(x)))
    heatmap_tester = np.zeros((len(y), len(x)))

    # temperature related
    bio7 = plt.imread('wc2/wc2.1_10m_bio_7.tif') # mean temp range
    bio10 = plt.imread('wc2/wc2.1_10m_bio_10.tif') # mean temp (cold quarter)
    bio11 = plt.imread('wc2/wc2.1_10m_bio_11.tif') # mean temp (warm quarter)

    # precipitation related
    bio12 = plt.imread('wc2/wc2.1_10m_bio_12.tif') # annual precip
    bio14 = plt.imread('wc2/wc2.1_10m_bio_14.tif') # precip (driest month)
    bio15 = plt.imread('wc2/wc2.1_10m_bio_15.tif') # precip seasonality 

    x_len = len(bio12)
    y_len = len(bio12[0])
    # temp heatmaps
    heatmap_temp_range = np.zeros((x_len, y_len))
    heatmap_temp_cold = np.zeros((x_len, y_len))
    heatmap_temp_warm = np.zeros((x_len, y_len))

    # precip heatmaps
    heatmap_precip = np.zeros((x_len, y_len))
    heatmap_precip_dry = np.zeros((x_len, y_len))
    heatmap_precip_season = np.zeros((x_len, y_len))




    for j in range(0, 2160):  
        for i in range(0, 1080):
            heatmap_temp_range[i][j] = bio7[i][j][0]
            heatmap_temp_cold[i][j] = bio10[i][j][0]
            heatmap_temp_warm[i][j] = bio11[i][j][0]
            heatmap_precip[i][j] = bio12[i][j][0]
            heatmap_precip_dry[i][j] = bio14[i][j][0]
            heatmap_precip_season[i][j] = bio15[i][j][0]
            


    for idx, i in enumerate(x):
        for idy, j in enumerate(y):
                latitude = j # -90 -> 90
                lat_conv = -6*(latitude-90) -1
                longitude = i # -180 -> 180
                long_conv = 6*(longitude+180) -1
                temp_range = heatmap_temp_range[int(lat_conv)][int(long_conv)]
                temp_cold = heatmap_temp_cold[int(lat_conv)][int(long_conv)]
                temp_warm = heatmap_temp_warm[int(lat_conv)][int(long_conv)]

                precip = heatmap_precip[int(lat_conv)][int(long_conv)]
                precip_dry = heatmap_precip_dry[int(lat_conv)][int(long_conv)]
                precip_season = heatmap_precip_season[int(lat_conv)][int(long_conv)]

                if temp_range == 0.0 and temp_cold == 0.0 and temp_warm == 0.0 and precip == 0.0 and precip_dry == 0.0 and precip_season == 0.0:

                    heatmap[idy, idx] = 0
                
                else:

                    X = torch.tensor([j, i, temp_range, temp_cold, temp_warm, precip, precip_dry, precip_season]).type(torch.float) # note ordering of j and i (kept confusing me yet again)
                    with torch.no_grad():
                        output = net(X.view(-1, 8))
                    sp_choice = output[0][sp_idx].item() # choose species of evaluation

                    heatmap_tester[idy, idx] = precip_season

                    if sp_choice < 0.025:
                        heatmap[idy, idx] = 0

                    else:
                        heatmap[idy, idx] = sp_choice

    X, Y = np.meshgrid(x, y)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
    ax = world.plot(figsize=(10, 6))
    cs = ax.contourf(X, Y, heatmap, levels = np.linspace(10**(-10), np.max(heatmap), 10), alpha = 0.5, cmap = 'plasma')
    cbar  = plt.colorbar(cs)
    # cbar.set_label('Specificity (positive predicition cut-off value)')  # Add a label to your colorbar

    sp_sum = sp_sum.tolist()

    sp_sum = [i/max(sp_sum) for i in sp_sum]

    vulnerability = sp_sum[sp_idx]

    text =  '- Vulnerability -\n'
    text += fr'{vulnerability:.3f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)  # bbox features
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show() 


elif p == "analyze":
    print('here')

    sp_idx = []
    sp_iden = [4345, 44570, 42961, 32861, 2071]
    for num, i in enumerate(sp_iden):
        sp_idx.append(list(labels).index(i))

    # accuracy for a specific species 
    true_p = np.zeros((1, 20))
    true_n = np.zeros((1, 20))
    false_p = np.zeros((1, 20))
    false_n = np.zeros((1, 20))
    total = np.zeros((1, 20))
    print('here')

    for idx, i in enumerate(sp_idx): # iterate through 
    # for counter, list_data in enumerate(sp_idx):
        # for species_no, sp_idx in enumerate(list_data):
        for data in test_loader:
            X, y = data # note ordering of j and i (kept confusing me yet again)
            with torch.no_grad():
                output = net(X.view(-1, 8)) 
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

    F_measure = (2*precision*recall)/(testing_2)
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
    

    print(len(species_features_test))



