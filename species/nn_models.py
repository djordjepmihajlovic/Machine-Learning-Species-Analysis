# implement our models (neural networks, linear regression etc.) here
# thinking we use both PyTorch & skLearn ? --> good documentation online plus most widely used in industry AFAIK

import torch
import torch.nn as nn
import torch.nn.functional as F

# basic neural network model: followed a YT tutorial to make this (https://www.youtube.com/watch?v=ixathu7U-LQ&t=1366s)

# first model idea: take in species & their locations; train model to predict distn. of a specific species
# features - locations
# labels - species

class FFNNet(nn.Module):
    def __init__(self, input_size = "data size", train_size = "hidden layer size", output_size = "label class size"):
        super().__init__() #basically runs nn.Module
        self.fc1 = nn.Linear(input_size, train_size) # self.fcx is x^th hidden layer of neural network
        self.fc2 = nn.Linear(train_size, train_size) 
        self.fc3 = nn.Linear(train_size, output_size)

    def forward(self, x): # define order + activation function (F.relu() in this case ReLu in notes) + optimizer(multiclass=softmax(usually))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)  # need to double check what this actually means 







