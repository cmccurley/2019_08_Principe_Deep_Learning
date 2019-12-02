# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:48:29 2019

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  AE010.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-29
    *  Desc:  Provides class definition for autoencoder with size
    *         100 bottleneck layer.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.model_selection as ms
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from fashion_mnist_master.utils import mnist_reader
from setParams import setParams
from torch.utils import data
from my_data import my_data
from torch.autograd import Variable
from dim_reduction import pca_dim_reduction, umap_dim_reduction, plot_dist_distributions
import seaborn as sns



######################################################################
##################### Function Definitions ###########################
######################################################################

###################### Define Neural Net Class #######################
class Feedforward(torch.nn.Module):

        def __init__(self, input_size, output_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.relu = torch.nn.ReLU()
            self.fc1 = torch.nn.Linear(self.input_size, 256)
            self.fc2 = torch.nn.Linear(256,128)
            self.fc3 = torch.nn.Linear(128,100)
            self.fc4 = torch.nn.Linear(100, self.output_size)

        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            hidden = self.fc2(relu)
            relu = self.relu(hidden)
            hidden = self.fc3(relu)
            relu = self.relu(hidden)
            output = self.fc4(relu)
            return output

###################### Define Autoencoder Class #######################
class Autoencoder100(torch.nn.Module):

    def __init__(self):
        super(Autoencoder100, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 400),
            nn.ReLU(True), nn.Linear(400, 150), nn.ReLU(True), nn.Linear(150, 100))
        self.decoder = nn.Sequential(
            nn.Linear(100, 150),
            nn.ReLU(True),
            nn.Linear(150, 400),
            nn.ReLU(True),
            nn.Linear(400, 500),
            nn.ReLU(True), nn.Linear(500, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder400(torch.nn.Module):

    def __init__(self):
        super(Autoencoder400, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 400))
        self.decoder = nn.Sequential(
            nn.Linear(400, 500),
            nn.ReLU(True), nn.Linear(500, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

