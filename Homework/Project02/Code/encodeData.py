# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:05:58 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  encodeData.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-30
    *  Desc:  Encodes the data through an autoencoder and saves.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################

## Default packages
import copy
import numpy as np

## PyTorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable

## Custom packages
from AE import Autoencoder
import matplotlib.pyplot as plt
from utils import predict

######################################################################
##################### Function Definitions ###########################
######################################################################

def encodeData(dataloaders_dict, feature_size, train_indices, val_indices, y_train, y_val, y_test, all_parameters):

    parameters = all_parameters["encode_parameters"]

    print(f'Encoding data with feature size {feature_size}...')

    ######################## Define Encoder ##########################
    model = Autoencoder(feature_size)
    load_path = parameters["model_load_path"] + str(feature_size) + '.pth'
    model.load_state_dict(torch.load(load_path))
    model.eval()

    #################### Encode Training Data ########################
    print('Encoding training data')
    num_samples = len(dataloaders_dict["train"].dataset)
    data_train_and_val = np.empty((feature_size,num_samples))

    idx = 0

    ## Loop through full training dataset
    for inputs, labels in dataloaders_dict["train"].dataset:

        ## Get latent representation of data point
        data_latent = model.encoder(inputs.flatten(start_dim=1))

        ## Add to data matrix
        data_train_and_val[:,idx] = data_latent.detach().numpy()
        idx = idx + 1

    #################### Encode Validation Data #######################
    print('Splitting into training and validation')

    data_train = data_train_and_val[:,train_indices]

    data_val = data_train_and_val[:,val_indices]


    ###################### Encode Test Data ###########################
    print('Encoding test data')
    num_samples = len(dataloaders_dict["test"].dataset)
    data_test = np.empty((feature_size,num_samples))

    idx = 0

    ## Loop through full dataset
    for inputs, labels in dataloaders_dict["test"].dataset:

        ## Get latent representation of data point
        data_latent = model.encoder(inputs.flatten(start_dim=1))

        ## Add to data matrix
        data_test[:,idx] = data_latent.detach().numpy()
        idx = idx + 1


    ######################## Save Data ###############################
    print('Saving data')

    data = dict()
    data["X_train"] = data_train
    data["X_val"] = data_val
    data["X_test"] = data_test
    data["y_train"] = y_train
    data["y_val"] = y_val
    data["y_test"] = y_test

    save_path = parameters["data_save_path"] + str(feature_size) + '.npy'
    np.save(save_path, data)

    return

