# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:32:51 2019

@author: Conma
"""


"""
***********************************************************************
    *  File:  trainMLP.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-30
    *  Desc:  Trains autoencoder, saves weights and returns model.
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

def trainSVM(dataloaders_dict, feature_size, all_parameters):

    parameters = all_parameters["svm_parameters"]
    
    print(f'Training SVM with feature size {feature_size}...')
    
    
    #################### Pass data through encoder ###################
    model = Autoencoder(feature_size)
    load_path = parameters["model_load_path"] + str(feature_size) + '.pth'
    model.load_state_dict(torch.load(load_path))
    model.eval()
    
    
    num_samples = len(dataloaders_dict["train"].dataset)
    data_train = np.empty((feature_size,num_samples))
    
    idx = 0
    
    ## Loop through full training dataset
    for inputs, labels in dataloaders_dict["train"].dataset:
        
        ## Get latent representation of data point
        data_latent = model.encoder(inputs.flatten(start_dim=1))
#        data_latent = torch.squeeze(data_latent)
    
        ## Add to data matrix
        data_train[:,idx] = data_latent.detach().numpy()
        idx = idx + 1
        
        
    num_samples = len(dataloaders_dict["train"].dataset)
    data_train = np.empty((feature_size,num_samples))
    
    idx = 0
    
    ## Loop through full dataset
    for inputs, labels in dataloaders_dict["train"].dataset:
        
        ## Get latent representation of data point
        data_latent = model.encoder(inputs.flatten(start_dim=1))
#        data_latent = torch.squeeze(data_latent)
    
        ## Add to data matrix
        data_train[:,idx] = data_latent.detach().numpy()
        idx = idx + 1
        
               

    print('Here')
    ######################### Learning Curve ##############################

    # plot the learning curve
    plt.figure()
    plt.plot(np.arange(0,len(learningCurve),1),learningCurve, c='blue')
    plt.plot(np.arange(0,len(valLearningCurve)*parameters["val_update"],parameters["val_update"]),valLearningCurve, c='orange')
    plt.title("Learing Curve", fontsize=18)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    
    save_path = parameters["image_save_path"] + '_learning_curve.png'
    plt.savefig(save_path)
    plt.close()


    ######################## Save Weights and Plot Images #################
    
    ## Save state dictionary of best model
    torch.save(best_model["modelParameters"], parameters["model_save_path"])
    
    ## Save a few validation images
    count = 0
    for inputs, labels in dataloaders_dict["val"]:
        if (count < 10):
            save_path = parameters["image_save_path"] + '_img_' + str(count) + '.png'
            y_pred = model(inputs.flatten(start_dim=1))
            loss_valid = loss_valid + criterion(y_pred, inputs.flatten(start_dim=1))
                        
            plt.figure()
            plt.imshow(y_pred[0,:].reshape((28,28)).detach().numpy())
            plt.savefig(save_path)
            plt.close()
            
            count = count + 1

    return