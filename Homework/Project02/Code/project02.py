# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:39:11 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  project02.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-29
    *  Desc:  Provides coded solutions to project 02 of EEL681,
    *         Deep Learning, taught by Dr. Jose Principe, Fall 2019.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################

## Default packages
import os
import copy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors

from fashion_mnist_master.utils import mnist_reader

import sklearn.model_selection as ms
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

## PyTorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import FashionMNIST
from torchvision import transforms

## Custom packages
from setParams import setParams
from trainAE import trainAE
from trainCNN import trainCNN


######################################################################
##################### Function Definitions ###########################
######################################################################


######################################################################
############################## Main ##################################
######################################################################
if __name__== "__main__":

    print('Running Main...')

    ####################### Set Parameters ###########################
    parameters = setParams()

    ####################### Import data ##############################
    print('Loading data...')
    cwd = os.getcwd()

    ## Create dataset
    dataset_train = FashionMNIST(parameters["dataPath"], train=True, transform=transforms.ToTensor(), download=False)
    dataset_test = FashionMNIST(parameters["dataPath"], train=False, transform=transforms.ToTensor(), download=False)
    
    ## Split indices for training and validation 
    indices_train = np.arange(len(dataset_train))
    y_train = dataset_train.targets.numpy()
    test_indices = np.arange(len(dataset_test))
    y_test = dataset_test.targets.numpy()
        
    ## Split training, validation and test sets
    y_train,y_val,train_indices,val_indices = train_test_split(y_train,indices_train,test_size = parameters["validationSize"],random_state=parameters["random_state"])
 
    
    ## Create data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    mmist_datasets = {'train': dataset_train, 'val': dataset_train, 'test': dataset_test}
    
    
    ## Create training and validation dataloaders
    dataloaders_dict = {'train': torch.utils.data.DataLoader(mmist_datasets['train'], batch_size=parameters["batch_size"],
                                               sampler=train_sampler, shuffle=False,num_workers=0),
                        'val': torch.utils.data.DataLoader(mmist_datasets['val'],batch_size=parameters["batch_size"],
                                               sampler=valid_sampler, shuffle=False,num_workers=0),
                        'test': torch.utils.data.DataLoader(mmist_datasets['test'], batch_size=parameters["batch_size"],
                                               sampler=test_sampler, shuffle=False,num_workers=0) }


    ######################################################################
    ########################## Train Autoencoder #########################
    ######################################################################
    ## Train autoencoder network with chosen bottlenck size
    
    if parameters["train_ae"]:
        
        for val in [10, 25, 50, 75, 100]:
            parameters["ae_parameters"]["ae_latent_size"] = val
            parameters["ae_parameters"]["model_save_path"] = os.getcwd() + '\\ae_model_parameters\\ae_latent_' + str(parameters["ae_parameters"]["ae_latent_size"]) + '.pth'
            parameters["ae_parameters"]["image_save_path"] = os.getcwd() + '\\ae_reconstructed_images\\ae_latent_' + str(parameters["ae_parameters"]["ae_latent_size"])
            trainAE(dataloaders_dict, parameters)


    ######################################################################
    ############################# Train CNN ##############################
    ######################################################################
    ## Train baseline CNN Network 
    if parameters["train_cnn"]:
        trainCNN(dataloaders_dict, parameters)
    
    ######################################################################
    ############################ Train SVM ###############################
    ######################################################################
    
    ## Load desired model
    
    ## Train SVM
    
    ## Get performance on test set
    
    ## Confusion matrices
    
    

##    ####################### Confusion Matrix #########################
#
##    np.save('model_auto_400.npy', best_model)
#
#    # revert model back to best performing
#    model.load_state_dict(best_model["modelParameters"])
#    model.eval()
#
#    # predict state labels
#    y_test_pred = model(X_test)
#    values, y_test_pred_index = y_test_pred.max(1)
#
#    # compute the loss
#    testLoss = criterion(y_test_pred, y_test)
#
#    testLoss = testLoss.detach().numpy()
#    testLoss = np.round(testLoss,2)
#
#    # plot the confusion matrix
#    plot_confusion_matrix(y_test.detach().numpy(), y_test_pred_index.detach().numpy(), parameters["classes"], testLoss, normalize=False, title='Normalized Confusion Matrix for Fashion-MNIST')


    print('================ DONE ================')

