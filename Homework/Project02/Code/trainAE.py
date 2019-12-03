# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:39:29 2019

@author: cmccurley
"""


"""
***********************************************************************
    *  File:  trainAE.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-29
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

def trainAE(dataloaders_dict, all_parameters):

    parameters = all_parameters["ae_parameters"]
    
    print(f'Training Autoencoder with latent size {parameters["ae_latent_size"]}...')

    ############# run a number of trials, save best model ############
    for trial in range(parameters["numTrials"]):

        learningCurve = []
        valLearningCurve = []

        ####################### Define Network ###########################

        # instantiate model
        model = Autoencoder(parameters["ae_latent_size"])

        # define loss function
        criterion = torch.nn.MSELoss()

        # define optimizer (stochastic gradient descent)
        optimizer = torch.optim.Adamax(model.parameters(), parameters["learning_rate"])

        ##################### Train the Network ##########################

        model.train()

        ################# Train a single network #####################
        for epoch in range(parameters["numEpochs"]):
            
            for inputs, labels in dataloaders_dict["train"]:

                #set gradients to zero
                optimizer.zero_grad()
    
                # forward pass
                y_pred = model(inputs) # predict output vector
    
                # compute loss
                loss = criterion(y_pred, labels)
    
                # backward pass
                loss.backward() # computes the gradients
                optimizer.step() # updates the weights
    
    
    
                ############### Add to learning curve #########################
                ## Compute total training loss
                model.eval()
                loss_train = 0
                for inputs, labels, index, _ in dataloaders_dict["train"]:
                    y_pred = model(inputs)
                    loss_train = loss_train + criterion(y_pred, labels)
                loss_train =  loss_train/len(dataloaders_dict["train"].dataset)
                learningCurve.append(loss_train)
                
                ## Compute total validation loss
                loss_valid = 0
                for inputs, labels, index, _ in dataloaders_dict["val"]:
                    y_pred = model(inputs)
                    loss_valid = loss_valid + criterion(y_pred, labels)
                loss_valid =  loss_valid/len(dataloaders_dict["val"].dataset)
                
                valLearningCurve.append(loss_valid)
                model.train()

            ## Update epoch training status
            if not(epoch % 1):
                
                ## Compute total training loss
                loss_train = 0
                for inputs, labels, index, _ in dataloaders_dict["train"]:
                    y_pred = model(inputs)
                    loss_train = loss_train + criterion(y_pred, labels)
                loss_train =  loss_train/len(dataloaders_dict["train"].dataset)
                
                ## Compute total validation loss
                loss_valid = 0
                for inputs, labels, index, _ in dataloaders_dict["val"]:
                    y_pred = model(inputs)
                    loss_valid = loss_valid + criterion(y_pred, labels)
                loss_valid =  loss_valid/len(dataloaders_dict["val"].dataset)

                ## Print loss to console
                print('Trial: {} Epoch {}: train loss: {:0.2f} valid loss: {:0.2f}'.format(trial, epoch, loss_train.item(), loss_valid.item()))


            best_model = dict()
            best_model["modelParameters"] = copy.deepcopy(model.state_dict())
            best_model["numEpochs"] = epoch


    ######################### Learning Curve ##########################

    # plot the learning curve
    plt.figure()
    plt.plot(np.arange(0,len(learningCurve),1),learningCurve, c='blue')
    plt.plot(np.arange(0,len(valLearningCurve),1),valLearningCurve, c='orange')
    plt.title("Learing Curve", fontsize=18)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)


########################### Compute Loss ################################
    
    ######################## Validation #################################
    ## Pass validation data throught the network
    y_val, y_pred, y_no_noise = predict(dataloaders_dict["val"], model)
    
    ## Plot prediction over groundtruth
    plt.figure()
    n_samp_y = len(y_val)
    plt.plot(np.arange(n_samp_y),y_no_noise, color='blue')
    plt.plot(np.arange(n_samp_y),y_pred, color='orange')
    plt.legend(['GT','MSE Pred'])
    plt.title('Validation Data', fontsize=14)
    
    
    ############################### Test ################################
    ## Pass test data throught the network
    y_test, y_pred, y_no_noise = predict(dataloaders_dict["test"], model)
    
    ## Plot prediction over groundtruth
    plt.figure()
    n_samp_y = len(y_test)
    plt.plot(np.arange(n_samp_y),y_no_noise, color='blue')
    plt.plot(np.arange(n_samp_y),y_pred, color='orange')
    plt.legend(['GT','MSE Pred'])
    plt.title('Test Data', fontsize=14)

    return model