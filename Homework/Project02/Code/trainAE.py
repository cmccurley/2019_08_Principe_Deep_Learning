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
            count = 0
            
            print(f'Trial: {trial} Epoch: {epoch}')
            
            for inputs, labels in dataloaders_dict["train"]:

                #set gradients to zero
                optimizer.zero_grad()
    
                # forward pass
                y_pred = model(inputs.flatten(start_dim=1)) # predict output vector
    
                # compute loss (desired is the input image)
                loss = criterion(y_pred, inputs.flatten(start_dim=1))
    
                # backward pass
                loss.backward() # computes the gradients
                optimizer.step() # updates the weights
                
                ## Add to training learning curve each batch
                learningCurve.append(loss)
            
                count = count + 1
    
                ############### Add to validation learning curve ##############
                if not(count%parameters["val_update"]):
            
                    print(f'Batch: {count}')
                    
                    ## Compute total validation loss
                    model.eval()
                    loss_valid = 0
                    for inputs, labels in dataloaders_dict["val"]:
                        y_pred = model(inputs.flatten(start_dim=1))
                        loss_valid = loss_valid + criterion(y_pred, inputs.flatten(start_dim=1))
                    loss_valid =  loss_valid/len(dataloaders_dict["val"])
                    
                    valLearningCurve.append(loss_valid)
                    model.train()

            ####################### Update Best Model #########################
            ## Update state dictionary of best model
            best_model = dict()
            best_model["modelParameters"] = copy.deepcopy(model.state_dict())
            best_model["numEpochs"] = epoch


    ######################### Learning Curve ##############################

    # plot the learning curve
    plt.figure()
    plt.plot(np.arange(0,len(learningCurve),1),learningCurve, c='blue')
    plt.plot(np.arange(0,len(valLearningCurve)*parameters["val_update"],parameters["val_update"]),valLearningCurve, c='orange')
    plt.title("Learing Curve", fontsize=18)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)


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