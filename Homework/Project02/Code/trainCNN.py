# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:57:54 2019

@author: Conma
"""


"""
***********************************************************************
    *  File:  trainCNN.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-29
    *  Desc:  Trains autoencoder, saves weights and returns model.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################

## Default packages
import os
import copy
import numpy as np

## PyTorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable

## Custom packages
from BaselineCNN import baselineCNN, baselineCNN2
import matplotlib.pyplot as plt
from utils import predict
from confusion_mat import plot_confusion_matrix

######################################################################
##################### Function Definitions ###########################
######################################################################

def trainCNN(dataloaders_dict, all_parameters):

    classes = all_parameters["classes"]
    parameters = all_parameters["cnn_parameters"]
    best_model = dict()

#    ## Check validation loss of trained model
#    print('Validating model...')
#    model = baselineCNN2()
#    model.load_state_dict(torch.load(parameters["model_save_path"]))
#    ## Compute total validation loss
#    model.eval()
#    criterion = torch.nn.CrossEntropyLoss()
#    loss_valid = 0
#    for inputs, labels in dataloaders_dict["val"]:
#        y_pred = model(inputs)
#        loss_valid = loss_valid + criterion(y_pred, labels)
#    loss_valid =  loss_valid/len(dataloaders_dict["val"])
#    print(f'Validation loss: {loss_valid}')

################################ Test trained model ###########################
    if parameters["test_model"]:
        ## Check test loss of trained model
        print('Testing model...')
        model = baselineCNN2()
        model.load_state_dict(torch.load(parameters["model_save_path"]))
        ## Compute total validation loss
        model.eval()
        y_true, y_pred = predict(dataloaders_dict["test"],model)

        ## Make confusion matrix
        acc = 1 - np.count_nonzero(y_true - y_pred)/len(dataloaders_dict["test"].dataset)
        plot_title = "CNN"
        plot_confusion_matrix(y_true, y_pred, classes, acc, normalize=False, title=plot_title)

        conf_mat_save_path = parameters["image_save_path"]
        plt.savefig(conf_mat_save_path)
        plt.close()
    else:
        ###################### Train Model ####################################
        print(f'Training Baseline CNN...')

        ############# run a number of trials, save best model ############
        for trial in range(parameters["numTrials"]):

            learningCurve = []
            valLearningCurve = []

            ####################### Define Network ###########################

            # instantiate model
    #        model = baselineCNN()

            model = baselineCNN2()

    #        load_path = os.getcwd() + '/cnn_model_parameters/baseline_cnn2.pth'
    #        model.load_state_dict(torch.load(load_path))
    #        model.load_state_dict(torch.load(parameters["model_save_path"]))

            # define loss function
            criterion = torch.nn.CrossEntropyLoss()

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
                    y_pred = model(inputs) # predict output vector

                    # compute loss (desired is the input image)
                    loss = criterion(y_pred, labels)

                    # backward pass
                    loss.backward() # computes the gradients
                    optimizer.step() # updates the weights

                    count = count + 1

                    ############### Add to validation learning curve ##############
                    if not(count%parameters["val_update"]):

                        print(f'Batch: {count}')

                        ## Add to training learning curve each batch
                        learningCurve.append(loss)

                        ## Compute total validation loss
                        model.eval()
                        loss_valid = 0
                        for inputs, labels in dataloaders_dict["val"]:
                            y_pred = model(inputs)
                            loss_valid = loss_valid + criterion(y_pred, labels)
                        loss_valid =  loss_valid/len(dataloaders_dict["val"])

                        valLearningCurve.append(loss_valid)
                        model.train()

                ####################### Update Best Model #########################
                ## Update state dictionary of best model
                best_model["modelParameters"] = copy.deepcopy(model.state_dict())
                best_model["numEpochs"] = epoch

                torch.save(best_model["modelParameters"], parameters["model_save_path"])
        ######################### Learning Curve ##############################

        # plot the learning curve
        plt.figure()
        plt.plot(np.arange(0,len(learningCurve)*parameters["val_update"],parameters["val_update"]),learningCurve, c='blue')
        plt.plot(np.arange(0,len(valLearningCurve)*parameters["val_update"],parameters["val_update"]),valLearningCurve, c='orange')
        plt.title("Learing Curve", fontsize=18)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cross-Entropy Loss', fontsize=12)

        save_path = parameters["image_save_path"] + '_learning_curve.png'
        plt.savefig(save_path)
        plt.close()

        ######################## Save Weights and Plot Images #################

        ## Save state dictionary of best model
        torch.save(best_model["modelParameters"], parameters["model_save_path"])


    return