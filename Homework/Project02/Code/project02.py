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
    model = trainAE(dataloaders_dict, parameters)


############################### Train Network ############################

#    params = {'batch_size': 128,
#          'shuffle': True,
#          'num_workers': 6}
#    training_set = my_data(X_train,y_train)
#    training_generator = data.DataLoader(training_set, **params)


#    ############# run a number of trials, save best model ############
#    for trial in range(parameters["numTrials"]):
#
#        learningCurve = []
#        valLearningCurve = []
#
#        ####################### Define Network ###########################
#        inputSize = X_train.shape[1]
#
#        # instantiate model
#        model = Feedforward(inputSize,  parameters["outputSize"])
#
#        # define loss function
#        criterion = torch.nn.CrossEntropyLoss()
#
#        # define optimizer (stochastic gradient descent)
#        optimizer = torch.optim.Adamax(model.parameters(), parameters["learningRate"])
#
#        ##################### Train the Network ##########################
#
#        model.train()
#
#        ################# train a single network #####################
#        for epoch in range(parameters["numEpochs"]):
#
#            #set gradients to zero
#            optimizer.zero_grad()
#
##            for idx, batch in enumerate(training_generator):
##                X_train = batch["image"]
##                y_train = batch["label"]
#
#            # forward pass
#            y_pred = model(X_train) # predict output vector
#
#            # compute loss
#            loss = criterion(y_pred, y_train)
#
#            # backward pass
#            loss.backward() # computes the gradients
#            optimizer.step() # updates the weights
#
#            if not(epoch %  parameters["updateIter"]):
#                learningCurve.append(loss)
#                model.eval()
#                valLearningCurve.append(criterion(model(X_val),y_val))
#                model.train()
#
#                # if gradient of validation goes positive, stop training
#                if ((epoch > 600) and np.sign(valLearningCurve[-1].detach().numpy() - valLearningCurve[-2].detach().numpy())):
#                    break
#
#            if not(epoch % 10):
#                print('Trial: {} Epoch {}: train loss: {}'.format(trial, epoch, loss.item()))
#
#
#
#            if (trial==0):
#                best_model = dict()
#                best_model["modelParameters"] = copy.deepcopy(model.state_dict())
#                best_model["learningCurve"] = learningCurve
#                best_model["valLearningCurve"] = valLearningCurve
#                best_model["numEpochs"] = epoch
#                best_model["validationLoss"] = valLearningCurve[-1]
#            else:
#                if (valLearningCurve[-1] > best_model["validationLoss"]):
#                    best_model["modelParameters"] = copy.deepcopy(model.state_dict())
#                    best_model["learningCurve"] = learningCurve
#                    best_model["valLearningCurve"] = valLearningCurve
#                    best_model["numEpochs"] = epoch
#                    best_model["validationLoss"] = valLearningCurve[-1]
#
#
#    ######################### Learning Curve ##########################
#
#    # retrieve optimal parameters
#    learningCurve = best_model["learningCurve"]
#    valLearningCurve = best_model["valLearningCurve"]
#
#    # plot the learning curve
#    plt.figure()
#    plt.plot(parameters["updateIter"]*np.arange(0,len(learningCurve),1),learningCurve, c='blue')
#    plt.plot(parameters["updateIter"]*np.arange(0,len(valLearningCurve),1),valLearningCurve, c='orange')
#    plt.title("Learing Curve", fontsize=18)
#    plt.xlabel('Iteration', fontsize=12)
#    plt.ylabel('Cross-Entropy Loss', fontsize=12)
#    plt.legend(['Training', 'Validation'])
##    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
##    plt.close()
#
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


##################################################################
############################# Autoencoder ########################
##################################################################


##    params = {'batch_size': 10,
##              'shuffle': True,
##              'num_workers': 6}
##    training_set = my_data(X_train,y_train)
##    training_generator = data.DataLoader(training_set, **params)
#
#
############## run a number of trials, save best model ############
#    print('Training autoencoder...')
#    for trial in range(parameters["numTrials"]):
#
#        learningCurve = []
#        valLearningCurve = []
#
#        ####################### Define Network ###########################
#        inputSize = X_train.shape[1]
#
#        # instantiate model
#        model = Autoencoder400()
#
#        # define loss function
#        criterion = torch.nn.MSELoss()
#
#        # define optimizer (stochastic gradient descent)
#        optimizer = torch.optim.Adamax(model.parameters(), parameters["learningRate"])
#
#        ##################### Train the Network ##########################
#
#        model.train()
#
#        ################# train a single network #####################
#        for epoch in range(parameters["numEpochs"]):
#
#            #set gradients to zero
#            optimizer.zero_grad()
#
##            for idx, batch in enumerate(training_generator):
##                X_train = batch["image"]
##                y_train = batch["label"]
#
#            # forward pass
#            y_pred = model(X_train) # predict output vector
#
#            # compute loss
#            loss = criterion(y_pred, X_train)
#
#            # backward pass
#            loss.backward() # computes the gradients
#            optimizer.step() # updates the weights
#
#            if not(epoch %  parameters["updateIter"]):
#                learningCurve.append(loss)
#                model.eval()
#                valLearningCurve.append(criterion(model(X_val),X_val))
#                model.train()
#
#                # if gradient of validation goes positive, stop training
#                if ((epoch > 20) and np.sign(valLearningCurve[-1].detach().numpy() - valLearningCurve[-2].detach().numpy())):
#                    break
#
#            if not(epoch % 1):
#                print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
#
#
#
#        if (trial==0):
#            best_model = dict()
#            best_model["modelParameters"] = copy.deepcopy(model.state_dict())
#            best_model["learningCurve"] = learningCurve
#            best_model["valLearningCurve"] = valLearningCurve
#            best_model["numEpochs"] = epoch
#            best_model["validationLoss"] = valLearningCurve[-1]
#        else:
#            if (valLearningCurve[-1] > best_model["validationLoss"]):
#                best_model["modelParameters"] = copy.deepcopy(model.state_dict())
#                best_model["learningCurve"] = learningCurve
#                best_model["valLearningCurve"] = valLearningCurve
#                best_model["numEpochs"] = epoch
#                best_model["validationLoss"] = valLearningCurve[-1]
#
########################## Learning Curve ##########################
##
#    # retrieve optimal parameters
#    learningCurve = best_model["learningCurve"]
#    valLearningCurve = best_model["valLearningCurve"]
#
#    # plot the learning curve
#    plt.figure()
#    plt.plot(parameters["updateIter"]*np.arange(0,len(learningCurve),1),learningCurve, c='blue')
#    plt.plot(parameters["updateIter"]*np.arange(0,len(valLearningCurve),1),valLearningCurve, c='orange')
#    plt.title("Learing Curve", fontsize=18)
#    plt.xlabel('Iteration', fontsize=12)
#    plt.ylabel('MSE', fontsize=12)
#    plt.legend(['Training', 'Validation'])
##    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
##    plt.close()


    print('================ DONE ================')

