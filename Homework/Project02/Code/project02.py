# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:39:11 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  project01.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-10-26
    *  Desc:  Provides coded solutions to project 01 of EEL681,
    *         Deep Learning, taught by Dr. Jose Principe, Fall 2019.
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

    X_train_all, y_train_all = mnist_reader.load_mnist(parameters["dataPath"], kind='train')
    X_test, y_test = mnist_reader.load_mnist(parameters["dataPath"], kind='t10k')

    # Normalize images between 0-1
    X_train_all, X_test = scale_pixels(X_train_all, X_test)


#    # Plot a sample from every class
#    plot_classes(X_train_all, y_train_all)


    ####################### Define Validation Set ####################

    # partition data into training and validation
    X_train, X_val, y_train, y_val = ms.train_test_split(X_train_all, y_train_all, test_size=parameters["data_parameters"]["validationSize"], random_state=42)


    ###################### Apply Dimensionality Reduction #############
#    n_dims = 400
##    X_train, X_val, X_test = pca_dim_reduction(X_train, X_val, X_test, n_dims)
#
#    X_train, X_val, X_test = umap_dim_reduction(X_train, X_val, X_test,n_dims, y_train)
#
#    ######################## Save Data ###############################
#    dataSet = dict()
#    dataSet["X_train"] = X_train
#    dataSet["y_train"] = y_train
#    dataSet["X_val"] = X_val
#    dataSet["y_val"] = y_val
#    dataSet["X_test"] = X_test
#    dataSet["y_test"] = y_test
#
#    np.save('data_umap_400.npy', dataSet)
#
#    ######################## Load Data ###############################
#    dataSet = np.load('reduced_dim_data/data_pca_100.npy').item()
#
#    X_train = dataSet["X_train"]
#    y_train = dataSet["y_train"]
#    X_val = dataSet["X_val"]
#    y_val = dataSet["y_val"]
#    X_test = dataSet["X_test"]
#    y_test = dataSet["y_test"]
#
#    plot_dist_distributions(X_train,y_train, parameters)
##
##    ######################## Visualize Reduction ######################
#    fig, ax = plt.subplots(1, figsize=(14, 10))
#    plt.scatter(X_train[:,0],X_train[:,1], s=1, c=y_train, cmap='Spectral', alpha=1.0)
#    plt.setp(ax, xticks=[], yticks=[])
#    cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
#    cbar.set_ticks(np.arange(10))
#    cbar.set_ticklabels(parameters["classes"])
#    plt.title('Fashion MNIST Embedded via UMAP')

    ###################### Convert data into torch format ############
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

#    ########################### Apply Autoencoder ######################
#
#    model = Autoencoder100()
#    model.load_state_dict(torch.load('autoencoder_100.pth'))
#    model.eval()
#
#    X_train = model.encoder(X_train)
#    X_val = model.encoder(X_val)
#    X_test = model.encoder(X_test)
#
#    X_train = X_train.detach().numpy()
#    X_val = X_val.detach().numpy()
#    X_test = X_test.detach().numpy()
#
#    dataSet = dict()
#    dataSet["X_train"] = X_train
#    dataSet["y_train"] = y_train
#    dataSet["X_val"] = X_val
#    dataSet["y_val"] = y_val
#    dataSet["X_test"] = X_test
#    dataSet["y_test"] = y_test
##
#    np.save('data_auto_100.npy', dataSet)

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

