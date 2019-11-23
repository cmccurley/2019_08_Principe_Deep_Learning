# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:39:11 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  hw05.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-10-26
    *  Desc:  Provides coded solutions to homework 05 of EEL681,
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
from setParams import setParams
from torch.utils import data
from torch.autograd import Variable
from confusion_mat import plot_confusion_matrix
import numpy.random


######################################################################
##################### Function Definitions ###########################
######################################################################

def readData(dataFilePath, window_size, plot_raw, plot_range, add_noise):
    """
    ******************************************************************
        *  Func:      readData()
        *  Desc:      Reads data from a .txt file
        *  Inputs:    
        *             dataFilePath - path to .txt data file
        *             window_size - length of FIR filter
        *  Outputs:   
        *             X: matrix of input data, split into windows
        *             y: vector of values one-step ahead of filter
    ******************************************************************
    """
    
    ## Load data
    x = np.loadtxt(dataFilePath)
    x = x - np.mean(x)
    num_samples = len(x)
    
    ## Plot time series in a specific range
    if plot_raw:
        n_samples_plot = np.arange(plot_range[0],plot_range[1])
        plt.figure()
        plt.plot(n_samples_plot,x[n_samples_plot])
        plt.title(f'Mackey-Glass Time Series \n Range: {plot_range[0]} - {plot_range[1]}', fontsize=14)
    
    ## Partition data into windows
    X = np.zeros((num_samples - window_size - 1, window_size))
    for idx in range(0,num_samples - window_size- 1):
        X[idx,:] = x[idx:(idx+window_size)]
        
    ## Generate labels as one step ahead predition value
    
     ## Add random noise
    if add_noise:
        noise = (0.95*numpy.random.normal(loc=0.0, scale=0.5)) + (0.05*numpy.random.normal(loc=1,scale=0.5))
        y = x + noise
        
    y = x[(window_size):]
    
    return X,y

def scale_pixels(X_train, X_test):
    """
    ******************************************************************
        *  Func:      scale_pixels()
        *  Desc:      Scales pixel values between 0-1
        *  Inputs:    X_train: uint8 matrix of nSamples by nFeatures
        *             X_test: uint8 matrix of nSamples by nFeatures
        *  Outputs:   X_train and X_test with values between 0-1 as floats
    ******************************************************************
    """

    # convert from integers to floats
    train_norm = X_train.astype('float32')
    test_norm = X_test.astype('float32')

    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    # return normalized images
    return train_norm, test_norm

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

    ## Load data, subtratct mean and partition into windows
    X_train, y_train = readData(parameters["dataPath"], parameters["window_size"], parameters["plot_raw"], parameters["plot_range"], parameters["add_noise"])

    # Normalize images between 0-1
    X_train_all, X_test = scale_pixels(X_train_all, X_test)

    ####################### Define Validation Set ####################

    # partition data into training and validation
    X_train, X_val, y_train, y_val = ms.train_test_split(X_train_all, y_train_all, test_size=parameters["data_parameters"]["validationSize"], random_state=42)


#    ###################### Apply Multiscale PCA #######################
    X_train, X_val, X_test = mpca(X_train, X_val, X_test)
#    
#    X_train, X_val, X_test = mpca_with_fft(X_train, X_val, X_test)
#    
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
#    np.save('MPCA_fft_data.npy', dataSet)
#    
    ######################## Load Data ###############################
#    dataSet = np.load('MPCA_fft_data.npy',allow_pickle=True).item()    
#    
#    X_train = dataSet["X_train"]
#    y_train = dataSet["y_train"]
#    X_val = dataSet["X_val"]
#    y_val = dataSet["y_val"]
#    X_test = dataSet["X_test"]
#    y_test = dataSet["y_test"]
    

#    ###################### Convert data into torch format ############
#    X_train = torch.FloatTensor(X_train)
#    y_train = torch.LongTensor(y_train)
#    X_val = torch.FloatTensor(X_val)
#    y_val = torch.LongTensor(y_val)
#    X_test = torch.FloatTensor(X_test)
#    y_test = torch.LongTensor(y_test)


################################ Train Network ############################
#
#    print('Training Network...')
#
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
#                if ((epoch > 200) and np.sign(valLearningCurve[-1].detach().numpy() - valLearningCurve[-2].detach().numpy())):
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


    print('================ DONE ================')

