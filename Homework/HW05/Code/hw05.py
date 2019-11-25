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
import math


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

#def MEELoss(y_pred, y_true, parameters):
#    """
#    ******************************************************************
#        *  Func:      MEELoss()
#        *  Desc:      Computes empirical estimate  of minimum error entropy.
#        *  Inputs:    
#        *             y_pred - vector of predicted labels
#        *             y_true - vector of true labels
#        *             parameters - dictionary of script parameters
#        *                          must include mee kernel bandwidth
#        *  Outputs:   
#        *             mee_loss_val: scalar of estimated mee loss
#    ******************************************************************
#    """
#    
#    gauss_norm_const = torch.empty(1, dtype=torch.float)
#    gauss_norm_const.fill_((1/(np.sqrt(2*3.14159)*parameters["mee_kernel_width"])))
#    
#    error = torch.zeros(1)
#    
#    norm_const = torch.empty(1, dtype=torch.float)
#    norm_const.fill_((1/((y_true.size()[0])**2)))
#    
#    ## get vector of instantaneous errors
#    error_vect = torch.abs(y_true.clone() - y_pred.clone())
#    
#    ## Update mee 
#    for ii in range(len(y_true)):
#        for jj in range(len(y_true)):
#            error = error + gauss_norm_const*torch.exp((-1)*(error_vect[ii].clone() - error_vect[jj].clone())/(2*(parameters["mee_kernel_width"]**2)))
#            
#    ## normalize error by number of samples squared
#    mee_loss_val = error*norm_const  
#    
#    return mee_loss_val


class MEELoss(torch.nn.Module):
    
    def __init__(self):
        super(MEELoss,self).__init__()
        
    def forward(self,y_pred, y_true, parameters):
        ## compute normalization constant for Gaussian kernel
#        gauss_norm_const = (1/(np.sqrt(2*3.14159)*parameters["mee_kernel_width"]))
        
        self.gauss_norm_const = torch.empty(1, dtype=torch.float)
        self.gauss_norm_const.fill_((1/(np.sqrt(2*3.14159)*parameters["mee_kernel_width"])))
        
        self.error = torch.zeros(1)
        
        self.norm_const = torch.empty(1, dtype=torch.float)
        self.norm_const.fill_((1/((y_true.size()[0])**2)))
        
        ## get vector of instantaneous errors
        self.error_vect = torch.abs(y_true.clone() - y_pred.clone())
        
        ## Update mee 
        for ii in range(len(y_true)):
            for jj in range(len(y_true)):
                self.error = self.error + self.gauss_norm_const*torch.exp((-1)*(self.error_vect[ii].clone() - self.error_vect[jj].clone())/(2*(parameters["mee_kernel_width"]**2)))
                
        ## normalize error by number of samples squared
        self.mee_loss_val = self.error*self.norm_const  
        
        return self.mee_loss_val


###################### Define Neural Net Class #######################
class Feedforward(torch.nn.Module):

        def __init__(self, input_size, output_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.relu = torch.nn.ReLU()
            self.tanh =  torch.nn.Tanh()
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
            hidden = self.fc4(relu)
            output = self.tanh(hidden)
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

    if parameters["gen_data"]:
        ## Load data, subtratct mean and partition into windows
        X, y = readData(parameters["dataPath"], parameters["window_size"], parameters["plot_raw"], parameters["plot_range"], parameters["add_noise"])
    
        ######################### Split data #############################
    
        ## Total number of samples in the dataset
        n_samples = len(y)
        
        ## Get number in test/valid split
        n_split = math.floor(n_samples/10)
        
        ## Separate train, valid and test sets
        X_train = X[0:(n_samples - 1000),:]
        y_train = y[0:(n_samples - 1000)]
        
        X_val = X[(n_samples - 1000):(n_samples - 1000)+500,:]
        y_val = y[(n_samples - 1000):(n_samples - 1000)+500]
        
        X_test = X[((n_samples - 1000)+500):,:]
        y_test = y[((n_samples - 1000)+500):]
        
        
        
        ######################## Save Data ###############################
        dataSet = dict()
        dataSet["X_train"] = X_train
        dataSet["y_train"] = y_train
        dataSet["X_val"] = X_val
        dataSet["y_val"] = y_val
        dataSet["X_test"] = X_test
        dataSet["y_test"] = y_test
#    
        np.save(parameters["saved_data_name"], dataSet)

    else:
        
        ######################## Load Data ###############################
        dataSet = np.load(parameters["saved_data_name"],allow_pickle=True).item()    
        
        X_train = dataSet["X_train"]
        y_train = dataSet["y_train"]
        X_val = dataSet["X_val"]
        y_val = dataSet["y_val"]
        X_test = dataSet["X_test"]
        y_test = dataSet["y_test"]
    

    ###################### Convert data into torch format ############
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

if (parameters["loss_type"] == 'mse'):
######################### Train Network with MSE ############################

    print('Training Network with MSE Loss...')

    ############# run a number of trials, save best model ############
    for trial in range(parameters["numTrials"]):

        learningCurve = []
        valLearningCurve = []

        ####################### Define Network ###########################
        inputSize = X_train.shape[1]

        # instantiate model
        model = Feedforward(inputSize,  parameters["outputSize"])

        # define loss function
        criterion = torch.nn.MSELoss()

        # define optimizer (stochastic gradient descent)
        optimizer = torch.optim.Adamax(model.parameters(), parameters["learningRate"])

        ##################### Train the Network ##########################

        model.train()

        ################# Train a single network #####################
        for epoch in range(parameters["numEpochs"]):

            for idx in range(X_train.shape[0]):
                #set gradients to zero
                optimizer.zero_grad()
    
                # forward pass
                y_pred = model(X_train[idx,:]) # predict output vector
    
                # compute loss
                loss = criterion(y_pred, y_train[idx])
    
                # backward pass
                loss.backward() # computes the gradients
                optimizer.step() # updates the weights
    
                if not(idx %  20):
                    learningCurve.append(loss)
#                    model.eval()
#                    valLearningCurve.append(criterion(model(X_val),y_val))
                    model.train()

#                # if gradient of validation goes positive, stop training
#                if ((epoch > 200) and np.sign(valLearningCurve[-1].detach().numpy() - valLearningCurve[-2].detach().numpy())):
#                    break

            if not(epoch % 1):
                y_pred = model(X_train)
                loss_train = criterion(y_pred, y_train)
                y_pred = model(X_val)
                loss_valid = criterion(y_pred, y_val)
                print('Trial: {} Epoch {}: train loss: {} valid loss: {}'.format(trial, epoch, loss_train.item(), loss_valid.item()))


            best_model = dict()
            best_model["modelParameters"] = copy.deepcopy(model.state_dict())
            best_model["numEpochs"] = epoch


    ######################### Learning Curve ##########################

    # retrieve optimal parameters
#    learningCurve = best_model["learningCurve"]
#    valLearningCurve = best_model["valLearningCurve"]

    # plot the learning curve
    plt.figure()
    plt.plot(np.arange(0,len(learningCurve),1),learningCurve, c='blue')
#    plt.plot(np.arange(0,len(valLearningCurve),1),valLearningCurve, c='orange')
    plt.title("Learing Curve", fontsize=18)
    plt.xlabel('Iteration', fontsize=12)
    
    if (parameters["loss_type"] == 'mse'):
        plt.ylabel('MSE Loss', fontsize=12)
    elif(parameters["loss_type"] == 'mee'):
         plt.ylabel('MEE Loss', fontsize=12)   

#    plt.legend(['Training', 'Validation'])
#    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
#    plt.close()


########################### Compute Loss ################################
    plt.figure()
    y_val = y_val.detach().numpy()
    n_samp_y = len(y_val)
    plt.plot(np.arange(n_samp_y),y_val, color='blue')
    
    y_pred = model(X_val)
    plt.plot(np.arange(n_samp_y),y_pred.detach().numpy(), color='orange')
    plt.legend(['GT','MSE Pred'])
    plt.title('Validation Data', fontsize=14)
    
    plt.figure()
    y_test = y_test.detach().numpy()
    n_samp_y = len(y_test)
    plt.plot(np.arange(n_samp_y),y_test, color='blue')
    
    y_pred = model(X_test)
    plt.plot(np.arange(n_samp_y),y_pred.detach().numpy(), color='orange')
    plt.legend(['GT','MSE Pred'])
    plt.title('Test Data', fontsize=14)
    
    
elif(parameters["loss_type"] == 'mee'):
    
######################### Train Network with MSE ############################

    print('Training Network with MEE Loss...')

    ############# run a number of trials, save best model ############
    for trial in range(parameters["numTrials"]):

        learningCurve = []
        valLearningCurve = []

        ####################### Define Network ###########################
        inputSize = X_train.shape[1]

        # instantiate model
        model = Feedforward(inputSize,  parameters["outputSize"])

        # define optimizer (stochastic gradient descent)
        optimizer = torch.optim.Adamax(model.parameters(), parameters["learningRate"])
        
        criterion = MEELoss()

        ##################### Train the Network ##########################
        
#        plt.figure()
#        y_val = y_val.detach().numpy()
#        n_samp_y = len(y_val)
#        plt.plot(np.arange(n_samp_y),y_val, color='blue')
#        
#        y_pred = model(X_val)
#        plt.plot(np.arange(n_samp_y),y_pred.detach().numpy(), color='orange')
#        plt.legend(['GT','MEE Pred'])
#        plt.title('Validation Data', fontsize=14)
        

        model.train()

        ################# Train a single network #####################
        for epoch in range(parameters["numEpochs"]):

            y_pred = torch.empty(parameters["batch_size"], requires_grad=True)
            y_true = torch.empty(parameters["batch_size"])
            batch_idx = 0
            batch_num = 0
            
            for idx in range(X_train.shape[0]):
    
                # forward pass (mini-batch of 20 samples)
                y_pred[batch_idx] = model(X_train[idx,:].clone()) # predict output 
                y_true[batch_idx] = y_train[idx].clone()
                batch_idx = batch_idx + 1
    
                ## update network and append loss to learning curve
                if (batch_idx == parameters["batch_size"]):
                    batch_idx = 0
                    batch_num = batch_num + 1

                    ## Compute MEE loss
#                    loss  = MEELoss(y_pred, y_true, parameters)
#                    loss  = criterion(y_pred, y_true, parameters)
                    loss  = criterion.forward(y_pred, y_true, parameters)
                    
                    # backward pass
#                    optimizer.zero_grad()
#                    loss.retain_grad()
                    loss.backward() # computes the gradients
                    optimizer.step() # updates the 
                    
                    
                    print(f'Batch: {batch_num}')
                    
                    learningCurve.append(loss)
#                    model.eval()
#                    valLearningCurve.append(criterion(model(X_val),y_val))
                    model.train()
                    
            print(f'epoch {epoch}')

            

#                # if gradient of validation goes positive, stop training
#                if ((epoch > 200) and np.sign(valLearningCurve[-1].detach().numpy() - valLearningCurve[-2].detach().numpy())):
#                    break

#            if not(epoch % 1):
#                y_pred = model(X_train)
#                loss_train = criterion(y_pred, y_train)
#                y_pred = model(X_val)
#                loss_valid = criterion(y_pred, y_val)
##                print('Trial: {} Epoch {}: train loss: {} valid loss: {}'.format(trial, epoch, loss_train.item(), loss_valid.item()))


            best_model = dict()
            best_model["modelParameters"] = copy.deepcopy(model.state_dict())
            best_model["numEpochs"] = epoch


    ######################### Learning Curve ##########################

    # retrieve optimal parameters
#    learningCurve = best_model["learningCurve"]
#    valLearningCurve = best_model["valLearningCurve"]

    # plot the learning curve
    plt.figure()
    plt.plot(np.arange(0,len(learningCurve),1),learningCurve, c='blue')
#    plt.plot(np.arange(0,len(valLearningCurve),1),valLearningCurve, c='orange')
    plt.title("Learing Curve", fontsize=18)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('MEE Loss', fontsize=12) 

        

#    plt.legend(['Training', 'Validation'])
#    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
#    plt.close()


########################### Compute Loss ################################
    plt.figure()
    y_val = y_val.detach().numpy()
    n_samp_y = len(y_val)
    plt.plot(np.arange(n_samp_y),y_val, color='blue')
    
    y_pred = model(X_val)
    plt.plot(np.arange(n_samp_y),y_pred.detach().numpy(), color='orange')
    plt.legend(['GT','MEE Pred'])
    plt.title('Validation Data', fontsize=14)
    
#    plt.figure()
#    y_test = y_test.detach().numpy()
#    n_samp_y = len(y_test)
#    plt.plot(np.arange(n_samp_y),y_test, color='blue')
#    
#    y_pred = model(X_test)
#    plt.plot(np.arange(n_samp_y),y_pred.detach().numpy(), color='orange')
#    plt.legend(['GT','MSE Pred'])
#    plt.title('Test Data', fontsize=14)
            
    


    print('================ DONE ================')

