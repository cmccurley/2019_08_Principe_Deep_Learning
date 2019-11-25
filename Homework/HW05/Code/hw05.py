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
from torch.utils.data import Dataset, DataLoader
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
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split


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



class MGdata(Dataset):
    """
    ******************************************************************
        *  Func:      MGdata()
        *  Desc:      Dataloader for the MG dataset.
        *  Inputs:    
        *             dataFilePath - path to .txt data file
        *             window_size - length of FIR filter
        *             Noise - boolean to add noise to desired
        *  Outputs:   
        *             sequence: matrix of input data, split into windows
        *             label: list of values one-step ahead of filter
        *             index : location of samples in the original data matrix
    ******************************************************************
    """
    def __init__(self, dataFilePath, window = 20, Noise = False):
        
        self.targets = []
        
        ## Load the data
        input_data = np.loadtxt(dataFilePath)
        input_data = input_data - np.mean(input_data)
        
        ## Partition data into windows
        self.inout_seq = []
        L = len(input_data)
        for i in range(L-window):
            train_seq = input_data[i:i+window]
            train_label = input_data[i+window:i+window+1]
            #Add noise using Middleton model if desired
            if (Noise):
                train_label_no_noise = train_label
                train_label += (.95*np.random.normal(loc=0,scale=np.sqrt(.5)) +
                                .05*np.random.normal(loc=1,scale=np.sqrt(.5)))
            self.inout_seq.append({  # sequence and desired
                    "sequence": train_seq,
                    "label": train_label,
                    "label_no_noise": train_label_no_noise
                })
            self.targets.append(train_label)
            
            self.Noise = Noise


    def __len__(self):
        return len(self.inout_seq)

    def __getitem__(self, index):

        datafiles = self.inout_seq[index]

        sequence = torch.FloatTensor(datafiles["sequence"])

        label_file = datafiles["label"]
        label = torch.FloatTensor(label_file)
    

        if self.Noise:  
            label_no_noise = torch.FloatTensor(datafiles["label_no_noise"])
            return sequence, label, index, label_no_noise
        else:
            return sequence, label, index 
        
    
def predict(dataloader,model):
    #Initialize and accumalate ground truth and predictions
    GT = np.array(0)
    Predictions = np.array(0)
    model.eval()
        
    ## Iterate over data.
    with torch.no_grad():
        for inputs, labels, index, labels_no_noise in dataloader:
            # forward
            outputs = model(inputs)
            
            #If validation, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            GT_no_noise = np.concatenate((GT,labels_no_noise.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,outputs.detach().cpu().numpy()),axis=None)
            
    return GT[1:],Predictions[1:], GT_no_noise[1:]


class MEELoss(torch.nn.Module):
    
    def __init__(self):
        super(MEELoss,self).__init__()
        
    def forward(self, y_pred, y_true, parameters):
        """
        ******************************************************************
            *  Func:      MEELoss()
            *  Desc:      Computes empirical estimate  of minimum error entropy.
            *  Inputs:    
            *             y_pred - vector of predicted labels
            *             y_true - vector of true labels
            *             parameters - dictionary of script parameters
            *                          must include mee kernel bandwidth
            *  Outputs:   
            *             mee_loss_val: scalar of estimated mee loss
        ******************************************************************
        """
        
        y_true = y_true
        
        n_samples = y_true.size()[0]
        
        ## compute normalizing constant for Gaussian kernel
        self.gauss_norm_const = torch.empty(1, dtype=torch.float)
        self.gauss_norm_const.fill_((1/(np.sqrt(2*np.pi)*parameters["mee_kernel_width"])))
        
        ## Compute normalizing constnant on error
        self.norm_const = torch.empty(1, dtype=torch.float)
        self.norm_const.fill_((1/((y_true.size()[0])**2)))
        
        ## get vector of instantaneous errors
        self.error_vect = torch.abs(y_true.clone() - y_pred.clone())
        
        ## Update mee 
        self.error = torch.pow(self.error_vect,2).sum(dim=1,keepdim=True).expand(n_samples,n_samples)
        self.error = self.error + self.error.t()
        self.error.addmm_(1,-2,self.error_vect,self.error_vect.t())
        self.error = self.error.clamp(min=1e-12).sqrt()
        
        ## Apply kernel
        self.mee_loss = torch.sum(self.gauss_norm_const*torch.exp(-((self.error)**2)/((2*(parameters["mee_kernel_width"]**2)))))
        
        ## Normalize by number of samples squared
        self.mee_loss_val = (-1)*torch.sum(self.mee_loss)*self.norm_const*(1/parameters["mee_kernel_width"])
        
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
            
#            hidden = self.fc1(x)
#            tanh = self.tanh(hidden)
#            hidden = self.fc2(tanh)
#            tanh = self.tanh(hidden)
#            hidden = self.fc3(tanh)
#            tanh = self.tanh(hidden)
#            hidden = self.fc4(tanh)
#            output = self.tanh(hidden)
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

    ## Create dataset
    dataset = MGdata(parameters["dataPath"], parameters["window_size"], parameters["add_noise"])
    
    indices = np.arange(len(dataset))
    y = dataset.targets
        
    ## Split training, validation and test sets
    y_train,y_val,train_indices,val_indices = train_test_split(y,indices,test_size = .2,random_state=24)
    y_val,y_test,val_indices,test_indices = train_test_split(y_val,val_indices,test_size =.5,random_state=24)
    
    ## Create data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    mg_datasets = {'train': dataset, 'val': dataset, 'test': dataset}
    
    
    ## Create training and validation dataloaders
    dataloaders_dict = {'train': torch.utils.data.DataLoader(mg_datasets['train'], batch_size=parameters["batch_size"],
                                               sampler=train_sampler, shuffle=False,num_workers=0),
                        'val': torch.utils.data.DataLoader(mg_datasets['val'],batch_size=parameters["batch_size"],
                                               sampler=valid_sampler, shuffle=False,num_workers=0),
                        'test': torch.utils.data.DataLoader(mg_datasets['test'], batch_size=parameters["batch_size"],
                                               sampler=test_sampler, shuffle=False,num_workers=0) }
    
#############################################################################
############################### Train with MSE ##############################
#############################################################################
if (parameters["loss_type"] == 'mse'):
######################### Train Network with MSE ############################

    print('Training Network with MSE Loss...')

    ############# run a number of trials, save best model ############
    for trial in range(parameters["numTrials"]):

        learningCurve = []
        valLearningCurve = []

        ####################### Define Network ###########################
        inputSize = parameters["window_size"]

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
            
            for inputs, labels, index in dataloaders_dict["train"]:

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
                for inputs, labels, index in dataloaders_dict["train"]:
                    y_pred = model(inputs)
                    loss_train = loss_train + criterion(y_pred, labels)
                loss_train =  loss_train/len(dataloaders_dict["train"].dataset)
                learningCurve.append(loss_train)
                
                ## Compute total validation loss
                loss_valid = 0
                for inputs, labels, index in dataloaders_dict["val"]:
                    y_pred = model(inputs)
                    loss_valid = loss_valid + criterion(y_pred, labels)
                loss_valid =  loss_valid/len(dataloaders_dict["val"].dataset)
                
                valLearningCurve.append(loss_valid)
                model.train()

            ## Update epoch training status
            if not(epoch % 1):
                
                ## Compute total training loss
                loss_train = 0
                for inputs, labels, index in dataloaders_dict["train"]:
                    y_pred = model(inputs)
                    loss_train = loss_train + criterion(y_pred, labels)
                loss_train =  loss_train/len(dataloaders_dict["train"].dataset)
                
                ## Compute total validation loss
                loss_valid = 0
                for inputs, labels, index in dataloaders_dict["val"]:
                    y_pred = model(inputs)
                    loss_valid = loss_valid + criterion(y_pred, labels)
                loss_valid =  loss_valid/len(dataloaders_dict["val"].dataset)

                ## Print loss to console
                print('Trial: {} Epoch {}: train loss: {:0.2f} valid loss: {:0.2f}'.format(trial, epoch, loss_train.item(), loss_valid.item()))


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
    plt.plot(np.arange(0,len(valLearningCurve),1),valLearningCurve, c='orange')
    plt.title("Learing Curve", fontsize=18)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)

#    plt.legend(['Training', 'Validation'])
#    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
#    plt.close()


########################### Compute Loss ################################
    
    ######################## Validation #################################
    ## Pass validation data throught the network
    y_val, y_pred = predict(dataloaders_dict["val"], model)
    
    ## Plot prediction over groundtruth
    plt.figure()
    n_samp_y = len(y_val)
    plt.plot(np.arange(n_samp_y),y_val, color='blue')
    plt.plot(np.arange(n_samp_y),y_pred, color='orange')
    plt.legend(['GT','MSE Pred'])
    plt.title('Validation Data', fontsize=14)
    
    
    ############################### Test ################################
    ## Pass test data throught the network
    y_test, y_pred = predict(dataloaders_dict["test"], model)
    
    ## Plot prediction over groundtruth
    plt.figure()
    n_samp_y = len(y_test)
    plt.plot(np.arange(n_samp_y),y_test, color='blue')
    plt.plot(np.arange(n_samp_y),y_pred, color='orange')
    plt.legend(['GT','MSE Pred'])
    plt.title('Test Data', fontsize=14)
    
#############################################################################
############################### Train with MEE ##############################
#############################################################################
elif(parameters["loss_type"] == 'mee'):
    
######################### Train Network with MSE ############################

    print('Training Network with MEE Loss...')
    

    ############# run a number of trials, save best model ############
    for trial in range(parameters["numTrials"]):

        learningCurve = []
        valLearningCurve = []

        ####################### Define Network ###########################
        inputSize = parameters["window_size"]

        # instantiate model
        model = Feedforward(inputSize,  parameters["outputSize"])

        # define loss function
        criterion = MEELoss()

        # define optimizer (stochastic gradient descent)
        optimizer = torch.optim.Adamax(model.parameters(), parameters["learningRate"])

        ##################### Train the Network ##########################

        model.train()

        ################# Train a single network #####################
        for epoch in range(parameters["numEpochs"]):
            
            for inputs, labels, index, _ in dataloaders_dict["train"]:

                #set gradients to zero
                optimizer.zero_grad()
    
                # forward pass
                y_pred = model(inputs) # predict output vector
    
                # compute loss
                loss = criterion(y_pred, labels, parameters)
    
                # backward pass
                loss.backward() # computes the gradients
                optimizer.step() # updates the weights
    
    
    
                ############### Add to learning curve #########################
                ## Compute total training loss
                model.eval()
                loss_train = 0
                for inputs, labels, index,_ in dataloaders_dict["train"]:
                    y_pred = model(inputs)
                    loss_train = loss_train + criterion(y_pred, labels, parameters)
                loss_train =  loss_train/len(dataloaders_dict["train"].dataset)
                learningCurve.append(loss_train)
                
                ## Compute total validation loss
                loss_valid = 0
                for inputs, labels, index,_ in dataloaders_dict["val"]:
                    y_pred = model(inputs)
                    loss_valid = loss_valid + criterion(y_pred, labels, parameters)
                loss_valid =  loss_valid/len(dataloaders_dict["val"].dataset)
                
                valLearningCurve.append(loss_valid)
                model.train()

            ## Update epoch training status
            if not(epoch % 1):
                
                ## Compute total training loss
                loss_train = 0
                for inputs, labels, index,_ in dataloaders_dict["train"]:
                    y_pred = model(inputs)
                    loss_train = loss_train + criterion(y_pred, labels, parameters)
                loss_train =  loss_train/len(dataloaders_dict["train"].dataset)
                
                ## Compute total validation loss
                loss_valid = 0
                for inputs, labels, index, _ in dataloaders_dict["val"]:
                    y_pred = model(inputs)
                    loss_valid = loss_valid + criterion(y_pred, labels, parameters)
                loss_valid =  loss_valid/len(dataloaders_dict["val"].dataset)

                ## Print loss to console
                print('Trial: {} Epoch {}: train loss: {:0.2f} valid loss: {:0.2f}'.format(trial, epoch, loss_train.item(), loss_valid.item()))


            best_model = dict()
            best_model["modelParameters"] = copy.deepcopy(model.state_dict())
            best_model["numEpochs"] = epoch


    ######################### Learning Curve ##########################

    # retrieve optimal parameters
#    learningCurve = best_model["learningCurve"]
#    valLearningCurve = best_model["valLearningCurve"]

    # plot the learning 
    for idx in range(len(learningCurve)):
        learningCurve[idx] = (-1)*torch.exp(learningCurve[idx])
        valLearningCurve[idx] = (-1)*torch.exp(valLearningCurve[idx])
    
    plt.figure()
    plt.plot(np.arange(0,len(learningCurve),1),learningCurve, c='blue')
    plt.plot(np.arange(0,len(valLearningCurve),1),valLearningCurve, c='orange')
    plt.title("Learing Curve", fontsize=18)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Information Potential', fontsize=12)

#    plt.legend(['Training', 'Validation'])
#    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
#    plt.close()


########################### Compute Loss ################################
    
    ######################## Validation #################################
    ## Pass validation data throught the network
    y_val, y_pred, y_no_noise = predict(dataloaders_dict["val"], model)
    
    ## Plot prediction over groundtruth
    plt.figure()
    n_samp_y = len(y_val)
    plt.plot(np.arange(n_samp_y),y_no_noise, color='blue')
    plt.plot(np.arange(n_samp_y),y_pred, color='red')
    plt.legend(['GT','MEE Pred'])
    plt.title('Validation Data', fontsize=14)
    
    
    ############################### Test ################################
    ## Pass test data throught the network
    y_test, y_pred, y_no_noise = predict(dataloaders_dict["test"], model)
    
    ## Plot prediction over groundtruth
    plt.figure()
    n_samp_y = len(y_test)
    plt.plot(np.arange(n_samp_y),y_no_noise, color='blue')
    plt.plot(np.arange(n_samp_y),y_pred, color='red')
    plt.legend(['GT','MEE Pred'])
    plt.title('Test Data', fontsize=14)
    

    


    print('================ DONE ================')

