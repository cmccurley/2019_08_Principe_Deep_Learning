# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:39:11 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  hw04.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-10-26
    *  Desc:  Provides coded solutions to homework 04 of EEL681,
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
from torch.autograd import Variable
import seaborn as sns
from sklearn.decomposition import PCA
from  PIL import Image



######################################################################
##################### Function Definitions ###########################
######################################################################

def readData(dataFilePath, labelFilePath, skipHeader):
    """
    ******************************************************************
        *  Func:      readData()
        *  Desc:      Reads data from a .asc file
        *  Inputs:
        *  Outputs:
    ******************************************************************
    """
    X = np.loadtxt(dataFilePath, skiprows=skipHeader)
    y = np.loadtxt(labelFilePath, skiprows=skipHeader)

    return X, y

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


def plot_confusion_matrix(y_true, y_pred, classes, totalLoss, normalize=False, title=None, cmap=plt.cm.Blues):

    """
    ******************************************************************
        *  Func:      plot_confusion_matrix()
        *  Desc:      This function was borrowed from scikit-learn.org
        *  Inputs:
        *  Outputs:
    ******************************************************************
    """

    if not(title):
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print(f'Confusion matrix \n Test Loss: {testLoss}')

    print(cm)

    accuracy = 1 - testLoss

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=f'Confusion matrix \n Test Loss: %.2f' % testLoss,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax

def mpca(X_train, X_val, X_test):
    """
    ******************************************************************
        *  Func:      mpca()
        *  Desc:
        *  Inputs:
        *  Outputs:
    ******************************************************************
    """
    mu = np.mean(X_train,axis=0)
    X_train = X_train - np.mean(X_train,axis=0)

    return X_train, X_val, X_test

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

    X_train_all, y_train_all = mnist_reader.load_mnist(parameters["dataPath"], kind='train')
    X_test, y_test = mnist_reader.load_mnist(parameters["dataPath"], kind='t10k')

    # Normalize images between 0-1
    X_train_all, X_test = scale_pixels(X_train_all, X_test)


#    # Plot a sample from every class
#    plot_classes(X_train_all, y_train_all)


    ####################### Define Validation Set ####################

    # partition data into training and validation
    X_train, X_val, y_train, y_val = ms.train_test_split(X_train_all, y_train_all, test_size=parameters["data_parameters"]["validationSize"], random_state=42)


    ###################### Apply Multiscale PCA #######################
#    X_train, X_val, X_test = mpca(X_train, X_val, X_test)
    
    
    ####################### Training Set ##############################
    
    ## Create multiple resolutions of the images
    nSamp = X_train.shape[0] # number of samples
    
    print('Creating multiple resolutions of the training data...')
    
    ## Level 1: scale in half
    print('Creating level 1 resolution...')
    X1 = np.empty((nSamp,14*14))
    for idx in range(0,nSamp):
        X1[idx,:] = np.array(Image.fromarray(X_train[idx,:].reshape(28,28)).resize((14,14))).reshape(14*14)
        
    ## Level 2: scale level 1 in half
    print('Creating level 2 resolution...')
    X2 = np.empty((nSamp,7*7))
    for idx in range(0,nSamp):
        X2[idx,:] = np.array(Image.fromarray(X1[idx,:].reshape(14,14)).resize((7,7))).reshape(7*7)
        
    ## Level 3: scale level 2 in half
    print('Creating level 3 resolution...')
    X3 = np.empty((nSamp,3*3))
    for idx in range(0,nSamp):
        X3[idx,:] = np.array(Image.fromarray(X2[idx,:].reshape(7,7)).resize((3,3))).reshape(3*3)
    
    ## Apply PCA to training data
    print('Applying PCA to training data...')
    
    ## Subtract means
    level_1_mean = np.mean(X1,axis=0)
    level_2_mean = np.mean(X2,axis=0)
    level_3_mean = np.mean(X3,axis=0)
    
    X1 = X1 - level_1_mean
    X2 = X2 - level_2_mean
    X3 = X3 - level_3_mean
    
    pca1 = PCA()
    pca1.fit(X1)
    eigVec1 = pca1.components_
    eigVal1 = pca1.explained_variance_
    
    X1_trans = np.dot(X1, eigVec1.T)
    
    pca2 = PCA()
    pca2.fit(X2)
    eigVec2 = pca2.components_
    eigVal2 = pca2.explained_variance_
    
    X2_trans = np.dot(X2, eigVec2.T)
    
    pca3 = PCA()
    pca3.fit(X3)
    eigVec3 = pca3.components_
    eigVal3 = pca3.explained_variance_
    
    X3_trans = np.dot(X3, eigVec3.T)
    
    
    print('Creating new feature vectors for training set...')
    nFeatures = (3*(14*14))+(3*(7*7))+(3*(3*3))
    X_train_new = np.zeros((nSamp,nFeatures))
    for idx in range(0,nSamp):
#    for idx in range(0,1):
        
        ## level 1
        x20 = X1[idx,:]
        x21 = X1_trans[idx,0]*eigVec1[0,:] + X1_trans[idx,1]*eigVec1[1,:] + X1_trans[idx,2]*eigVec1[2,:]
        x22 = X1_trans[idx,0]*eigVec1[0,:] + X1_trans[idx,1]*eigVec1[1,:]
        x23 = X1_trans[idx,0]*eigVec1[0,:]
        
        x20 = x20.reshape((14*14))
        x21 = x21.reshape((14*14))
        x22 = x22.reshape((14*14))
        x23 = x23.reshape((14*14))    
        
        currentSamp1 = np.concatenate((x21,x22,x23))
        
        ## level 2
        x20 = X2[idx,:]
        x21 = X2_trans[idx,0]*eigVec2[0,:] + X2_trans[idx,1]*eigVec2[1,:] + X2_trans[idx,2]*eigVec2[2,:]
        x22 = X2_trans[idx,0]*eigVec2[0,:] + X2_trans[idx,1]*eigVec2[1,:]
        x23 = X2_trans[idx,0]*eigVec2[0,:]
        
        x20 = x20.reshape((7*7))
        x21 = x21.reshape((7*7))
        x22 = x22.reshape((7*7))
        x23 = x23.reshape((7*7))
#    
        currentSamp2 = np.concatenate((x21,x22,x23))
        
        ## level 3
        x20 = X3[idx,:]
        x21 = X3_trans[idx,0]*eigVec3[0,:] + X3_trans[idx,1]*eigVec3[1,:] + X3_trans[idx,2]*eigVec3[2,:]
        x22 = X3_trans[idx,0]*eigVec3[0,:] + X3_trans[idx,1]*eigVec3[1,:]
        x23 = X3_trans[idx,0]*eigVec3[0,:]
        
        x20 = x20.reshape((3*3))
        x21 = x21.reshape((3*3))
        x22 = x22.reshape((3*3))
        x23 = x23.reshape((3*3))
#    
        currentSamp3 = np.concatenate((x21,x22,x23))
        
        ## concatenate all features
        samp = np.concatenate((currentSamp1,currentSamp2,currentSamp3))
        
        X_train_new[idx,:] = samp
        
        
    ####################### Validation Set ##############################
    
    
    nSamp = X_val.shape[0] # number of samples
    
    print('Creating multiple resolutions of the validation data...')
    
    ## Level 1: scale in half
    print('Creating level 1 resolution...')
    X1 = np.empty((nSamp,14*14))
    for idx in range(0,nSamp):
        X1[idx,:] = np.array(Image.fromarray(X_val[idx,:].reshape(28,28)).resize((14,14))).reshape(14*14)
        
    ## Level 2: scale level 1 in half
    print('Creating level 2 resolution...')
    X2 = np.empty((nSamp,7*7))
    for idx in range(0,nSamp):
        X2[idx,:] = np.array(Image.fromarray(X1[idx,:].reshape(14,14)).resize((7,7))).reshape(7*7)
        
    ## Level 3: scale level 2 in half
    print('Creating level 3 resolution...')
    X3 = np.empty((nSamp,3*3))
    for idx in range(0,nSamp):
        X3[idx,:] = np.array(Image.fromarray(X2[idx,:].reshape(7,7)).resize((3,3))).reshape(3*3)
        
        
    X1 = X1 - level_1_mean
    X2 = X2 - level_2_mean
    X3 = X3 - level_3_mean
    
    X1_trans = np.dot(X1, eigVec1.T)
    X2_trans = np.dot(X2, eigVec1.T)
    X3_trans = np.dot(X3, eigVec1.T)
    
    print('Creating new feature vectors for validation set...')
    nFeatures = (3*(14*14))+(3*(7*7))+(3*(3*3))
    X_train_new = np.zeros((nSamp,nFeatures))
    for idx in range(0,nSamp):
#    for idx in range(0,1):
        
        ## level 1
        x20 = X1[idx,:]
        x21 = X1_trans[idx,0]*eigVec1[0,:] + X1_trans[idx,1]*eigVec1[1,:] + X1_trans[idx,2]*eigVec1[2,:]
        x22 = X1_trans[idx,0]*eigVec1[0,:] + X1_trans[idx,1]*eigVec1[1,:]
        x23 = X1_trans[idx,0]*eigVec1[0,:]
        
        x20 = x20.reshape((14*14))
        x21 = x21.reshape((14*14))
        x22 = x22.reshape((14*14))
        x23 = x23.reshape((14*14))    
        
        currentSamp1 = np.concatenate((x21,x22,x23))
        
        ## level 2
        x20 = X2[idx,:]
        x21 = X2_trans[idx,0]*eigVec2[0,:] + X2_trans[idx,1]*eigVec2[1,:] + X2_trans[idx,2]*eigVec2[2,:]
        x22 = X2_trans[idx,0]*eigVec2[0,:] + X2_trans[idx,1]*eigVec2[1,:]
        x23 = X2_trans[idx,0]*eigVec2[0,:]
        
        x20 = x20.reshape((7*7))
        x21 = x21.reshape((7*7))
        x22 = x22.reshape((7*7))
        x23 = x23.reshape((7*7))
#    
        currentSamp2 = np.concatenate((x21,x22,x23))
        
        ## level 3
        x20 = X3[idx,:]
        x21 = X3_trans[idx,0]*eigVec3[0,:] + X3_trans[idx,1]*eigVec3[1,:] + X3_trans[idx,2]*eigVec3[2,:]
        x22 = X3_trans[idx,0]*eigVec3[0,:] + X3_trans[idx,1]*eigVec3[1,:]
        x23 = X3_trans[idx,0]*eigVec3[0,:]
        
        x20 = x20.reshape((3*3))
        x21 = x21.reshape((3*3))
        x22 = x22.reshape((3*3))
        x23 = x23.reshape((3*3))
#    
        currentSamp3 = np.concatenate((x21,x22,x23))
        
        ## concatenate all features
        samp = np.concatenate((currentSamp1,currentSamp2,currentSamp3))
        
        X_train_new[idx,:] = samp
        
        
#    fig, ax = plt.subplots(2,2)
#    ax[0,0].imshow(x20.reshape((14,14)))
#    ax[0,1].imshow(x21.reshape((14,14)))
#    ax[1,0].imshow(x22.reshape((14,14)))
#    ax[1,1].imshow(x23.reshape((14,14)))
    
    
    
    
     

         
         
    
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
#    np.save('MPCA_data.npy', dataSet)
#
    ###################### Convert data into torch format ############
#    X_train = torch.FloatTensor(X_train)
#    y_train = torch.LongTensor(y_train)
#    X_val = torch.FloatTensor(X_val)
#    y_val = torch.LongTensor(y_val)
#    X_test = torch.FloatTensor(X_test)
#    y_test = torch.LongTensor(y_test)


############################### Train Network ############################



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



    print('================ DONE ================')

