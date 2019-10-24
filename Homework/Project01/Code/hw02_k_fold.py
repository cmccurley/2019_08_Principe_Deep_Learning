# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:39:11 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  hw02_k_fold.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-09-21
    *  Desc:  Provides coded solutions to homework set 02 of EEL681,
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
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



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
        print(f'Confusion matrix \n Test Loss: {totalLoss}')
    
    print(cm)
    
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

###################### Define Neural Net Class #######################
class Feedforward(torch.nn.Module):
    
        def __init__(self, input_size, hidden_size, output_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
            self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            
            
            self.hidden_1_size = 5
            self.hidden_2_size = 10
            self.fc3 = torch.nn.Linear(self.input_size, self.hidden_1_size)
            self.fc4 = torch.nn.Linear(self.hidden_1_size, self.hidden_2_size)
            self.fc5 = torch.nn.Linear(self.hidden_2_size, self.output_size)
#            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, x):
            hidden = self.fc3(x)
            relu = self.relu(hidden)
            h2 = self.fc4(relu)
            relu = self.relu(h2)
            output = self.fc5(relu)
#            output = self.sigmoid(output)
            return output


######################################################################
############################## Main ##################################
######################################################################
if __name__== "__main__":
    
    print('Running Main...')
    
    ####################### Set Parameters ###########################
    parameters = dict()
    parameters["hiddenSize"] = 10
    parameters["outputSize"] = 6
    parameters["learningRate"] = 0.01
    parameters["numEpochs"] = 10000
    parameters["numTrials"] = 10
    parameters["validationSize"] = 0.2
    parameters["testSize"] = 0.2
    parameters["labels"] = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5', 'Awake']
    parameters["updateIter"] = 200 
    parameters["numFolds"] = 5
    
    ####################### Import data ##############################
    print('Loading data...')
    cwd = os.getcwd()
    
    # import data form patient 1
    dataFilePath = '../Data/Sleepdata1 Input.asc'
    labelFilePath = '../Data/Sleepdata1 Desired.asc'
    skipHeader = 1
    X_p1,y_p1 =  readData(dataFilePath, labelFilePath, skipHeader)
    y_p1 = np.argmax(y_p1,axis=1) # convert from one-hot to index
    
    # import data form patient 2
    dataFilePath = '../Data/Sleepdata2 Input.asc'
    labelFilePath = '../Data/Sleepdata2 Desired.asc'
    skipHeader = 1
    X_p2,y_p2 =  readData(dataFilePath, labelFilePath, skipHeader)
    y_p2 = np.argmax(y_p2,axis=1) # convert from one-hot to index
    
    # combine data from both patients
    X_all = np.concatenate((X_p1,X_p2))
    y_all = np.concatenate((y_p1,y_p2))

    ################### Define Training/Test Subset ##################

    # partition data into training and validation    
    X, X_test, y, y_test = ms.train_test_split(X_all, y_all, test_size=parameters["testSize"], random_state=42)
    
    # convert data into torch format
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    #################### Define Training Folds #######################
    kf = KFold(n_splits=parameters["numFolds"], random_state=42, shuffle=True)
    kf.get_n_splits(X)
    
    iteration = 1
    for train_index, test_index in kf.split(X):
        print(f'Fold: {iteration}')
#        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
   
        # convert data into torch format
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)
    
        ############# run a number of trials, save best model ############
        for trial in range(parameters["numTrials"]):
            
            learningCurve = []
            valLearningCurve = []
            
             ####################### Define Network ###########################
            inputSize = X_p1.shape[1]
        #    outputSize = y_p1.shape[1]
            outputSize=1
            
            # instantiate model
            model = Feedforward(inputSize, parameters["hiddenSize"],  parameters["outputSize"])
            
            # define loss function
            criterion = torch.nn.CrossEntropyLoss()
            
            # define optimizer (stochastic gradient descent)
            optimizer = torch.optim.SGD(model.parameters(), parameters["learningRate"])
        
            ##################### Train the Network ##########################
        
            model.train()
        
            ################# train a single network #####################
            for epoch in range(parameters["numEpochs"]):
                
                #set gradients to zero
                optimizer.zero_grad()
                
                # forward pass
                y_pred = model(X_train) # predict output vector
                
                # compute loss
                loss = criterion(y_pred, y_train)
                
                if not(epoch %  parameters["updateIter"]):
                    learningCurve.append(loss)
                    model.eval()
                    valLearningCurve.append(criterion(model(X_val),y_val))
                    model.train()
                    
                    # if gradient of validation goes positive, stop training
                    if ((epoch > 4000) and np.sign(valLearningCurve[-1].detach().numpy() - valLearningCurve[-2].detach().numpy())):
                        break
                
                if not(epoch % 500):
                    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
                
                # backward pass
                loss.backward() # computes the gradients
                optimizer.step() # updates the weights
    
            if ((iteration == 1) and (trial==0)):
                best_model = dict()
                best_model["modelParameters"] = copy.deepcopy(model.state_dict())
                best_model["learningCurve"] = learningCurve
                best_model["valLearningCurve"] = valLearningCurve
                best_model["numEpochs"] = epoch
                best_model["validationLoss"] = valLearningCurve[-1]
            else:
                if (valLearningCurve[-1] > best_model["validationLoss"]):
                    best_model["modelParameters"] = copy.deepcopy(model.state_dict())
                    best_model["learningCurve"] = learningCurve
                    best_model["valLearningCurve"] = valLearningCurve
                    best_model["numEpochs"] = epoch
                    best_model["validationLoss"] = valLearningCurve[-1]
        iteration = iteration + 1
       
    ######################### Learning Curve ##########################
    
    # retrieve optimal parameters
    learningCurve = best_model["learningCurve"]
    valLearningCurve = best_model["valLearningCurve"]
    
    # plot the learning curve
    plt.figure()
    plt.plot(parameters["updateIter"]*np.arange(0,len(learningCurve),1),learningCurve, c='blue')
    plt.plot(parameters["updateIter"]*np.arange(0,len(valLearningCurve),1),valLearningCurve, c='orange')
    plt.title("Learing Curve", fontsize=18)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.legend(['Training', 'Validation'])
#    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
#    plt.close()
    

    
    
    ####################### Confusion Matrix #########################
    
    # revert model back to best performing
    model.load_state_dict(best_model["modelParameters"])
    model.eval()
    
    # predict state labels
    y_test_pred = model(X_test)
    values, y_test_pred_index = y_test_pred.max(1)
    
    # compute the loss
    testLoss = criterion(y_test_pred, y_test)
    
    testLoss = testLoss.detach().numpy()
    testLoss = np.round(testLoss,2)
    
    # plot the confusion matrix
    plot_confusion_matrix(y_test.detach().numpy(), y_test_pred_index.detach().numpy(), parameters["labels"], testLoss, normalize=False, title='Normalized Confusion Matrix of Sleep \n States for P2')

    

   
    
    print('================ DONE ================')

