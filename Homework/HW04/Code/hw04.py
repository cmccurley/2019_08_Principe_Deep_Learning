# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:39:11 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  hw04.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-10-07
    *  Desc:  Provides coded solutions to homework set 04 of EEL681,
    *         Deep Learning, taught by Dr. Jose Principe, Fall 2019.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################
import os
import copy
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.model_selection as ms
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torch.nn.utils.rnn import pack_sequence


######################################################################
##################### Function Definitions ###########################
######################################################################
 
def readData(testDataPath):
    """
    ******************************************************************
        *  Func:      readData()
        *  Desc:      Reads data from a .asc file
        *  Inputs:
        *  Outputs: 
    ******************************************************************
    """
    grammar_df = pd.read_csv(testDataPath)
    grammar_mat = grammar_df.to_numpy()
    X_test = grammar_mat[:,:-1]
    y_test = grammar_mat[:,-1]
    
    # convert zeros to -1
    for ind in range(0, X_test.shape[0]):
        X_test[ind,np.where(X_test[ind,:]==0)] =  -1
        
#    y_test[np.where(y_test==0)] = -1
    
    return X_test, y_test

def generateData():
    """
    ******************************************************************
        *  Func:      generateData()
        *  Desc:      Creates training corpus following grammar
        *             outlined in hw03.
        *  Inputs:
        *  Outputs: 
    ******************************************************************
    """
    # generate in-sample and out-of-sample data
    in_grammar_data, in_grammar_labels = genInGrammar(500)
    out_grammar_data, out_grammar_labels = genOutGrammar(500)
    
    # randomly shuffle data
    idx = np.random.permutation(2*len(in_grammar_labels))
    X_all = in_grammar_data + out_grammar_data
    y_all = np.concatenate((in_grammar_labels, out_grammar_labels))
    
    # ap
    y = y_all[idx]
    
    X = list()
    for i in idx:
        X.append(X_all[i])
        
    return X, y

def genInGrammar(nSamples):
        
    """
    ******************************************************************
        *  Func:      genInGrammar()
        *  Desc:      
        *  Inputs:
        *  Outputs: 
    ******************************************************************
    """
    # initialize data structure
    in_grammar_data = list()
    
    # generate labels
    in_grammar_labels = np.ones(nSamples)
    
    sampInd = 0
    while (sampInd < nSamples):
        
        k =  random.randint(2,61) # generate random length of vector
        
        samp = np.zeros(k)
        samp[0] =  np.random.randint(2, size=1)
        prevSamp = samp[0] # initialize previous sample
        
        
        # fill binary sample following grammar
        for ind in range(1,k):
            
            samp[ind] = np.random.randint(2, size=1)
            
            # make sure new sample is not a one if previous sample was a one
            if (samp[ind] and prevSamp):
                samp[ind] = 0
            
             
            prevSamp = samp[ind]
            
        # verify both digits are in the grammar
        if (len(np.unique(samp))>1):
            
            # convert zeros to -1
            samp[np.where(samp==0)] = -1
            
            # add sample to sample list
            in_grammar_data.append(samp)
            
            # iterate sample number
            sampInd = sampInd + 1
            
    return in_grammar_data, in_grammar_labels

def genOutGrammar(nSamples):
        
    """
    ******************************************************************
        *  Func:      genOutGrammar()
        *  Desc:      
        *  Inputs:
        *  Outputs: 
    ******************************************************************
    """
    
    # generate labels
#    out_grammar_labels = (-1)*np.ones(nSamples)
    out_grammar_labels = np.zeros(nSamples)
    
    # initialize data structure
    out_grammar_data = list()
    
    sampInd = 0
    while (sampInd < nSamples):
        
        validPoint = False
        
        k =  random.randint(2,61) # generate random length of vector
        
        samp = np.zeros(k)
        samp[0] =  np.random.randint(2, size=1)
        prevSamp = samp[0] # initialize previous sample
        
        
        # fill binary sample following grammar
        for ind in range(1,k):
            
            samp[ind] = np.random.randint(2, size=1)
            
        # verify both digits are in the grammar
        if (len(np.unique(samp))>1):
            prevSamp = samp[0]
            for n in range(1, len(samp)):
                if (prevSamp and samp[n]):
                    validPoint = True
            
            if validPoint == True:
                
                # convert zeros to -1
                samp[np.where(samp==0)] = -1
                
                # add sample to sample list
                out_grammar_data.append(samp)
                
                # iterate sample number
                sampInd = sampInd + 1
    
    
    return out_grammar_data, out_grammar_labels

def padData(X, length):
        
    """
    ******************************************************************
        *  Func:      padData()
        *  Desc:      Zero pad data to match specific length 
        *  Inputs:    X - list of variable-length training samples
        *  Outputs:   X_padded - nSamplesxlength matrix
    ******************************************************************
    """
    
    X_padded = np.zeros((len(X),length))
    
    for ind in range(0,len(X)):
        samp = X[ind]
        lengthSamp = len(samp)
        X_padded[ind,0:lengthSamp] = samp
    
    return X_padded
 
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

###################### Define Neural Net Classes #####################
    
class TDNN(torch.nn.Module):
    
    def __init__(self, window_size, sequence_length, hidden_size, output_size):
        super(TDNN, self).__init__()
        
        # Define net parameters 
        self.window_size = window_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        
        # define net layers
        self.input_filter = nn.Conv1d(1,1,self.window_size, bias=True)
        self.fc1 = nn.Linear(self.sequence_length - self.window_size + 1, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x):
        convFeat = self.input_filter(x)
        hidden1 = self.fc1(convFeat)
        act1 = self.tanh(hidden1)
        hidden2 = self.fc2(act1)
        act2 = self.sigmoid(hidden2)
        output = act2
        
        return output

class RNN(nn.Module):

    def __init__(self, sequence_len, num_outputs, input_size, hidden_size, num_layers):
        super().__init__()
        # Define net parameters 
        self.seq_len = sequence_len
        self.num_layers = num_layers
        self.input_size = input_size
        self.outputs = num_outputs
        self.hidden_size = hidden_size
        
        # define net layers
        self.rnn = nn.RNN(input_size, hidden_size, batch_first = True)
        self.linear = nn.Linear(self.seq_len*self.hidden_size, self.outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self._init_hidden(batch_size)
        x = x.view(batch_size, self.seq_len, self.input_size)

        # [batch_size, seq_len, hidden_size]
        rnn_out, _ = self.rnn(x, hidden)
        linear_out = self.sigmoid(torch.flatten(self.linear(torch.flatten(rnn_out,1))))
        return linear_out

    def _init_hidden(self, batch_size):
        
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

######################################################################
############################## Main ##################################
######################################################################
if __name__== "__main__":
    
    print('Running Main...')
    
    ####################### Set Parameters ###########################
    parameters = dict()
    
    parameters["currentModel"] = 'rnn'
    
    parameters["learningRate"] = 0.001
    parameters["numEpochs"] = 10000
    parameters["numTrials"] = 10
    parameters["validationSize"] = 0.1
    parameters["labels"] = ['Out of Grammar', 'In Grammar']
    parameters["updateIter"] = 200 
    
    # Network parameters
    parameters["testDataLength"] = 80
    
    # TDNN7
    parameters["windowSize1"] = 7
    parameters["tdnn1_hidden_size"] = 5
    
    # TDNN20
    parameters["windowSize2"] = 20 
    parameters["tdnn2_hidden_size"] = 10

    # RNN
    parameters["rnn_hidden_size"] = 2
    parameters["rnn_input_dim"] = 1
    parameters["rnn_num_layers"] = 1
    
    
    ####################### Import data ##############################
    print('Loading test data...')
    testDataPath = os.getcwd()+'\\Data\\hmw3test.csv'
    X_test_all, y_test_all = readData(testDataPath)
    
    ####################### Generate data ############################
    print('Generating train data...')
    X_train_all, y_train_all = generateData()
    
    ######################## Pad data for TDNN #######################
    
    X_train_all = padData(X_train_all, parameters["testDataLength"])

    ####################### Define Training Subset ###################

    # partition data into training and validation    
    X_train, X_val, y_train, y_val = ms.train_test_split(X_train_all, y_train_all, test_size=parameters["validationSize"], random_state=42)
    
    # convert data into torch format
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    X_train = X_train.unsqueeze(dim=1)
    X_val = X_val.unsqueeze(dim=1)
  
    ####################### Define Testing Subset ####################
    
    # test on p2
    X_test = torch.FloatTensor(X_test_all)
    y_test = torch.FloatTensor(y_test_all)
    
    X_test = X_test.unsqueeze(dim=1)
    
    
    if (parameters["currentModel"] == 'tdnn7'):
        ####################################################################
        ############################ TDNN 1 ################################
        ####################################################################
        
        print('Working with TDNN7...')
        
        ######################## Define TDNN1 ##############################
       
       # get input/ output shapes
        inputSize = X_train.shape[2]
        outputSize = 1
            
        # instantiate model
        model = TDNN(parameters["windowSize1"], inputSize, parameters["tdnn1_hidden_size"], outputSize)
        
        # define loss function
        criterion = torch.nn.BCELoss()
        
        # define optimizer (stochastic gradient descent)
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learningRate"])
           
        
        ############# run a number of trials, save best model ############
        for trial in range(parameters["numTrials"]):
            
            learningCurve = []
            valLearningCurve = []
        
            ##################### Train the Network ##########################
        
            model.train()
        
            ################# train a single network #####################
            for epoch in range(parameters["numEpochs"]):
                
                #set gradients to zero
                optimizer.zero_grad()
                
                # forward pass
                y_pred = model(X_train) # predict output vector
                y_pred = y_pred.flatten()
                
                # compute loss
                loss = criterion(y_pred, y_train)
                
                if not(epoch %  parameters["updateIter"]):
                    learningCurve.append(loss)
                    model.eval()
                    valLearningCurve.append(criterion(model(X_val),y_val))
                    model.train()
                    
                    # if gradient of validation goes positive, stop training
                    if ((epoch > 1000) and np.sign(valLearningCurve[-1].detach().numpy() - valLearningCurve[-2].detach().numpy())):
                        break
                
                if not(epoch % 500):
                    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
                
                # backward pass
                loss.backward() # computes the gradients
                optimizer.step() # updates the weights
    
            if (trial==0):
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
                    
           
        ######################## Learning Curve ##########################
        
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
    ##    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
    ##    plt.close()
        
       
        
        ####################### Confusion Matrix #########################
        
        # revert model back to best performing
        model.load_state_dict(best_model["modelParameters"])
        model.eval()
        
        # predict state labels
        y_test_pred = model(X_test)
        y_test_pred = y_test_pred.flatten()
        
        y_test_pred[y_test_pred>0.5] = 1
        y_test_pred[y_test_pred<0.5] = 0
        
        # compute the loss
        testLoss = criterion(y_test_pred, y_test)
        
        testLoss = testLoss.detach().numpy()
        testLoss = np.round(testLoss,2)
        
        # plot the confusion matrix
        plot_confusion_matrix(y_test.detach().numpy(), y_test_pred.detach().numpy(), parameters["labels"], testLoss, normalize=False, title='TDNN 7 - Confusion Matrix')

    
        print('================ DONE ================')
    
    elif (parameters["currentModel"] == 'tdnn20'):
        ####################################################################
        ############################ TDNN 20 ###############################
        ####################################################################
        print('Working with TDNN20...')
        
        
        ######################## Define TDNN2 ##############################
       
       # get input/ output shapes
        inputSize = X_train.shape[2]
        outputSize = 1
            
        # instantiate model
        model = TDNN(parameters["windowSize2"], inputSize, parameters["tdnn2_hidden_size"], outputSize)
        
        # define loss function
        criterion = torch.nn.BCELoss()
        
        # define optimizer (stochastic gradient descent)
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learningRate"])
           
        
        ############# run a number of trials, save best model ############
        for trial in range(parameters["numTrials"]):
            
            learningCurve = []
            valLearningCurve = []
        
            ##################### Train the Network ##########################
        
            model.train()
        
            ################# train a single network #####################
            for epoch in range(parameters["numEpochs"]):
                
                #set gradients to zero
                optimizer.zero_grad()
                
                # forward pass
                y_pred = model(X_train) # predict output vector
                y_pred = y_pred.flatten()
                
                # compute loss
                loss = criterion(y_pred, y_train)
                
                if not(epoch %  parameters["updateIter"]):
                    learningCurve.append(loss)
                    model.eval()
                    valLearningCurve.append(criterion(model(X_val),y_val))
                    model.train()
                    
                    # if gradient of validation goes positive, stop training
                    if ((epoch > 1000) and np.sign(valLearningCurve[-1].detach().numpy() - valLearningCurve[-2].detach().numpy())):
                        break
                
                if not(epoch % 500):
                    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
                
                # backward pass
                loss.backward() # computes the gradients
                optimizer.step() # updates the weights
    
            if (trial==0):
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
                    
           
        ######################## Learning Curve ##########################
        
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
    ##    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
    ##    plt.close()
        
       
        
        ####################### Confusion Matrix #########################
        
        # revert model back to best performing
        model.load_state_dict(best_model["modelParameters"])
        model.eval()
        
        # predict state labels
        y_test_pred = model(X_test)
        y_test_pred = y_test_pred.flatten()
        
        y_test_pred[y_test_pred>0.5] = 1
        y_test_pred[y_test_pred<0.5] = 0
        
        # compute the loss
        testLoss = criterion(y_test_pred, y_test)
        
        testLoss = testLoss.detach().numpy()
        testLoss = np.round(testLoss,2)
        
        # plot the confusion matrix
        plot_confusion_matrix(y_test.detach().numpy(), y_test_pred.detach().numpy(), parameters["labels"], testLoss, normalize=False, title='TDNN 20 - Confusion Matrix')
    
        
        print('================ DONE ================')
    
    elif (parameters["currentModel"] == 'rnn'):
        ####################################################################
        ############################## RNN #################################
        ####################################################################
        
        print('Working with RNN...')
        
        ######################## Define RNN ##############################
       
       # get input/ output shapes
        inputSize = X_train.shape[2]
        outputSize = 1
        
        # instantiate model
        model = RNN(inputSize, outputSize, parameters["rnn_input_dim"], parameters["rnn_hidden_size"], parameters["rnn_num_layers"])
        
        
        # define loss function
        criterion = torch.nn.BCELoss()
        
        # define optimizer (stochastic gradient descent)
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learningRate"])
        
        ############# run a number of trials, save best model ############
        for trial in range(parameters["numTrials"]):
            
            learningCurve = []
            valLearningCurve = []
        
            ##################### Train the Network ##########################
        
            model.train()
        
            ################# train a single network #####################
            for epoch in range(parameters["numEpochs"]):
                
                #set gradients to zero
                optimizer.zero_grad()
                
                # forward pass
                y_pred = model(X_train) # predict output vector
                y_pred = y_pred.flatten()
                
                # compute loss
                loss = criterion(y_pred, y_train)
                
                if not(epoch %  parameters["updateIter"]):
                    learningCurve.append(loss)
                    model.eval()
                    valLearningCurve.append(criterion(model(X_val),y_val))
                    model.train()
                    
                    # if gradient of validation goes positive, stop training
                    if ((epoch > 1000) and np.sign(valLearningCurve[-1].detach().numpy() - valLearningCurve[-2].detach().numpy())):
                        break
                
                if not(epoch % 500):
                    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
                
                # backward pass
                loss.backward() # computes the gradients
                optimizer.step() # updates the weights
    
            if (trial==0):
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
                    
           
        ######################## Learning Curve ##########################
        
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
    ##    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
    ##    plt.close()
        
       
        
        ####################### Confusion Matrix #########################
        
        # revert model back to best performing
        model.load_state_dict(best_model["modelParameters"])
        model.eval()
        
        # predict state labels
        y_test_pred = model(X_test)
        y_test_pred = y_test_pred.flatten()
        
        y_test_pred[y_test_pred>0.5] = 1
        y_test_pred[y_test_pred<0.5] = 0
        
        # compute the loss
        testLoss = criterion(y_test_pred, y_test)
        
        testLoss = testLoss.detach().numpy()
        testLoss = np.round(testLoss,2)
        
        # plot the confusion matrix
        plot_confusion_matrix(y_test.detach().numpy(), y_test_pred.detach().numpy(), parameters["labels"], testLoss, normalize=False, title='TDNN - 7 Normalized Confusion Matrix')
    
        
        print('================ DONE ================')


