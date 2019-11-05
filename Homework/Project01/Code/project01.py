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

def plot_classes(X_train, y_train):

    """
    ******************************************************************
        *  Func:      plot_classes()
        *  Desc:      Plots a grid of samples (1 from each class)
        *  Inputs:    X_train, uint8 matrix of nSamples by nFeatures
        *             y_train uint8 vector of class labels (0-9)
        *  Outputs:
    ******************************************************************
    """

    fig, ax = plt.subplots(5,2)

    # "T-shirt/Top"
    idx = np.where(y_train==0)[0][0]
    ax[0,0].imshow(X_train[idx,:].reshape((28,28)), cmap='gray')
    ax[0,0].set_title('T-shirt/Top')
    ax[0,0].axis('off')

    # "Trouser"
    idx = np.where(y_train==1)[0][0]
    ax[0,1].imshow(X_train[idx,:].reshape((28,28)), cmap='gray')
    ax[0,1].set_title('Trouser')
    ax[0,1].axis('off')

    # "Pullover"
    idx = np.where(y_train==2)[0][0]
    ax[1,0].imshow(X_train[idx,:].reshape((28,28)), cmap='gray')
    ax[1,0].set_title('Pullover')
    ax[1,0].axis('off')

    # "Dress"
    idx = np.where(y_train==3)[0][0]
    ax[1,1].imshow(X_train[idx,:].reshape((28,28)), cmap='gray')
    ax[1,1].set_title('Dress')
    ax[1,1].axis('off')

    # "Coat"
    idx = np.where(y_train==4)[0][0]
    ax[2,0].imshow(X_train[idx,:].reshape((28,28)), cmap='gray')
    ax[2,0].set_title('Coat')
    ax[2,0].axis('off')

    # "Sandal"
    idx = np.where(y_train==5)[0][0]
    ax[2,1].imshow(X_train[idx,:].reshape((28,28)), cmap='gray')
    ax[2,1].set_title('Sandal')
    ax[2,1].axis('off')

    # "Shirt"
    idx = np.where(y_train==6)[0][0]
    ax[3,0].imshow(X_train[idx,:].reshape((28,28)), cmap='gray')
    ax[3,0].set_title('Shirt')
    ax[3,0].axis('off')

    # "Sneaker"
    idx = np.where(y_train==7)[0][0]
    ax[3,1].imshow(X_train[idx,:].reshape((28,28)), cmap='gray')
    ax[3,1].set_title('Sneaker')
    ax[3,1].axis('off')

    # "Bag"
    idx = np.where(y_train==8)[0][0]
    ax[4,0].imshow(X_train[idx,:].reshape((28,28)), cmap='gray')
    ax[4,0].set_title('Bag')
    ax[4,0].axis('off')

    # "Ankle Boot"
    idx = np.where(y_train==9)[0][0]
    ax[4,1].imshow(X_train[idx,:].reshape((28,28)), cmap='gray')
    ax[4,1].set_title('Ankle Boot')
    ax[4,1].axis('off')


    plt.tight_layout()
    plt.suptitle("Fashion-MNIST Class Examples")

    return


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

###################### Define Autoencoder Class #######################
class Autoencoder100(torch.nn.Module):

    def __init__(self):
        super(Autoencoder100, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 400),
            nn.ReLU(True), nn.Linear(400, 150), nn.ReLU(True), nn.Linear(150, 100))
        self.decoder = nn.Sequential(
            nn.Linear(100, 150),
            nn.ReLU(True),
            nn.Linear(150, 400),
            nn.ReLU(True),
            nn.Linear(400, 500),
            nn.ReLU(True), nn.Linear(500, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder400(torch.nn.Module):

    def __init__(self):
        super(Autoencoder400, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 400))
        self.decoder = nn.Sequential(
            nn.Linear(400, 500),
            nn.ReLU(True), nn.Linear(500, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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
    ######################## Load Data ###############################
    dataSet = np.load('reduced_dim_data/data_pca_100.npy').item()

    X_train = dataSet["X_train"]
    y_train = dataSet["y_train"]
    X_val = dataSet["X_val"]
    y_val = dataSet["y_val"]
    X_test = dataSet["X_test"]
    y_test = dataSet["y_test"]

    plot_dist_distributions(X_train,y_train, parameters)
##
##    ######################## Visualize Reduction ######################
#    fig, ax = plt.subplots(1, figsize=(14, 10))
#    plt.scatter(X_train[:,0],X_train[:,1], s=1, c=y_train, cmap='Spectral', alpha=1.0)
#    plt.setp(ax, xticks=[], yticks=[])
#    cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
#    cbar.set_ticks(np.arange(10))
#    cbar.set_ticklabels(parameters["classes"])
#    plt.title('Fashion MNIST Embedded via UMAP')

#    ###################### Convert data into torch format ############
<<<<<<< HEAD
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)



    params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}
    training_set = my_data(X_train,y_train)
    training_generator = data.DataLoader(training_set, **params)


    ############# run a number of trials, save best model ############
    for trial in range(parameters["numTrials"]):

        learningCurve = []
        valLearningCurve = []

        ####################### Define Network ###########################
        inputSize = X_train.shape[1]

        # instantiate model
        model = Feedforward(inputSize,  parameters["outputSize"])

        # define loss function
        criterion = torch.nn.CrossEntropyLoss()

        # define optimizer (stochastic gradient descent)
        optimizer = torch.optim.Adamax(model.parameters(), parameters["learningRate"])

        ##################### Train the Network ##########################

        model.train()

        ################# train a single network #####################
        for epoch in range(parameters["numEpochs"]):

            #set gradients to zero
            optimizer.zero_grad()

            for idx, batch in enumerate(training_generator):
                X_train = batch["image"]
                y_train = batch["label"]

                # forward pass
                y_pred = model(X_train) # predict output vector

                # compute loss
                loss = criterion(y_pred, y_train)
                
                print('Epoch: {} idx: {} train loss: {}'.format(epoch, idx, loss.item()))

                # backward pass
                loss.backward() # computes the gradients
                optimizer.step() # updates the weights

            if not(epoch %  parameters["updateIter"]):
                learningCurve.append(loss)
                model.eval()
                valLearningCurve.append(criterion(model(X_val),y_val))
                model.train()

                # if gradient of validation goes positive, stop training
                if ((epoch > 20) and np.sign(valLearningCurve[-1].detach().numpy() - valLearningCurve[-2].detach().numpy())):
                    break

            if not(epoch % 2):
                print('Epoch {}: train loss: {}'.format(epoch, loss.item()))



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
=======
#    X_train = torch.FloatTensor(X_train)
#    y_train = torch.LongTensor(y_train)
#    X_val = torch.FloatTensor(X_val)
#    y_val = torch.LongTensor(y_val)
#    X_test = torch.FloatTensor(X_test)
#    y_test = torch.LongTensor(y_test)

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
>>>>>>> a171e7a911ca52e6194b85e50f07779f0add1055

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

