# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:43:06 2019

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  Visualizations.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-29
    *  Desc:  Provides function definitions for visualization tools.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import confusion_matrix


######################################################################
##################### Function Definitions ###########################
######################################################################

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

