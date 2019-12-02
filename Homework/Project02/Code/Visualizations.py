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