# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:20:36 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  confusion_mat.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-09
    *  Desc:  Plot confusion matrix
**********************************************************************
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes, testLoss, normalize=False, title=None, cmap=plt.cm.Blues):

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
        print(f'Confusion matrix \n Test Accuracy: {testLoss}')

    print(cm)

    tit = title + ' \n Test Accuracy: %.2f' % testLoss
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=tit,
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