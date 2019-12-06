# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:32:51 2019

@author: Conma
"""


"""
***********************************************************************
    *  File:  trainSVM.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-30
    *  Desc:  Trains multi-class SVM using one vs one training.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################

## Default packages
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

## Custom packages
from confusion_mat import plot_confusion_matrix


######################################################################
##################### Function Definitions ###########################
######################################################################

def trainSVM(dataloaders_dict, feature_size, all_parameters):

    classes = all_parameters["classes"]
    parameters = all_parameters["svm_parameters"]

    print(f'Training SVM with feature size {feature_size}...')

    ####################### Load Data ################################
    print('Loading data...')
    data_object = np.load(parameters["data_load_path"], allow_pickle=True)
    data = data_object.item()

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    ###################### Train Multiclass SVM ######################

    print('Fitting SVM...')
    clf = svm.SVC(decision_function_shape='ovo', gamma='scale')
    clf.fit(X_train.T, y_train)

    ############## Compute accuracy on training and validation sets ######
#    print('Computing training and validation accuracy...')
#    train_acc = clf.score(X_train.T, y_train)
#    val_acc = clf.score(X_val.T, y_val)


#    print(f'Train accuracy: {train_acc}, Validation accuracy: {val_acc}')

    ############## Compute accuracy on test set ######################
    print('Computing test accuracy...')
    y_pred_test = clf.predict(X_test.T)
    acc = clf.score(X_test.T, y_test)

    ###################### Plot Confusion Matrices ###################
    plot_title = "SVM on " + str(feature_size) + "D Data"
    plot_confusion_matrix(y_test, y_pred_test, classes, acc, normalize=False, title=plot_title)

    conf_mat_save_path = parameters["image_save_path"] + str(feature_size) + '_default.png'
    plt.savefig(conf_mat_save_path)
    plt.close()
    return