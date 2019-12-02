# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:46:24 2019

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  utils.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-29
    *  Desc:  Provides utility and helper functions.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################
import numpy as np
import torch

######################################################################
##################### Function Definitions ###########################
######################################################################

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


def predict(dataloader,model):
    """
    ******************************************************************
        *  Func:      predict()
        *  Desc:      Passes dataset through a model and returns model outputs.
        *  Inputs:    
        *             dataloader: Pytorch data loader for a dataset
        *             window_size:  Trained model which will compute output.
        *  Outputs:   
        *             GT: list of desired values for a dataset
        *             Predictions: list of model outputs for a dataset
        *             GT_no_noise: list of desired values without added noise  
    ******************************************************************
    """    

    #Initialize and accumalate ground truth and predictions
    GT = np.array(0)
    Predictions = np.array(0)
    GT_no_noise =  np.array(0)
    model.eval()
        
    ## Iterate over data.
    with torch.no_grad():
        for inputs, labels, index, labels_no_noise in dataloader:
            # forward
            outputs = model(inputs)
            
            #If validation, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            GT_no_noise = np.concatenate((GT_no_noise,labels_no_noise.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,outputs.detach().cpu().numpy()),axis=None)
            
    return GT[1:],Predictions[1:], GT_no_noise[1:]
