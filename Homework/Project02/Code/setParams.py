#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:33:10 2019

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  setParams.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-10-29
    *  Desc:  Sets parameters for project01.py
**********************************************************************
"""

import os

def setParams():
    """
    ******************************************************************
        *  Func:      setParams()
        *  Desc:
        *  Inputs:
        *  Outputs:
    ******************************************************************
    """
    parameters = dict()

    ################### General Parameters ###########################
    parameters["dataPath"] = os.getcwd() + '/fashion_mnist_master/data/fashion'
    parameters["classes"] = ["T-shirt/Top","Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    parameters["class_num"] = [0,1,2,3,4,5,6,7,8,9]
    parameters["width"] = 28
    parameters["image_size"] = 784
    
    
    parameters["train_ae"] = False
    parameters["train_cnn"] = False

    ################### Data Parameters #############################
    parameters["validationSize"] = 0.2
    parameters["random_state"] = 24

    ################### Autoencoder Parameters ######################
    ae_parameters = dict()
    
    ae_parameters["ae_latent_size"] = 10
    ae_parameters["learning_rate"] = 0.01
    ae_parameters["numEpochs"] = 10
    ae_parameters["numTrials"] = 1
    ae_parameters["updateIter"] = 10
    ae_parameters["val_update"] = 100
    ae_parameters["model_save_path"] = os.getcwd() + '\\ae_model_parameters\\ae_latent_' + str(ae_parameters["ae_latent_size"]) + '.pth'
    ae_parameters["image_save_path"] = os.getcwd() + '\\ae_reconstructed_images\\ae_latent_' + str(ae_parameters["ae_latent_size"])
    
    parameters["ae_parameters"] = ae_parameters
    
    ################### Baseline CNN Parameters ######################
    cnn_parameters = dict()
    
    cnn_parameters["learning_rate"] = 0.01
    cnn_parameters["numEpochs"] = 20
    cnn_parameters["numTrials"] = 1
    cnn_parameters["updateIter"] = 10
    cnn_parameters["val_update"] = 300
    cnn_parameters["model_save_path"] = os.getcwd() + '\\cnn_model_parameters\\baseline_cnn.pth'
    cnn_parameters["image_save_path"] = os.getcwd() + '\\cnn_model_parameters\\baseline_cnn' 
    
    parameters["cnn_parameters"] = cnn_parameters
    
    ######################## SVM Parameters ###########################
    svm_parameters = dict()
    
    svm_parameters["learning_rate"] = 0.01
    svm_parameters["numEpochs"] = 20
    svm_parameters["numTrials"] = 1
    svm_parameters["updateIter"] = 10
    svm_parameters["val_update"] = 300
    svm_parameters["model_save_path"] = os.getcwd() + '\\cnn_model_parameters\\baseline_cnn.pth'
    svm_parameters["image_save_path"] = os.getcwd() + '\\cnn_model_parameters\\baseline_cnn' 
    
    parameters["cnn_parameters"] = cnn_parameters
    
    ####################### MEE Parameters ##########################
    parameters["mee_kernel_width"] = 1.2
    parameters["batch_size"] = 200

    return parameters