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

    ################### Data Parameters #############################
    parameters["validationSize"] = 0.2
    parameters["random_state"] = 24

    ################### Autoencoder Parameters ######################
    ae_parameters = dict()
    
    ae_parameters["ae_latent_size"] = 10
    ae_parameters["learning_rate"] = 0.01
    ae_parameters["numEpochs"] = 800
    ae_parameters["numTrials"] = 1
    ae_parameters["updateIter"] = 10
    
    parameters["ae_parameters"] = ae_parameters
    
    
    parameters["outputSize"] = 10
    ####################### MEE Parameters ##########################
    parameters["mee_kernel_width"] = 1.2
    parameters["batch_size"] = 200

    return parameters