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
    data_parameters = dict()
    data_parameters["validationSize"] = 0.15
    parameters["data_parameters"] = data_parameters

    ################### Network Parameters ##########################
    parameters["outputSize"] = 10
    parameters["learningRate"] = 0.01
    parameters["numEpochs"] = 800
    parameters["numTrials"] = 1
    parameters["updateIter"] = 10

    return parameters