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
#    parameters["hiddenSize"] = 14
#    parameters["outputSize"] = 6
#    parameters["learningRate"] = 0.1
#    parameters["numEpochs"] = 10000
#    parameters["numTrials"] = 10
#    parameters["validationSize"] = 0.1
#    parameters["labels"] = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5', 'Awake']
#    parameters["updateIter"] = 200 
    
    return parameters