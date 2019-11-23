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
    *  Date:  2019-11-23
    *  Desc:  Sets parameters for hw05.py
**********************************************************************
"""

import os
import numpy as np

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

    ################### Data Generation Parameters ###########################
    parameters["dataPath"] = 'MGdata.txt'
    parameters["window_size"] = 6
    parameters["plot_raw"] = True
    parameters["plot_range"] = np.array([[200],[400]])
    parameters["add_noise"] = False

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