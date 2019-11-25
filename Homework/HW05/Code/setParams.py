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
    parameters["gen_data"] = False
    parameters["dataPath"] = 'MGdata.txt'
    parameters["window_size"] = 20
    parameters["plot_raw"] = False
    parameters["plot_range"] = np.array([[200],[400]])
    parameters["add_noise"] = False
    parameters["saved_data_name"] = 'MGdata_split.npy'

    ################### Data Parameters #############################
    data_parameters = dict()
    data_parameters["validationSize"] = 0.15
    parameters["data_parameters"] = data_parameters

    ################### Network Parameters ##########################
    parameters["loss_type"] = 'mse'
    parameters["outputSize"] = 1
    parameters["learningRate"] = 0.01
    parameters["numEpochs"] = 10
    parameters["numTrials"] = 1
    parameters["updateIter"] = 10
    
    ####################### MEE Parameters ##########################
    parameters["mee_kernel_width"] = 0.001
    parameters["batch_size"] = 200

    return parameters