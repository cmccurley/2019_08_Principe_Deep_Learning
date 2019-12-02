# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:52:03 2019

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  MyMNISTData.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-10-26
    *  Desc:  Provides data reader and data loader for Fashion MNIST.
**********************************************************************
"""
######################################################################
######################### Import Packages ############################
######################################################################
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.model_selection as ms
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from setParams import setParams
from torch.utils import data
from torch.autograd import Variable
from confusion_mat import plot_confusion_matrix
import numpy.random
import math
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split


######################################################################
##################### Function Definitions ###########################
######################################################################

def readData(dataFilePath, window_size, plot_raw, plot_range, add_noise):
    """
    ******************************************************************
        *  Func:      readData()
        *  Desc:      Reads data from a .txt file
        *  Inputs:    
        *             dataFilePath - path to .txt data file
        *             window_size - length of FIR filter
        *  Outputs:   
        *             X: matrix of input data, split into windows
        *             y: vector of values one-step ahead of filter
    ******************************************************************
    """
    
    ## Load data
    x = np.loadtxt(dataFilePath)
    x = x - np.mean(x)
    num_samples = len(x)
    
    ## Plot time series in a specific range
    if plot_raw:
        n_samples_plot = np.arange(plot_range[0],plot_range[1])
        plt.figure()
        plt.plot(n_samples_plot,x[n_samples_plot])
        plt.title(f'Mackey-Glass Time Series \n Range: {plot_range[0]} - {plot_range[1]}', fontsize=14)
    
    ## Partition data into windows
    X = np.zeros((num_samples - window_size - 1, window_size))
    for idx in range(0,num_samples - window_size- 1):
        X[idx,:] = x[idx:(idx+window_size)]
        
    ## Generate labels as one step ahead predition value
    
    ## Add random noise
    if add_noise:
        noise = (0.95*numpy.random.normal(loc=0.0, scale=0.5)) + (0.05*numpy.random.normal(loc=1,scale=0.5))
        y = x + noise
        
    y = x[(window_size):]
    
    return X,y



class MGdata(Dataset):
    """
    ******************************************************************
        *  Func:      MGdata()
        *  Desc:      Dataloader for the MG dataset.
        *  Inputs:    
        *             dataFilePath - path to .txt data file
        *             window_size - length of FIR filter
        *             Noise - boolean to add noise to desired
        *  Outputs:   
        *             sequence: matrix of input data, split into windows
        *             label: list of values one-step ahead of filter
        *             index : location of samples in the original data matrix
    ******************************************************************
    """
    def __init__(self, dataFilePath, window = 20, Noise = False):
        
        self.targets = []
        self.targets_no_noise =[]
        
        ## Load the data
        input_data = np.loadtxt(dataFilePath)
        input_data = input_data - np.mean(input_data)
        
        ## Partition data into windows
        self.inout_seq = []
        L = len(input_data)
        for i in range(L-window):
            train_seq = input_data[i:i+window]
            train_label = input_data[i+window:i+window+1].copy()
            train_label_no_noise = input_data[i+window:i+window+1].copy()
            #Add noise using Middleton model if desired
            if (Noise):
                train_label += (.95*np.random.normal(loc=0,scale=np.sqrt(.5)) +
                                .05*np.random.normal(loc=1,scale=np.sqrt(.5)))
            self.inout_seq.append({  # sequence and desired
                    "sequence": train_seq,
                    "label": train_label,
                    "label_no_noise": train_label_no_noise
                })
            self.targets.append(train_label)
            self.targets_no_noise.append(train_label_no_noise)
            
            self.Noise = Noise


    def __len__(self):
        return len(self.inout_seq)

    def __getitem__(self, index):

        datafiles = self.inout_seq[index]

        sequence = torch.FloatTensor(datafiles["sequence"])

        label_file = datafiles["label"]
        label = torch.FloatTensor(label_file)
        
        label_no_noise = torch.FloatTensor(datafiles["label_no_noise"])
    

        if self.Noise:  
            label_no_noise = torch.FloatTensor(datafiles["label_no_noise"])
            return sequence, label, index, label_no_noise
        else:
            return sequence, label, index, label_no_noise