# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:51:31 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  BaselineCNN.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-29
    *  Desc:  Provides class definition for baseline CNN.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable

######################################################################
##################### Function Definitions ###########################
######################################################################

class baselineCNN(nn.Module):
    """
    ******************************************************************
        *  Func:      baselineCNN()
        *  Desc:      Baseline CNN architecture developed by Michael
        *             Li. Accessed: 2019-12-03. 
        *             Available: https://towardsdatascience.com/
        *             build-a-fashion-mnist-cnn-pytorch-style-efb297e22582
    ******************************************************************
    """
    def __init__(self):
        super().__init__()
        
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
    
      # define forward function
    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # fc1
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        
        # fc2
        t = self.fc2(t)
        t = F.relu(t)
        
        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.
    
        return t