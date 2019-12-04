# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:35:56 2019

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  MLP.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-12-04
    *  Desc:  Provides class definition for MLP with size with
    *         variable size bottleneck layer.
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

###################### Define Neural Net Class #######################
class Feedforward(torch.nn.Module):

        def __init__(self, latent_size):
            super(Feedforward, self).__init__()
            self.latent_size = latent_size
            self.forward_pass = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 200),
            nn.ReLU(True), 
            nn.Linear(200, self.latent_size), 
            nn.ReLU(True),
            nn.Linear(self.latent_size, 10), 
            nn.ReLU(True),
            nn.Softmax(dim=1)
            )

        def forward(self, x):
            x = self.forward_pass(x)
            return x

    



