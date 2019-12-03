# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:48:29 2019

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  AE.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-29
    *  Desc:  Provides class definition for autoencoder with size with
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

        def __init__(self, input_size, output_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.relu = torch.nn.ReLU()
            self.fc1 = torch.nn.Linear(self.input_size, 256)
            self.fc2 = torch.nn.Linear(256,128)
            self.fc3 = torch.nn.Linear(128,100)
            self.fc4 = torch.nn.Linear(100, self.output_size)

        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            hidden = self.fc2(relu)
            relu = self.relu(hidden)
            hidden = self.fc3(relu)
            relu = self.relu(hidden)
            output = self.fc4(relu)
            return output

###################### Define Autoencoder Classes #######################
class Autoencoder(torch.nn.Module):

    def __init__(self, ae_latent_size):
        super(Autoencoder, self).__init__()
        self.ae_latent_size = ae_latent_size
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 200),
            nn.ReLU(True), 
            nn.Linear(200, self.ae_latent_size), 
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(self.ae_latent_size, 200),
            nn.ReLU(True),
            nn.Linear(200, 500),
            nn.ReLU(True),
            nn.Linear(500, 28 * 28), 
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    



