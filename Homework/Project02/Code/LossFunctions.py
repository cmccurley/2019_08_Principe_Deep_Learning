# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:52:28 2019

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  LossFunctions.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-10-26
    *  Desc:  Provides class definitions for loss functions.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################

import numpy as np
import torch

######################################################################
##################### Function Definitions ###########################
######################################################################

class MEELoss(torch.nn.Module):
    
    def __init__(self):
        super(MEELoss,self).__init__()
        
    def forward(self, y_pred, y_true, parameters):
        """
        ******************************************************************
            *  Func:      MEELoss()
            *  Desc:      Computes empirical estimate  of minimum error entropy.
            *  Inputs:    
            *             y_pred - vector of predicted labels
            *             y_true - vector of true labels
            *             parameters - dictionary of script parameters
            *                          must include mee kernel bandwidth
            *  Outputs:   
            *             mee_loss_val: scalar of estimated mee loss
        ******************************************************************
        """
        
        n_samples = y_true.size()[0]
        
        ## compute normalizing constant for Gaussian kernel
        self.gauss_norm_const = torch.empty(1, dtype=torch.float)
        self.gauss_norm_const.fill_((1/(np.sqrt(2*np.pi)*parameters["mee_kernel_width"])))
        
        ## Compute normalizing constnant on error
        self.norm_const = torch.empty(1, dtype=torch.float)
        self.norm_const.fill_((1/((y_true.size()[0])**2)))
        
        ## get vector of instantaneous errors
        self.error_vect = torch.abs(y_true.clone() - y_pred.clone())
        
        ## Update mee 
        self.error = torch.pow(self.error_vect,2).sum(dim=1,keepdim=True).expand(n_samples,n_samples)
        self.error = self.error + self.error.t()
        self.error.addmm_(1,-2,self.error_vect,self.error_vect.t())
        self.error = self.error.clamp(min=1e-12).sqrt()
        
        ## Apply kernel
        self.mee_loss = torch.sum(self.gauss_norm_const*torch.exp(-((self.error)**2)/((2*(parameters["mee_kernel_width"]**2)))))
        
        ## Normalize by number of samples squared
        self.mee_loss_val = (-1)*torch.sum(self.mee_loss)*self.norm_const*(1/parameters["mee_kernel_width"])
        
        return self.mee_loss_val