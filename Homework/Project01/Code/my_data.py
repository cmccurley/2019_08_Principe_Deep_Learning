#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:15:24 2019

@author: cmccurley
"""

from torch.utils.data import Dataset

class my_data(Dataset):
    def __init__(self, X, y, img_transform = None, label_transform = None):  #root_dir is a list

        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):

        img = self.X[index,:]
        label = self.y[index]
        label = int(label)

        sample = {'image': img, 'label': label}

        return sample
