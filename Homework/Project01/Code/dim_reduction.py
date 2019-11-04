#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 08:24:06 2019

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  dim_reduction.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-10-31
    *  Desc:  Sets parameters for project01.py
**********************************************************************
"""
####################### Import Packages ##############################
import numpy as np
from sklearn.decomposition import PCA
import umap
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


def pca_dim_reduction(X_train, X_val, X_test, n_dims):

    """
    ******************************************************************
        *  Func:      pca_dim_reduction()
        *  Desc:      Performs dim reduction using Principal Component
        *             Analysis
        *  Inputs:    X_train, uint8 matrix of nSamples by nFeatures
        *             y_train uint8 vector of class labels (0-9)
        *  Outputs:
    ******************************************************************
    """

    clf = PCA(n_components=n_dims)
    print('Fitting PCA...')
    clf.fit(X_train)

#    print(clf.explained_variance_ratio_)

    print(np.sum(clf.explained_variance_ratio_))

    X_train_reduced = clf.transform(X_train)
    X_val_reduced = clf.transform(X_val)
    X_test_reduced = clf.transform(X_test)



    return X_train_reduced, X_val_reduced, X_test_reduced

def umap_dim_reduction(X_train, X_val, X_test, n_dims, y_train):

    """
    ******************************************************************
        *  Func:      pca_dim_reduction()
        *  Desc:      Performs dim reduction using Principal Component
        *             Analysis
        *  Inputs:    X_train, uint8 matrix of nSamples by nFeatures
        *             y_train uint8 vector of class labels (0-9)
        *  Outputs:
    ******************************************************************
    """

    clf = umap.UMAP(n_components=n_dims)
    print('Fitting UMAP...')
    clf.fit(X_train,y_train)

    X_train_reduced = clf.transform(X_train)
    X_val_reduced = clf.transform(X_val)
    X_test_reduced = clf.transform(X_test)



    return X_train_reduced, X_val_reduced, X_test_reduced

def umap_dim_reduction(X_train, X_val, X_test, n_dims):

    """
    ******************************************************************
        *  Func:      laplacian_dim_reduction()
        *  Desc:      Performs dim reduction using Laplacian Eigenmaps
        *  Inputs:    X_train, uint8 matrix of nSamples by nFeatures
        *             y_train uint8 vector of class labels (0-9)
        *  Outputs:
    ******************************************************************
    """

    clf = umap.UMAP(n_components=n_dims)
    print('Fitting Laplacian Eigenmpas...')
    clf.fit(X_train,y_train)

    X_train_reduced = clf.transform(X_train)
    X_val_reduced = clf.transform(X_val)
    X_test_reduced = clf.transform(X_test)



    return X_train_reduced, X_val_reduced, X_test_reduced


def plot_dist_distributions(X,y, parameters):
    """
    ******************************************************************
        *  Func:      plot_dist_distributions()
        *  Desc:      Plots distributions of dissimilarities from
        *             class means
        *  Inputs:    X: uint8 matrix of nSamples by nFeatures
        *             y: uint8 vector of class labels (0-9)
        *  Outputs:
    ******************************************************************
    """

    for idx in range(0,len(np.unique(y))):
        X_in_class = np.squeeze(X[np.where(y==idx),:])
        X_out_class = np.squeeze(X[np.where(y!=idx),:])

        mean_in_class = np.mean(X_in_class, axis=0)
        mean_in_class = np.expand_dims(mean_in_class,axis=1).T
#        mean_out_class = np.mean(X_out_class, axis=0)

        dist_mat_in_class = euclidean_distances(mean_in_class,X_in_class)
        dist_mat_out_class = euclidean_distances(mean_in_class,X_out_class)

        plt.figure()
        plt.hist(dist_mat_in_class[0,:], fc=(1, 0, 0, 1), density=False)
        plt.hist(dist_mat_out_class[0,:], fc=(0, 0, 1, 0.5), density=False)

        class_label = parameters["classes"][idx]

        plt.title(f'{class_label}', fontsize=14)


        # save figure
        plot_title = 'hist_before_' + str(idx) + '.png'
        plt.savefig(plot_title)


        plt.close()




    return
