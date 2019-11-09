# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:19:24 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  mpca.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-11-09
    *  Desc:  Create new feature representations using PCA-M.
**********************************************************************
"""
import numpy as np
from sklearn.decomposition import PCA
from  PIL import Image
import matplotlib.pyplot as plt

def mpca(X_train, X_val, X_test):
    """
    ******************************************************************
        *  Func:      mpca()
        *  Desc:      Create new feature representations using PCA-M.
        *  Inputs:    nSamples x nFeatures training, validation, and test sets
        *  Outputs:   Multiscale feature vectors for each data set
    ******************************************************************
    """
    ####################### Training Set ##############################
    
    ## Create multiple resolutions of the images
    nSamp = X_train.shape[0] # number of samples
    
    print('Creating multiple resolutions of the training data...')
    
    ## Level 1: scale in half
    print('Creating level 1 resolution...')
    X1 = np.empty((nSamp,14*14))
    for idx in range(0,nSamp):
        X1[idx,:] = np.array(Image.fromarray(X_train[idx,:].reshape(28,28)).resize((14,14))).reshape(14*14)
        
    ## Level 2: scale level 1 in half
    print('Creating level 2 resolution...')
    X2 = np.empty((nSamp,7*7))
    for idx in range(0,nSamp):
        X2[idx,:] = np.array(Image.fromarray(X1[idx,:].reshape(14,14)).resize((7,7))).reshape(7*7)
        
    ## Level 3: scale level 2 in half
    print('Creating level 3 resolution...')
    X3 = np.empty((nSamp,3*3))
    for idx in range(0,nSamp):
        X3[idx,:] = np.array(Image.fromarray(X2[idx,:].reshape(7,7)).resize((3,3))).reshape(3*3)
    
    ## Apply PCA to training data
    print('Applying PCA to training data...')
    
    ## Subtract means
    level_1_mean = np.mean(X1,axis=0)
    level_2_mean = np.mean(X2,axis=0)
    level_3_mean = np.mean(X3,axis=0)
    
    X1 = X1 - level_1_mean
    X2 = X2 - level_2_mean
    X3 = X3 - level_3_mean
    
    pca1 = PCA()
    pca1.fit(X1)
    eigVec1 = pca1.components_
    eigVal1 = pca1.explained_variance_
    
    X1_trans = np.dot(X1, eigVec1.T)
    
    pca2 = PCA()
    pca2.fit(X2)
    eigVec2 = pca2.components_
    eigVal2 = pca2.explained_variance_
    
    X2_trans = np.dot(X2, eigVec2.T)
    
    pca3 = PCA()
    pca3.fit(X3)
    eigVec3 = pca3.components_
    eigVal3 = pca3.explained_variance_
    
    X3_trans = np.dot(X3, eigVec3.T)
    
    
    print('Creating new feature vectors for training set...')
    nFeatures = (4*(14*14))+(4*(7*7))+(4*(3*3))
    X_train_new = np.zeros((nSamp,nFeatures))
    for idx in range(0,nSamp):
#    for idx in range(0,1):
        
        ## level 1
        x20 = X1[idx,:]
        x21 = X1_trans[idx,0]*eigVec1[0,:]
        x22 = X1_trans[idx,1]*eigVec1[1,:]
        x23 = X1_trans[idx,2]*eigVec1[2,:]
        
        x000 = x20.reshape((14,14))
        x001 = x21.reshape((14,14))
        x002 = x22.reshape((14,14))
        x003 = x23.reshape((14,14))
        
        x20 = x20.reshape((14*14))
        x21 = x21.reshape((14*14))
        x22 = x22.reshape((14*14))
        x23 = x23.reshape((14*14))    
        
        
        currentSamp1 = np.concatenate((x20,x21,x22,x23))
        
        ## level 2
        x20 = X2[idx,:]
        x21 = X2_trans[idx,0]*eigVec2[0,:]
        x22 = X2_trans[idx,1]*eigVec2[1,:]
        x23 = X2_trans[idx,2]*eigVec2[2,:]
        
        x010 = x20.reshape((7,7))
        x011 = x21.reshape((7,7))
        x012 = x22.reshape((7,7))
        x013 = x23.reshape((7,7))
        
        x20 = x20.reshape((7*7))
        x21 = x21.reshape((7*7))
        x22 = x22.reshape((7*7))
        x23 = x23.reshape((7*7))
#    
        currentSamp2 = np.concatenate((x20,x21,x22,x23))
        
        ## level 3
        x20 = X3[idx,:]
        x21 = X3_trans[idx,0]*eigVec3[0,:]
        x22 = X3_trans[idx,1]*eigVec3[1,:]
        x23 = X3_trans[idx,2]*eigVec3[2,:]
        
        x020 = x20.reshape((3,3))
        x021 = x21.reshape((3,3))
        x022 = x22.reshape((3,3))
        x023 = x23.reshape((3,3))
        
        x20 = x20.reshape((3*3))
        x21 = x21.reshape((3*3))
        x22 = x22.reshape((3*3))
        x23 = x23.reshape((3*3))
        
#    
        currentSamp3 = np.concatenate((x20,x21,x22,x23))
        
        ## concatenate all features
        samp = np.concatenate((currentSamp1,currentSamp2,currentSamp3))
        
        X_train_new[idx,:] = samp
        
        
        if not(idx%10000):
            plt.figure()
            plt.imshow(X_train[idx,:].reshape((28,28)),cmap='gray')
            plt.title(f'Example {idx}', fontsize=16)
            
            savePath = 'Example_' + str(idx) + '_full_res.png'
            plt.savefig(savePath)
            
            fig, ax = plt.subplots(3,4)
            ax[0,0].imshow(x000, cmap='gray')
            ax[0,1].imshow(x001, cmap='gray')
            ax[0,2].imshow(x002, cmap='gray')
            ax[0,3].imshow(x003, cmap='gray')
            
            ax[1,0].imshow(x010, cmap='gray')
            ax[1,1].imshow(x011, cmap='gray')
            ax[1,2].imshow(x012, cmap='gray')
            ax[1,3].imshow(x013, cmap='gray')
            
            ax[2,0].imshow(x020, cmap='gray')
            ax[2,1].imshow(x021, cmap='gray')
            ax[2,2].imshow(x022, cmap='gray')
            ax[2,3].imshow(x023, cmap='gray')
            
            savePath = 'Example_' + str(idx) + '_multi_res.png'
            plt.savefig(savePath)
            
#            fig.suptitle(f'Example {idx}', fontsize=16)
            
            fig.tight_layout()
            
            
        
    ####################### Validation Set ##############################
    
    
    nSamp = X_val.shape[0] # number of samples
    
    print('Creating multiple resolutions of the validation data...')
    
    ## Level 1: scale in half
    print('Creating level 1 resolution...')
    X1 = np.empty((nSamp,14*14))
    for idx in range(0,nSamp):
        X1[idx,:] = np.array(Image.fromarray(X_val[idx,:].reshape(28,28)).resize((14,14))).reshape(14*14)
        
    ## Level 2: scale level 1 in half
    print('Creating level 2 resolution...')
    X2 = np.empty((nSamp,7*7))
    for idx in range(0,nSamp):
        X2[idx,:] = np.array(Image.fromarray(X1[idx,:].reshape(14,14)).resize((7,7))).reshape(7*7)
        
    ## Level 3: scale level 2 in half
    print('Creating level 3 resolution...')
    X3 = np.empty((nSamp,3*3))
    for idx in range(0,nSamp):
        X3[idx,:] = np.array(Image.fromarray(X2[idx,:].reshape(7,7)).resize((3,3))).reshape(3*3)
        
        
    X1 = X1 - level_1_mean
    X2 = X2 - level_2_mean
    X3 = X3 - level_3_mean
    
    X1_trans = np.dot(X1, eigVec1.T)
    X2_trans = np.dot(X2, eigVec2.T)
    X3_trans = np.dot(X3, eigVec3.T)
    
    print('Creating new feature vectors for validation set...')
    nFeatures = (4*(14*14))+(4*(7*7))+(4*(3*3))
    X_val_new = np.zeros((nSamp,nFeatures))
    for idx in range(0,nSamp):
#    for idx in range(0,1):
        
        ## level 1
        x20 = X1[idx,:]
        x21 = X1_trans[idx,0]*eigVec1[0,:]
        x22 = X1_trans[idx,1]*eigVec1[1,:]
        x23 = X1_trans[idx,2]*eigVec1[2,:]
        
        x20 = x20.reshape((14*14))
        x21 = x21.reshape((14*14))
        x22 = x22.reshape((14*14))
        x23 = x23.reshape((14*14))    
        
        currentSamp1 = np.concatenate((x20,x21,x22,x23))
        
        ## level 2
        x20 = X2[idx,:]
        x21 = X2_trans[idx,0]*eigVec2[0,:]
        x22 = X2_trans[idx,1]*eigVec2[1,:]
        x23 = X2_trans[idx,2]*eigVec2[2,:]
        
        x20 = x20.reshape((7*7))
        x21 = x21.reshape((7*7))
        x22 = x22.reshape((7*7))
        x23 = x23.reshape((7*7))
#    
        currentSamp2 = np.concatenate((x20,x21,x22,x23))
        
        ## level 3
        x20 = X3[idx,:]
        x21 = X3_trans[idx,0]*eigVec3[0,:]
        x22 = X3_trans[idx,1]*eigVec3[1,:]
        x23 = X3_trans[idx,2]*eigVec3[2,:]
        
        x20 = x20.reshape((3*3))
        x21 = x21.reshape((3*3))
        x22 = x22.reshape((3*3))
        x23 = x23.reshape((3*3))
#    
        currentSamp3 = np.concatenate((x20,x21,x22,x23))
        
        ## concatenate all features
        samp = np.concatenate((currentSamp1,currentSamp2,currentSamp3))
        
        X_val_new[idx,:] = samp
        
    ########################### Test Set ################################
    
    nSamp = X_test.shape[0] # number of samples
    
    print('Creating multiple resolutions of the test data...')
    
    ## Level 1: scale in half
    print('Creating level 1 resolution...')
    X1 = np.empty((nSamp,14*14))
    for idx in range(0,nSamp):
        X1[idx,:] = np.array(Image.fromarray(X_test[idx,:].reshape(28,28)).resize((14,14))).reshape(14*14)
        
    ## Level 2: scale level 1 in half
    print('Creating level 2 resolution...')
    X2 = np.empty((nSamp,7*7))
    for idx in range(0,nSamp):
        X2[idx,:] = np.array(Image.fromarray(X1[idx,:].reshape(14,14)).resize((7,7))).reshape(7*7)
        
    ## Level 3: scale level 2 in half
    print('Creating level 3 resolution...')
    X3 = np.empty((nSamp,3*3))
    for idx in range(0,nSamp):
        X3[idx,:] = np.array(Image.fromarray(X2[idx,:].reshape(7,7)).resize((3,3))).reshape(3*3)
        
        
    X1 = X1 - level_1_mean
    X2 = X2 - level_2_mean
    X3 = X3 - level_3_mean
    
    X1_trans = np.dot(X1, eigVec1.T)
    X2_trans = np.dot(X2, eigVec2.T)
    X3_trans = np.dot(X3, eigVec3.T)
    
    print('Creating new feature vectors for test set...')
    nFeatures = (4*(14*14))+(4*(7*7))+(4*(3*3))
    X_test_new = np.zeros((nSamp,nFeatures))
    for idx in range(0,nSamp):
#    for idx in range(0,1):
        
        ## level 1
        x20 = X1[idx,:]
        x21 = X1_trans[idx,0]*eigVec1[0,:]
        x22 = X1_trans[idx,1]*eigVec1[1,:]
        x23 = X1_trans[idx,2]*eigVec1[2,:]
        
        x20 = x20.reshape((14*14))
        x21 = x21.reshape((14*14))
        x22 = x22.reshape((14*14))
        x23 = x23.reshape((14*14))    
        
        currentSamp1 = np.concatenate((x20,x21,x22,x23))
        
        ## level 2
        x20 = X2[idx,:]
        x21 = X2_trans[idx,0]*eigVec2[0,:]
        x22 = X2_trans[idx,1]*eigVec2[1,:]
        x23 = X2_trans[idx,2]*eigVec2[2,:]
        
        x20 = x20.reshape((7*7))
        x21 = x21.reshape((7*7))
        x22 = x22.reshape((7*7))
        x23 = x23.reshape((7*7))
#    
        currentSamp2 = np.concatenate((x20,x21,x22,x23))
        
        ## level 3
        x20 = X3[idx,:]
        x21 = X3_trans[idx,0]*eigVec3[0,:]
        x22 = X3_trans[idx,1]*eigVec3[1,:]
        x23 = X3_trans[idx,2]*eigVec3[2,:]
        
        x20 = x20.reshape((3*3))
        x21 = x21.reshape((3*3))
        x22 = x22.reshape((3*3))
        x23 = x23.reshape((3*3))
#    
        currentSamp3 = np.concatenate((x20,x21,x22,x23))
        
        ## concatenate all features
        samp = np.concatenate((currentSamp1,currentSamp2,currentSamp3))
        
        X_test_new[idx,:] = samp

    return X_train_new, X_val_new, X_test_new

def mpca_with_fft(X_train, X_val, X_test):
    """
    ******************************************************************
        *  Func:      mpca_with_fft()
        *  Desc:      Create new feature representations using PCA-M.
        *  Inputs:    nSamples x nFeatures training, validation, and test sets
        *  Outputs:   Multiscale feature vectors for each data set
    ******************************************************************
    """
    ####################### Training Set ##############################
    
    ## Create multiple resolutions of the images
    nSamp = X_train.shape[0] # number of samples
    
    print('Creating multiple resolutions of the training data...')
    
    ## Level 1: scale in half
    print('Creating level 1 resolution...')
    X1 = np.empty((nSamp,14*14))
    for idx in range(0,nSamp):
        X1[idx,:] = np.array(Image.fromarray(X_train[idx,:].reshape(28,28)).resize((14,14))).reshape(14*14)
        
    ## Level 2: scale level 1 in half
    print('Creating level 2 resolution...')
    X2 = np.empty((nSamp,7*7))
    for idx in range(0,nSamp):
        X2[idx,:] = np.array(Image.fromarray(X1[idx,:].reshape(14,14)).resize((7,7))).reshape(7*7)
        
    ## Level 3: scale level 2 in half
    print('Creating level 3 resolution...')
    X3 = np.empty((nSamp,3*3))
    for idx in range(0,nSamp):
        X3[idx,:] = np.array(Image.fromarray(X2[idx,:].reshape(7,7)).resize((3,3))).reshape(3*3)
    
    ## Apply PCA to training data
    print('Applying PCA to training data...')
    
    ## Subtract means
    level_1_mean = np.mean(X1,axis=0)
    level_2_mean = np.mean(X2,axis=0)
    level_3_mean = np.mean(X3,axis=0)
    
    X1 = X1 - level_1_mean
    X2 = X2 - level_2_mean
    X3 = X3 - level_3_mean
    
    pca1 = PCA()
    pca1.fit(X1)
    eigVec1 = pca1.components_
    eigVal1 = pca1.explained_variance_
    
    X1_trans = np.dot(X1, eigVec1.T)
    
    pca2 = PCA()
    pca2.fit(X2)
    eigVec2 = pca2.components_
    eigVal2 = pca2.explained_variance_
    
    X2_trans = np.dot(X2, eigVec2.T)
    
    pca3 = PCA()
    pca3.fit(X3)
    eigVec3 = pca3.components_
    eigVal3 = pca3.explained_variance_
    
    X3_trans = np.dot(X3, eigVec3.T)
    
    
    print('Creating new feature vectors for training set...')
    nFeatures = (4*(14*14))+(4*(7*7))+(4*(3*3))
    X_train_new = np.zeros((nSamp,nFeatures))
    for idx in range(0,nSamp):
#    for idx in range(0,1):
        
        ## level 1
        x20 = X1[idx,:]
        x21 = X1_trans[idx,0]*eigVec1[0,:]
        x22 = X1_trans[idx,1]*eigVec1[1,:]
        x23 = X1_trans[idx,2]*eigVec1[2,:]
        
#        fig, ax = plt.subplots(2,2)
#        ax[0,0].imshow(x20.reshape((14,14)))
#        ax[0,1].imshow(x21.reshape((14,14)))
#        ax[1,0].imshow(x22.reshape((14,14)))
#        ax[1,1].imshow(x23.reshape((14,14)))
        
        ## convert to frequency domain
        f = np.fft.fft2(x20.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x20 = magnitude_spectrum
        
        f = np.fft.fft2(x21.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x21 = magnitude_spectrum
        
        f = np.fft.fft2(x22.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x22 = magnitude_spectrum
        
        f = np.fft.fft2(x23.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x23 = magnitude_spectrum
        
        x20 = x20.reshape((14*14))
        x21 = x21.reshape((14*14))
        x22 = x22.reshape((14*14))
        x23 = x23.reshape((14*14))    
        
        currentSamp1 = np.concatenate((x20,x21,x22,x23))
        
        ## level 2
        x20 = X2[idx,:]
        x21 = X2_trans[idx,0]*eigVec2[0,:]
        x22 = X2_trans[idx,1]*eigVec2[1,:]
        x23 = X2_trans[idx,2]*eigVec2[2,:]
        
        ## convert to frequency domain
        f = np.fft.fft2(x20.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x20 = magnitude_spectrum
        
        f = np.fft.fft2(x21.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x21 = magnitude_spectrum
        
        f = np.fft.fft2(x22.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x22 = magnitude_spectrum
        
        f = np.fft.fft2(x23.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x23 = magnitude_spectrum
        
        x20 = x20.reshape((7*7))
        x21 = x21.reshape((7*7))
        x22 = x22.reshape((7*7))
        x23 = x23.reshape((7*7))
#    
        currentSamp2 = np.concatenate((x20,x21,x22,x23))
        
        ## level 3
        x20 = X3[idx,:]
        x21 = X3_trans[idx,0]*eigVec3[0,:]
        x22 = X3_trans[idx,1]*eigVec3[1,:]
        x23 = X3_trans[idx,2]*eigVec3[2,:]
        
        ## convert to frequency domain
        f = np.fft.fft2(x20.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x20 = magnitude_spectrum
        
        f = np.fft.fft2(x21.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x21 = magnitude_spectrum
        
        f = np.fft.fft2(x22.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x22 = magnitude_spectrum
        
        f = np.fft.fft2(x23.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x23 = magnitude_spectrum
        
        x20 = x20.reshape((3*3))
        x21 = x21.reshape((3*3))
        x22 = x22.reshape((3*3))
        x23 = x23.reshape((3*3))
#    
        currentSamp3 = np.concatenate((x20,x21,x22,x23))
        
        ## concatenate all features
        samp = np.concatenate((currentSamp1,currentSamp2,currentSamp3))
        
        X_train_new[idx,:] = samp
        
        
    ####################### Validation Set ##############################
    
    
    nSamp = X_val.shape[0] # number of samples
    
    print('Creating multiple resolutions of the validation data...')
    
    ## Level 1: scale in half
    print('Creating level 1 resolution...')
    X1 = np.empty((nSamp,14*14))
    for idx in range(0,nSamp):
        X1[idx,:] = np.array(Image.fromarray(X_val[idx,:].reshape(28,28)).resize((14,14))).reshape(14*14)
        
    ## Level 2: scale level 1 in half
    print('Creating level 2 resolution...')
    X2 = np.empty((nSamp,7*7))
    for idx in range(0,nSamp):
        X2[idx,:] = np.array(Image.fromarray(X1[idx,:].reshape(14,14)).resize((7,7))).reshape(7*7)
        
    ## Level 3: scale level 2 in half
    print('Creating level 3 resolution...')
    X3 = np.empty((nSamp,3*3))
    for idx in range(0,nSamp):
        X3[idx,:] = np.array(Image.fromarray(X2[idx,:].reshape(7,7)).resize((3,3))).reshape(3*3)
        
        
    X1 = X1 - level_1_mean
    X2 = X2 - level_2_mean
    X3 = X3 - level_3_mean
    
    X1_trans = np.dot(X1, eigVec1.T)
    X2_trans = np.dot(X2, eigVec2.T)
    X3_trans = np.dot(X3, eigVec3.T)
    
    print('Creating new feature vectors for validation set...')
    nFeatures = (4*(14*14))+(4*(7*7))+(4*(3*3))
    X_val_new = np.zeros((nSamp,nFeatures))
    for idx in range(0,nSamp):
#    for idx in range(0,1):
        
        ## level 1
        x20 = X1[idx,:]
        x21 = X1_trans[idx,0]*eigVec1[0,:]
        x22 = X1_trans[idx,1]*eigVec1[1,:]
        x23 = X1_trans[idx,2]*eigVec1[2,:]
        
        ## convert to frequency domain
        f = np.fft.fft2(x20.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x20 = magnitude_spectrum
        
        f = np.fft.fft2(x21.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x21 = magnitude_spectrum
        
        f = np.fft.fft2(x22.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x22 = magnitude_spectrum
        
        f = np.fft.fft2(x23.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x23 = magnitude_spectrum
        
        x20 = x20.reshape((14*14))
        x21 = x21.reshape((14*14))
        x22 = x22.reshape((14*14))
        x23 = x23.reshape((14*14))    
        
        currentSamp1 = np.concatenate((x20,x21,x22,x23))
        
        ## level 2
        x20 = X2[idx,:]
        x21 = X2_trans[idx,0]*eigVec2[0,:]
        x22 = X2_trans[idx,1]*eigVec2[1,:]
        x23 = X2_trans[idx,2]*eigVec2[2,:]
        
        ## convert to frequency domain
        f = np.fft.fft2(x20.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x20 = magnitude_spectrum
        
        f = np.fft.fft2(x21.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x21 = magnitude_spectrum
        
        f = np.fft.fft2(x22.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x22 = magnitude_spectrum
        
        f = np.fft.fft2(x23.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x23 = magnitude_spectrum
        
        x20 = x20.reshape((7*7))
        x21 = x21.reshape((7*7))
        x22 = x22.reshape((7*7))
        x23 = x23.reshape((7*7))
#    
        currentSamp2 = np.concatenate((x20,x21,x22,x23))
        
        ## level 3
        x20 = X3[idx,:]
        x21 = X3_trans[idx,0]*eigVec3[0,:]
        x22 = X3_trans[idx,1]*eigVec3[1,:]
        x23 = X3_trans[idx,2]*eigVec3[2,:]
        
        ## convert to frequency domain
        f = np.fft.fft2(x20.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x20 = magnitude_spectrum
        
        f = np.fft.fft2(x21.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x21 = magnitude_spectrum
        
        f = np.fft.fft2(x22.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x22 = magnitude_spectrum
        
        f = np.fft.fft2(x23.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x23 = magnitude_spectrum
        
        x20 = x20.reshape((3*3))
        x21 = x21.reshape((3*3))
        x22 = x22.reshape((3*3))
        x23 = x23.reshape((3*3))
#    
        currentSamp3 = np.concatenate((x20,x21,x22,x23))
        
        ## concatenate all features
        samp = np.concatenate((currentSamp1,currentSamp2,currentSamp3))
        
        X_val_new[idx,:] = samp
        
    ########################### Test Set ################################
    
    nSamp = X_test.shape[0] # number of samples
    
    print('Creating multiple resolutions of the test data...')
    
    ## Level 1: scale in half
    print('Creating level 1 resolution...')
    X1 = np.empty((nSamp,14*14))
    for idx in range(0,nSamp):
        X1[idx,:] = np.array(Image.fromarray(X_test[idx,:].reshape(28,28)).resize((14,14))).reshape(14*14)
        
    ## Level 2: scale level 1 in half
    print('Creating level 2 resolution...')
    X2 = np.empty((nSamp,7*7))
    for idx in range(0,nSamp):
        X2[idx,:] = np.array(Image.fromarray(X1[idx,:].reshape(14,14)).resize((7,7))).reshape(7*7)
        
    ## Level 3: scale level 2 in half
    print('Creating level 3 resolution...')
    X3 = np.empty((nSamp,3*3))
    for idx in range(0,nSamp):
        X3[idx,:] = np.array(Image.fromarray(X2[idx,:].reshape(7,7)).resize((3,3))).reshape(3*3)
        
        
    X1 = X1 - level_1_mean
    X2 = X2 - level_2_mean
    X3 = X3 - level_3_mean
    
    X1_trans = np.dot(X1, eigVec1.T)
    X2_trans = np.dot(X2, eigVec2.T)
    X3_trans = np.dot(X3, eigVec3.T)
    
    print('Creating new feature vectors for test set...')
    nFeatures = (4*(14*14))+(4*(7*7))+(4*(3*3))
    X_test_new = np.zeros((nSamp,nFeatures))
    for idx in range(0,nSamp):
#    for idx in range(0,1):
        
        ## level 1
        x20 = X1[idx,:]
        x21 = X1_trans[idx,0]*eigVec1[0,:]
        x22 = X1_trans[idx,1]*eigVec1[1,:]
        x23 = X1_trans[idx,2]*eigVec1[2,:]
        
        ## convert to frequency domain
        f = np.fft.fft2(x20.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x20 = magnitude_spectrum
        
        f = np.fft.fft2(x21.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x21 = magnitude_spectrum
        
        f = np.fft.fft2(x22.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x22 = magnitude_spectrum
        
        f = np.fft.fft2(x23.reshape((14,14)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x23 = magnitude_spectrum
        
        x20 = x20.reshape((14*14))
        x21 = x21.reshape((14*14))
        x22 = x22.reshape((14*14))
        x23 = x23.reshape((14*14))    
        
        currentSamp1 = np.concatenate((x20,x21,x22,x23))
        
        ## level 2
        x20 = X2[idx,:]
        x21 = X2_trans[idx,0]*eigVec2[0,:]
        x22 = X2_trans[idx,1]*eigVec2[1,:]
        x23 = X2_trans[idx,2]*eigVec2[2,:]
        
        ## convert to frequency domain
        f = np.fft.fft2(x20.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x20 = magnitude_spectrum
        
        f = np.fft.fft2(x21.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x21 = magnitude_spectrum
        
        f = np.fft.fft2(x22.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x22 = magnitude_spectrum
        
        f = np.fft.fft2(x23.reshape((7,7)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x23 = magnitude_spectrum
        
        x20 = x20.reshape((7*7))
        x21 = x21.reshape((7*7))
        x22 = x22.reshape((7*7))
        x23 = x23.reshape((7*7))
#    
        currentSamp2 = np.concatenate((x20,x21,x22,x23))
        
        ## level 3
        x20 = X3[idx,:]
        x21 = X3_trans[idx,0]*eigVec3[0,:]
        x22 = X3_trans[idx,1]*eigVec3[1,:]
        x23 = X3_trans[idx,2]*eigVec3[2,:]
        
        ## convert to frequency domain
        f = np.fft.fft2(x20.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x20 = magnitude_spectrum
        
        f = np.fft.fft2(x21.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x21 = magnitude_spectrum
        
        f = np.fft.fft2(x22.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x22 = magnitude_spectrum
        
        f = np.fft.fft2(x23.reshape((3,3)))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        x23 = magnitude_spectrum
        
        x20 = x20.reshape((3*3))
        x21 = x21.reshape((3*3))
        x22 = x22.reshape((3*3))
        x23 = x23.reshape((3*3))
#    
        currentSamp3 = np.concatenate((x20,x21,x22,x23))
        
        ## concatenate all features
        samp = np.concatenate((currentSamp1,currentSamp2,currentSamp3))
        
        X_test_new[idx,:] = samp


    return X_train_new, X_val_new, X_test_new
