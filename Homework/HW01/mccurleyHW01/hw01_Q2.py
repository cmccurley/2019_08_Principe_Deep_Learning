# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:39:11 2019

@author: Conma
"""

"""
***********************************************************************
    *  File:  hw01_Q2.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-09-10
    *  Desc:  Provides coded solutions to homework set 01 of EEL681,
    *         Deep Learning, taught by Dr. Jose Principe, Fall 2019.
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


######################################################################
##################### Function Definitions ###########################
######################################################################
 
def genData():
    """
    ******************************************************************
        *  Func:      genData()
        *  Desc:      Generate Star Problem data set.
        *  Inputs:
        *  Outputs: 
    ******************************************************************
    """
    X = np.array(([1,0],[0,1],[-1,0],[0,-1],[0.5,0.5],[-0.5,0.5],[0.5,-0.5],[-0.5,-0.5]), dtype=float)
    y = np.array(([1],[1],[1],[1],[0],[0],[0],[0]), dtype=float)
    return X,y

 
def plotData(X,y,savePath,title):
    """
    ******************************************************************
        *  Func:      plotData()
        *  Desc:Plot  Star Problem data points colored by label.
        *  Inputs:
        *  Outputs: 
    ******************************************************************
    """
    # plot figure colored by label
    plt.figure(figsize=(8,8))
    colors = ['blue','red']
    plt.scatter(X[:,0],X[:,1], c=np.squeeze(y),cmap=matplotlib.colors.ListedColormap(colors))
    plt.title(title, fontsize=18)
    plt.xlabel('Feature 1', fontsize=16)
    plt.ylabel('Feature 2', fontsize=16)
    plt.xlim(-2,2)
    plt.ylim(-2,2)

    # Save figure to desired path
    plt.savefig(savePath)
    
    return

def plotBoundaries(X,y,savePath,title):
    """
    ******************************************************************
        *  Func:      plotData()
        *  Desc:
        *  Inputs:
        *  Outputs: 
    ******************************************************************
    """
    # plot data and predicted decision boundaries
    plt.figure(figsize=(8,8))
    colors = ['blue','red']
    plt.scatter(X[:,0],X[:,1], c=np.squeeze(y),cmap=matplotlib.colors.ListedColormap(colors))
    plt.title(title, fontsize=18)
    plt.xlabel('Feature 1', fontsize=16)
    plt.ylabel('Feature 2', fontsize=16)
    plt.xlim(-2,2)
    plt.ylim(-2,2)

    # Save figure to desired path
#    plt.savefig(savePath)
    return

############################ Define MLP class #################################
class MLP(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 4
#    self.learningRate = 0.8
    self.learningRate = 1.2

    #weights
    self.W1 = np.random.randn(self.inputSize+1, self.hiddenSize) # (3x4) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize+1, self.outputSize) # (4x1) weight matrix from hidden to output layer

  def propagateForward(self, X):
    # pass each data point through the network and get a predicted output
    
    self.z = np.dot(X, self.W1) # net of first layer
    self.z2 = self.sigmoid(self.z) # apply activation
    self.z2 = np.concatenate((self.z2, np.ones((X.shape[0],1))), axis=1) #add ones for bias
    self.z3 = np.dot(self.z2, self.W2) # net of output layer
    out = self.sigmoid(self.z3) # apply activation at output layer
    return out 

  def sigmoid(self, s):
    # sigmoid activation function, returns value [-1,1]
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    # returns the derivative of the sigmoid activation function
    return s * (1 - s)

  def stepAct(self, s):
      # returns {0,1} based on sign of input
      actVal = 0
      if (s>0):
          actVal = 1
      return actVal

  def backwardPass(self, X, y, yPred):
    # propogate error through the network and update weights
    self.out_error = y - yPred # output error
    self.out_delta = self.out_error*self.sigmoidPrime(yPred) 

    self.z2_error = self.out_delta.dot(self.W2.T)
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # local gradients at output layer for each datapoint

    # update weights in each layer
    self.W1 += self.learningRate*X.T.dot(self.z2_delta[:,0:-1])
    self.W2 += self.learningRate*self.z2.T.dot(self.out_delta) 
    
    # To learn horizontal lines only
    self.W1[0,0] = 0
    self.W1[0,1] = 0
    self.W1[1,2] = 0
    self.W1[1,3] = 0
    

  def train (self, X, y):
    yPred = self.propagateForward(X)
    self.backwardPass(X, y, yPred)

######################################################################
############################## Main ##################################
######################################################################
if __name__== "__main__":
    
    print('Running Main...')
    
    ####################### Generate data ############################
    X,y =  genData()
    
    #add column of onesfor bias
    X = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
    
#    # Plot data points
#    cwd = os.getcwd()
#    os.chdir('..\\Report\\Images\\')
#    cwd = os.getcwd()
#    savePath = cwd + "\\Q2_data.jpg"
#    plotData(X, y, savePath, 'Question 2 Data')


    ####################### Define Network ###########################
    net = MLP()
    
    ##################### Train the Network ##########################
    
    allLoss = []
    allWeights = np.empty((12,1))   
    for i in range(5000):
        print(f"Iteration: {i}")
        loss = np.mean(np.square(y - net.propagateForward(X)))
        allLoss = np.append(allLoss,[loss])
        print("Loss: \n" + str(loss)) # mean sum squared loss
        
        #stop training if loss falls below threshold
        if (loss < 0.001):
            break
        net.train(X, y)
        vec =  net.W1.reshape((net.W1.shape[0]*net.W1.shape[1],1))
        allWeights = np.append(allWeights, vec, axis=1)
        
    #################### Learning Curve ##############################
    # plot the learning curve
    plt.figure()
    plt.plot(np.arange(0,i+1,1),allLoss)
    plt.title("Learing Curve", fontsize=18)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Mean-Square Error (MSE)', fontsize=12)
    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_learning_curve_exact.jpg')
    plt.close()
    
    ##################### Weight Tracks ##############################
    # plot weight tracks in the first hidden layer
    plt.figure()
    
    for weight in range(0,allWeights.shape[0]):
        plt.plot(np.arange(0,i+1,1),allWeights[weight,:])
    plt.title("Weight Tracks", fontsize=18)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Weight Value', fontsize=12)
    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_weight_tracks_exact.jpg')
    plt.close()
    
    
    ################# Generalization of Performance ##################
   
    #create grid for displaying boundaries
    grid_x = np.linspace(-2,2,1000)
    grid_y = np.linspace(-2,2,1000)
    map_X, map_Y = np.meshgrid(grid_x, grid_y)
    map_vals = np.vstack((map_X.reshape((np.prod(map_X.shape),)),map_Y.reshape((np.prod(map_Y.shape),))))
    
    X_test = map_vals.T
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0],1))), axis=1)

    #pass grid through network to get labels for boundary coloring
    regions = net.propagateForward(X_test)
    regions = regions.reshape(map_X.shape,)
    
    #Make output binary
    regions[regions>0.5] = 1
    regions[regions<=0.5] = 0
    
    #plot decision boundaries
    plt.figure()
    plt.imshow(regions,origin='lower',extent=np.array([np.min(grid_x),np.max(grid_x),np.min(grid_y),np.max(grid_y)]),cmap='jet')
    plt.title("Decision Regions", fontsize=18)
#    plt.savefig("E:\\University of Florida\\Classes\\2019_08_Principe_Deep_Learning\\Homework\\HW01\\Report\\Images\\Q2_decision_regions_exact.jpg")
    plt.savefig('C:\\Users\\Conma\\Desktop\\HW01\\Report\\Images\\Q2_decision_regions_exact.jpg')
    plt.close()
   
    
    print('================ DONE ================')

