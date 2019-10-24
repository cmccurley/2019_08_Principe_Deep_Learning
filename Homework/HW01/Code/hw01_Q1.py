# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:03:10 2019

@author: Conma
"""

# -*- coding: utf-8 -*-
"""
***********************************************************************
    *  File:  hw01_Q1.py
    *  Name:  Connor H. McCurley
    *  Date:  2019-09-10
    *  Desc:  Provides coded solutions to homework set 01 of EEL681,
    *         Deep Learning, taught by Dr. Jose Principe, Fall 2019.
**********************************************************************
"""
import numpy as np
import matplotlib.pyplot as plt



""" ====================== Network definitions ============================="""

#data set 1 solution 1
def noseNet(X):

  L1 = np.array([[0,1,1],[1,1,0],[-1,1,0]])
  L2 = np.array([2,-0.5,-0.5,-2])
  
  X = np.vstack((X,np.ones(np.shape(X)[1])))  
  
  O1 = np.sign(L1 @ X)  
  O1 = np.vstack((O1,np.ones(np.shape(O1)[1])))
  
  Y = np.sign(L2 @ O1)
  Y[Y==0] = -1
   
  return Y


def mouthNet(X):

  L1 = np.array([[0,1,3],[0,1,4],[1,0,3],[1,0,-3]])
  L2 = np.array([-3,3,0.5,-1,-7])
  
  X = np.vstack((X,np.ones(np.shape(X)[1])))  
  
  O1 = np.sign(L1 @ X)  
  O1 = np.vstack((O1,np.ones(np.shape(O1)[1])))
  
  Y = np.sign(L2 @ O1)
  
  Y[Y==0] = -1
   
  return Y
          
def leftEyeNet(X):

  L1 = np.array([[0,1,-5],[0,1,-4],[1,0,5],[1,0,4]])
  L2 = np.array([-3,3,0.5,-1,-7])
  
  X = np.vstack((X,np.ones(np.shape(X)[1])))  
  
  O1 = np.sign(L1 @ X)  
  O1 = np.vstack((O1,np.ones(np.shape(O1)[1])))
  
  Y = np.sign(L2 @ O1)
  
  Y[Y==0] = -1
   
  return Y

def rightEyeNet(X):

  L1 = np.array([[0,1,-5],[0,1,-4],[1,0,-5],[1,0,-4]])
  L2 = np.array([-3,3,-2,2,-6])
  
  X = np.vstack((X,np.ones(np.shape(X)[1])))  
  
  O1 = np.sign(L1 @ X)  
  O1 = np.vstack((O1,np.ones(np.shape(O1)[1])))
  
  Y = np.sign(L2 @ O1)
  
  Y[Y==0] = -1
   
  return Y

def maskNet(X):
    
    X = np.vstack((X,np.ones(np.shape(X)[1]))) 
    
    #nose 
    noseL1 = np.array([[0,1,1],[1,1,0],[-1,1,0]])
    noseL2 = np.array([2,-0.5,-0.5,-2])
    noseO1 = np.sign(noseL1 @ X)  
    noseO1 = np.vstack((noseO1,np.ones(np.shape(noseO1)[1])))
    noseY = np.sign(noseL2 @ noseO1)
    noseY[noseY==0] = -1
    
    #mouth
    mouthL1 = np.array([[0,1,3],[0,1,4],[1,0,3],[1,0,-3]])
    mouthL2 = np.array([-3,3,0.5,-1,-7])
    mouthO1 = np.sign(mouthL1 @ X)  
    mouthO1 = np.vstack((mouthO1,np.ones(np.shape(mouthO1)[1])))
    mouthY = np.sign(mouthL2 @ mouthO1)
    mouthY[mouthY==0] = -1
    
    #left eye
    leyeL1 = np.array([[0,1,-5],[0,1,-4],[1,0,5],[1,0,4]])
    leyeL2 = np.array([-3,3,0.5,-1,-7])
    leyeO1 = np.sign(leyeL1 @ X)  
    leyeO1 = np.vstack((leyeO1,np.ones(np.shape(leyeO1)[1])))
    leyeY = np.sign(leyeL2 @ leyeO1)
    leyeY[leyeY==0] = -1
    
    #right eye
    reyeL1 = np.array([[0,1,-5],[0,1,-4],[1,0,-5],[1,0,-4]])
    reyeL2 = np.array([-3,3,-2,2,-6])
    reyeO1 = np.sign(reyeL1 @ X)  
    reyeO1 = np.vstack((reyeO1,np.ones(np.shape(reyeO1)[1])))
    reyeY = np.sign(reyeL2 @ reyeO1)
    reyeY[reyeY==0] = -1
  
    L3 =  np.array([[1,1,1,1]])
    X_out = np.squeeze(np.array([[noseY],[mouthY],[leyeY],[reyeY]])) 
    Y =  np.sign(L3 @ X_out)
    Y[Y==0] =-1

    return Y
  

""" ================== Display Discrimination Regions ======================"""
#create grid for displaying boundaries
grid_x = np.linspace(-10,10,100)
grid_y = np.linspace(-10,10,100)
map_X, map_Y = np.meshgrid(grid_x, grid_y)
map_vals = np.vstack((map_X.reshape((np.prod(map_X.shape),)),map_Y.reshape((np.prod(map_Y.shape),))))


################################# Nose ########################################
#pass grid through network to get labels for boundary coloring
regions = noseNet(map_vals)
regions = regions.reshape(map_X.shape,)

#plot decision boundaries
plt.figure()
plt.imshow(regions,origin='lower',extent=np.array([np.min(grid_x),np.max(grid_x),np.min(grid_y),np.max(grid_y)]))
plt.title("Nose")
plt.savefig("E:\\University of Florida\\Classes\\2019_08_Principe_Deep_Learning\\Homework\\HW01\\Report\\Images\\Nose.jpg")
plt.close()

################################ Mouth ########################################
regions = mouthNet(map_vals)
regions = regions.reshape(map_X.shape,)

#plot decision boundaries
plt.figure()
plt.imshow(regions,origin='lower',extent=np.array([np.min(grid_x),np.max(grid_x),np.min(grid_y),np.max(grid_y)]))
plt.title("Mouth")
plt.savefig("E:\\University of Florida\\Classes\\2019_08_Principe_Deep_Learning\\Homework\\HW01\\Report\\Images\\Mouth.jpg")
plt.close()

############################### Left Eye ######################################
regions = leftEyeNet(map_vals)
regions = regions.reshape(map_X.shape,)

#plot decision boundaries
plt.figure()
plt.imshow(regions,origin='lower',extent=np.array([np.min(grid_x),np.max(grid_x),np.min(grid_y),np.max(grid_y)]))
plt.title("Left Eye")
plt.savefig("E:\\University of Florida\\Classes\\2019_08_Principe_Deep_Learning\\Homework\\HW01\\Report\\Images\\LeftEye.jpg")
plt.close()

############################## Right Eye ######################################
regions = rightEyeNet(map_vals)
regions = regions.reshape(map_X.shape,)

#plot decision boundaries
plt.figure()
plt.imshow(regions,origin='lower',extent=np.array([np.min(grid_x),np.max(grid_x),np.min(grid_y),np.max(grid_y)]))
plt.title("Right Eye")
plt.savefig("E:\\University of Florida\\Classes\\2019_08_Principe_Deep_Learning\\Homework\\HW01\\Report\\Images\\RightEye.jpg")
plt.close()




