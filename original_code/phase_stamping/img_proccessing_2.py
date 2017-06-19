# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:33:53 2016

@author: smadaan
"""

def ZookZik_1X_v1_MinimumSADCluster(R,RefImage):
    import numpy as np
    
    Sad = np.zeros(R.shape[2])
    for t in range(R.shape[2]):
        Sad[t]= np.sum(np.absolute(RefImage-R[:,:,t]))

    min_sadvalue_intermediate = np.amin(Sad)
    min_sadframe_intermediate = np.argmin(Sad)
    
    min_sadframeandvalue = [min_sadframe_intermediate,min_sadvalue_intermediate];
     
    ## use this to use histogram but avoid the for loop in t
    ##http://stackoverflow.com/questions/18851471/numpy-histogram-on-multi-dimensional-array
    ##
    return min_sadframeandvalue