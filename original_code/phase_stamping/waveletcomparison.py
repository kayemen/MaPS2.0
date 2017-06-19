# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 21:36:37 2016

@author: smadaan
"""

def SADWaC(i1, i2, level = 1,sublevel='cD1',wav_name='db1',mode = 'sym',maskon = 0, mask = 0,dispon =0):
    
    import pywt    
    import numpy as np
    import matplotlib.pyplot as plt
    
    if (maskon==1):
        i1 = i1*mask
        i2 = i2*mask
    
    
    
    if (level == 1):
        (c1A1, (c1H1, c1V1, c1D1)) = pywt.dwt2(i1,wav_name,  mode=mode) 
        (c2A1, (c2H1, c2V1, c2D1)) = pywt.dwt2(i2,wav_name,  mode=mode)
        
        if (sublevel == 'cA1'):
            SAD = np.sum(np.absolute(c1A1-c2A1))
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1A1, cmap=plt.cm.gray, hold = 'on')

        if (sublevel == 'cV1'):
            SAD = np.sum(np.absolute(c1V1-c2V1))
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1V1, cmap=plt.cm.gray, hold = 'on')
                
        if (sublevel == 'cH1'):
            SAD = np.sum(np.absolute(c1H1-c2H1))
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1H1, cmap=plt.cm.gray, hold = 'on')
                
        if (sublevel == 'cD1'):
            SAD = np.sum(np.absolute(c1D1-c2D1))
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1D1, cmap=plt.cm.gray, hold = 'on')
         
         
         
    if (level == 2):
        (c1A2, (c1H2, c1V2, c1D2), (c1H1, c1V1, c1D1)) = pywt.wavedec2(i1,wav_name,  mode=mode, level = 2) 
        (c2A2, (c2H2, c2V2, c2D2), (c2H1, c2V1, c2D1)) = pywt.wavedec2(i2,wav_name,  mode=mode, level = 2)
        
        if (sublevel == 'cA2'):
            SAD = np.sum(np.absolute(c1A2-c2A2))
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1A2, cmap=plt.cm.gray, hold = 'on')
                
        if (sublevel == 'cV1'):
            SAD = np.sum(np.absolute(c1V1-c2V1))
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1V1, cmap=plt.cm.gray, hold = 'on')
                
        if (sublevel == 'cH1'):
            SAD = np.sum(np.absolute(c1H1-c2H1))
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1H1, cmap=plt.cm.gray, hold = 'on')
                
        if (sublevel == 'cD1'):
            SAD = np.sum(np.absolute(c1D1-c2D1)) 
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1D1, cmap=plt.cm.gray, hold = 'on')
                
        if (sublevel == 'cV2'):
            SAD = np.sum(np.absolute(c1V2-c2V2))
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1V2, cmap=plt.cm.gray, hold = 'on')
                
        if (sublevel == 'cH2'):
            SAD = np.sum(np.absolute(c1H2-c2H2))
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1H2, cmap=plt.cm.gray, hold = 'on')
                
        if (sublevel == 'cD2'):
            SAD = np.sum(np.absolute(c1D2-c2D2)) 
            if (dispon==1):
                plt.figure(631)
                plt.imshow(c1D2, cmap=plt.cm.gray, hold = 'on')
                
            
    return SAD    
    