# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 00:29:44 2016

@author: smadaan
"""

#%%        
##Read the representative frames from the avg interpolated wavelet file
R3 = np.zeros(((sizeA1[0],sizeA1[1],meanperiodframes)))
for b in np.arange(index_start_number_BF,index_start_number_BF+meanperiodframes):
    R3[:,:,b-index_start_number_BF]= importtiff(repfolder_avg_interpolated,int(b),prefix=repname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            #R reads the frame number b from the stagemotioncorrected files.
#%%        
##Read the representative frames from the avg interpolated wavelet file
R4 = np.zeros(((sizeA1[0],sizeA1[1],meanperiodframes)))
for b in np.arange(index_start_number_BF,index_start_number_BF+meanperiodframes):
    R4[:,:,b-index_start_number_BF]= importtiff(repfolder_avg_interpolated,int(b),prefix=repname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            #R reads the frame number b from the stagemotioncorrected files.

