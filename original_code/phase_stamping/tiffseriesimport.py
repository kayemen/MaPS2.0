# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:46:29 2016

@author: smadaan
"""


def importtiff(folderpath,frame_number,prefix='FRAMEX',index_start_number=0,num_digits=5):
    '''this function loads a single tiff frame. If you have a tiff files named
       frame0011 to frame9999 and they are in folder E:/somefolder and you
       want to access the 2nd frame in the sequence, then you use the function 
       as follows (remember frames start with index 0 in python and in fiji)
       folderpath = 'E:/somefolder'
       prefix = 'frame'
       frame_number = 1 
       
       secondframe = importtiff(folderpath,frame_number,prefix=prefix,index_start_number=11,num_digits=4)
       
       By default, 
       prefix = 'FRAMEX'
       index_start_number = 1
       num_digits = 5
       
    
    
    '''
    import skimage.external.tifffile as tff
#    index_start_number = 1
    current_index = frame_number + index_start_number 
#    num_digits = 5
#    prefix = 'FRAMEX'
#    folderpath = 'E:/1USC/Dr. Fraser/Projects/Zebrafish Heart Atlas/150428-0502_PSRun/TEST_MISREG_Z_1_PIX_WAVE/150429_3a_homo_54hpf_sample1/Side 1/Recon alg and files/Data_Files/Phase_Side1_crop_seq_bfr'
    filename = folderpath+'/'+prefix+str(current_index).zfill(num_digits)+'.tif'
    image = tff.imread(filename)
    return image
    
    
    
    
    
    
    
    
    
    
def writetiff(data,folderpath,frame_number,prefix='FRAMEX',index_start_number=0,num_digits=5):
    '''this function loads a single tiff frame. If you have a tiff files named
       frame0011 to frame9999 and they are in folder E:/somefolder and you
       want to access the 2nd frame in the sequence, then you use the function 
       as follows (remember frames start with index 0 in python and in fiji)
       folderpath = 'E:/somefolder'
       prefix = 'frame'
       frame_number = 1 
       
       secondframe = importtiff(folderpath,frame_number,prefix=prefix,index_start_number=11,num_digits=4)
       
       By default, 
       prefix = 'FRAMEX'
       index_start_number = 1
       num_digits = 5
       
    
    
    '''
    import skimage.external.tifffile as tff
#    index_start_number = 1
    current_index = frame_number + index_start_number 
#    num_digits = 5
#    prefix = 'FRAMEX'
#    folderpath = 'E:/1USC/Dr. Fraser/Projects/Zebrafish Heart Atlas/150428-0502_PSRun/TEST_MISREG_Z_1_PIX_WAVE/150429_3a_homo_54hpf_sample1/Side 1/Recon alg and files/Data_Files/Phase_Side1_crop_seq_bfr'
    filename = folderpath+'/'+prefix+str(current_index).zfill(num_digits)+'.tif'
    tff.imsave(filename,data)
