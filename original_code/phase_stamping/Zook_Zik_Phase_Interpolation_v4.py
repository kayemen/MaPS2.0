# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:43:25 2016

@author: smadaan
"""


###############################################################################
###############################################################################
### This program has been written by Sara Madaan###
### Contact smadaan@usc.edu for questions       ###
### 2312 Ellendale Place                        ###
### Los Angeles , CA 90007                      ###
###
### This program is for 4D reconstruction using ZookZik
###
### Description of INPUTS
###
### name = name of BF image set
### name_kymo = name of kymograph image generated from FIJI
### FirstRefFrame = This is the refence for z position
### Period_apx = Manually estimated number of periods;
### heartthickness = Number of microns moved while imaging;
### Cor_threshhold = Lower Bound on correlation values for comparison during
### phase stamping on original images
### Cor_threshhold_segmented = Lower Bound on correlation values for comparison
### during phase stamping on segmented images
###
###
### Description of OUTPUTS
###
### R = reference frames for phasestamping
### zstamp = z value for every frame in original images
### pstamp = phase value for every frame in original images
### zpstamp = 2D matrix with frame number as entries for a given (z,p) value
### zpstamp_redundancy = 2D matrix with redundancy as entries for a given (z,p)
### value
### emptypercentage = % of unfilled entries in the zpstamp
###
###
### Description of steps
### Step 01 : Take the inputs for the first data set
### Step 02 : Find the turning points of stage movement from the kymograph
### Step 03 : Using the reference frame to infer the width of sliding box
### Step 04 : Subtracting Stage Motion and storing first 10% of zook frames
### Step 05 : Using the StageMotion Corrected image to get the representative
### images
### Step 06 : Using the reference Image set to infer phases
### Step 07 : Compiling the z and p info in a table
### Step 08 : Read Fluorescent Images to find the best render-able image in 4D
### Step 09 : Read Fluorecent Images to generate the tiff files per phase
###
### v4 -This version, takes frames at a 4X frame rate and then it samples 1 out of every 4 frames to find
### sampled frames at 1X frame rate. Then it interpolates these frames to a 4X frame rate to create sampled interpolated frames.
### Then it uses these sampled interpolated frames to find the representative heartbeat .Then it phase stamps the sampled frames
### to the representative frames found from the sampled interpolated frames. Then it also creates representative frames from
### the original frames by just chosing the frames with the same frame number as the representative frames
### did in the sampled interpolated frames. It assumes that the sampled interpolated frames and the original frames are essentially the same.
### Then it phase stamps the sampled frames using the original frames as well.
###############################################################################
###############################################################################


#%%
#from __future__ import division
from os import makedirs
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skimage.external.tifffile as tff
plt.close("all")

#import sys
#sys.path.append('C:\\Users\\saram_000\\Documents\\WinPython-64bit-2.7.9.2\\python-2.7.9.amd64\\Lib\\site-packages\\libtiff'
tic = time.time()   # start counting time

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%%% This program has been written by Sara Madaan   %%%')
print('%%% Contact smadaan@usc.edu for questions       %%%')
print('%%%')
print('%%% This program is for 4D reconstruction using ZookZik')
print('%%%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%%% v4 -This version, takes frames at a 4X frame rate and then it samples 1 out of every 4 frames to find %%%')
print('%%% sampled frames at 1X frame rate. Then it interpolates these frames to a 4X frame rate to create sampled interpolated frames. %%%')
print('%%% Then it uses these sampled interpolated frames to find the representative heartbeat .Then it phase stamps the sampled frames  %%%')
print('%%% to the representative frames found from the sampled interpolated frames. Then it also creates representative frames from  %%%')
print('%%% the original frames by just chosing the frames with the same frame number as the representative frames %%%')
print('%%% did in the sampled interpolated frames. It assumes that the sampled interpolated frames and the original frames are essentially the same.  %%%')
print('%%% Then it phase stamps the sampled frames using the original frames as well.  %%%')
print('%%% %%%')
print('%%% %%%')




## Step 01 : Take the inputs for the data set


location='I:/160719_PhaseStamping_PhaseSupersamplingTest/PhaseStamping_Supersampling_480fps_54hpf_50Kframes_copy4'
#this is the location where all the code and the files are saved
folder_reconstructedimages = '/ReconstructedImages'
#this is the folder where the reconstructed fluorescent images are saved
newpath_reconstructedimages = location + folder_reconstructedimages
#this is the path to the folder for reconstructed images
makedirs(newpath_reconstructedimages)
#this line creates this folder. So you must make sure that this folder doesn't
#already exist
## important - add a test to check if the folder already exists and alert the
#user

folder_otherdata = '/Figures'
# this is the folder whee the figures and the excel sheets are saved
newpath_otherdata = location + folder_otherdata
#this is the path to the folder_otherdata
makedirs(newpath_otherdata)
#this line creates this folder. So you must make sure that this folder doesn't
#already exist
## important - add a test to check if the folder already exists and alert the
#user
folder_BF = 'Phase_seq'
prefix_BF = 'FRAMEX'    # this is the name of the Brightfeild tiff file
folderpath_BF = location + '/' + folder_BF
index_start_number_BF = 0
num_digits_BF = 5
total_frames_BF = 50000
#frame = importtiff(folderpath_BF,frame_number,prefix='FRAMEX',index_start_number=1,num_digits=5)

name_kymo_temp ='kymo_final_XZ_41.tif' # this is the name of the kymograph image
name_kymo = location + '/' + name_kymo_temp

folder_FL = 'Fluorescent_Side1_crop_seq_bfr'
prefix_FL = 'FRAMEX'    # this is the name of the Brightfeild tiff file
folderpath_FL = location + '/' + folder_FL
index_start_number_FL = 0
num_digits_FL = 5
total_frames_FL = 50000

format_all_files = 'tif' # this is the format for all the image files we use


###############################################################################
###############################################################################

TotalBFimages = total_frames_BF # total number of BF images is equal to the
# zeroth variable in the shape=BF_image variable
Tstart = index_start_number_BF                # this needs to be described
Tend = TotalBFimages     # this needs to be described

framestobedisplayed = TotalBFimages  # this is the number of frames to be
#displayed in the first stage motion plot. One can chose to see fewer
#time points int he stage motion plot if they want to zoom into the plot better

ignore_zooks_at_start = 0  ##this the number of zooks in the beginning of the image collection that are to be ignored
ignore_startzook = 0 #clarify what this is
ignore_endzook = 0  #clarify what this is
ignore_end_waveletframe = 10
BF_resolution = 0.6296 #clarify what this is
BF_fps = 480    #clarify what this is
FL_fps = 480    #clarify what this is
stepsizeforBFvsFL = 1   # This can be 1, 2 or 3 #clarify what this is
numberofframesatbeginningforcoordination = 0 # This can be 0 or 1 #clarify what
# this is

FirstMaximapoint = 1199  # this is the first maxima point for the stage motion
# this means tha if we acquire 130 frames per zook, then the first maxima point
#is 130
FirstMinimapoint = 0 #this is the first minima point. This tells us which was
#the first frame acquired in the first zook . here it is frame number 1
ZookZikPeriod = 1200 # number of frames in a zook

xlsname_zstamp = location+folder_otherdata+'/'+folder_BF+'_ZSTAMP.xls' # Name of the
                 # excelsheet to store intermediate results

xlsname_pstamp = location+folder_otherdata+'/'+folder_BF+'_PSTAMP.xls' # % Name of the
                 #excelsheet to store the final zp tables
xlsname_zpstamp = location+folder_otherdata+'/'+folder_BF+'_ZPTable.xls' #Name of the
                  #excelsheet to store the final zp tables
xlsname_bestbrothers = location+folder_otherdata+'/'+folder_BF+'_BestBrothers.xls' #Name of
                       #the excelsheet to store the final zp tables
figurenames = location+folder_otherdata+folder_BF +'_'
              # Starting part of the name of figure

resampling_factor = 5 ##The resampling fator for this code cannot eb changed and must stay fixed at 5
time_resampling_factor = 4

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
#%%

#Step 2


tic = time.time()   # start counting time

#from __future__ import division
from os import makedirs
import numpy as np
from scipy.signal import argrelextrema
from scipy import stats
import math
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skimage.external.tifffile as tff
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('            ')
print(' Step 02 : Finding Turning points started')


#tiff_kymo = tff.imread(name_kymo) # import the kymograph tif file
#kymo_img = np.asarray(tiff_kymo) # create a numpy 3D array using the
##kymograph tif file
#plt.figure(250)
#plt.imshow(kymo_img, cmap=plt.cm.gray, hold = 'on')
#
#shape_kymo_img = np.asarray(np.shape(kymo_img))
#
#matchingvalue =0; #this is the value for the whitest pixel

## Step 2.1 : Find the last (rightmost) white pixel in every row of
## kymograph
#
shape_kymo_img = np.asarray([total_frames_BF,1])
z_stage = np.zeros(shape_kymo_img[0])
#for i in range(shape_kymo_img[0]):
#    k = kymo_img[i,:]
#    a = np.where(k == matchingvalue)
#    a = np.asarray(a)
#    z_stage[i]=a[0,-1] #This denotes the pixel index of the rightmost
#    #black pixel in kymograph
#
#
### Step 2.2 : Find the maxima points
#
## maximapoints = FirstMaximapoint:ZookZikPeriod:size(A,1);
## m = z_stage(maximapoints);
#frame_number = np.zeros(framestobedisplayed)
#for x in range(framestobedisplayed):
#    frame_number[x] = x
#
##
##maximapoints = argrelextrema(z_stage, np.greater)
##maximapoints = np.asarray(maximapoints)
##maximapoints = maximapoints.transpose()
##maximapoints = maximapoints.flatten()
#maximapoints = np.transpose(np.asarray(np.where((z_stage[1:] - z_stage[:57695])<-10)))
#maximapoints = maximapoints.flatten()
#m = z_stage[maximapoints]
#plt.figure(251)
#plt.hold(True)
#plt.plot(frame_number,z_stage[0:framestobedisplayed]) #,'b','linewidth',2)
##AxesLabel( 'Frame Number','z-stage motion (in pixels)','Zook-Zik from kymograph')




'''
# for local maxima
argrelextrema(z_stage, np.greater)

# for local minima
argrelextrema(x, np.less)
'''




##
# added by Sara 2015_01_23
# displays the zook zik curve separately, so that you can see the bad
# frames more clearly and zoom in better



## Step 2.3 : Find the minima points, the first frame for reference and
## display them for first framestobedisplayed number of frames


FirstRefFrame = 0 + ignore_startzook + (ignore_zooks_at_start*ZookZikPeriod)

#minimapoints = np.asarray(FirstMinimapoint) # This is assuming
## that the start of BF image sequence oincides with start of zook
#minimapoints = minimapoints.astype(np.int64)
#lo = np.asarray([z_stage[FirstMinimapoint]])
#for i in range(len(maximapoints)-1):
##    print(i)
#    g = np.argmin(z_stage[maximapoints[i]:maximapoints[i+1]])
##    print(g)
#    minimapoints = np.append(minimapoints,maximapoints[i]+g)
#
#    n = z_stage[minimapoints[-1]]
#    lo = np.append(lo,n)
#
#    if maximapoints[i] < framestobedisplayed:
#        plt.plot(maximapoints[i],m[i],'ro')
#        plt.hold(True)
#
#    if minimapoints[i] < framestobedisplayed:
#        plt.plot(minimapoints[i],lo[i],'bo')
#        plt.hold(True)
#
#
#
#plt.hold(False)

### Step 2.4 :Find the statistics on zook and zik lengths and plot
### normalized values for all based on the average
#averagezik = np.round(np.mean(minimapoints[1:len(minimapoints)-1] -
#                maximapoints[0:len(minimapoints)-2]))
#
#
#averagezook = np.round(np.mean(maximapoints[0:len(maximapoints)] -
#                    minimapoints[0:len(maximapoints)]+1))
#
#
#modezik = np.round(stats.mode(minimapoints[1:len(minimapoints)-1] -
#                maximapoints[0:len(minimapoints)-2]))
#
#
#modezook = np.round(stats.mode(maximapoints[0:len(maximapoints)] -
#                minimapoints[0:len(maximapoints)]+1))
#
#maxzik = np.max(minimapoints[1:len(minimapoints)-1] -
#                maximapoints[0:len(minimapoints)-2])
#
#maxzook = np.max(maximapoints[0:len(maximapoints)] -
#                minimapoints[0:len(maximapoints)]+1)
#
#
#some_value = minimapoints[1:len(minimapoints)-1] - \
#                maximapoints[0:len(minimapoints)-2]
#av_some_value = some_value/averagezik
#
#
#
#
#another_value = maximapoints[0:len(maximapoints)] - \
#                minimapoints[0:len(maximapoints)]+1
#av_another_value = another_value/averagezook
#

#x_axis1 = np.array([])
#x_axis2 = np.array([])
#for x in range(len(minimapoints)-2):
#    x_axis1 = np.append(x_axis1,x)
#for x in range(len(maximapoints)):
#    x_axis2 = np.append(x_axis2,x)
#
#

#plt.figure(252)
#plt.hold(True)
#plt.plot(x_axis1,av_some_value,'ro-.',linewidth=2)
#plt.plot(x_axis2,av_another_value,'go-.',linewidth=2)
##AxesLabel( 'Stage Ramping index ','Normalized lengths',strcat('<zook> = ',num2str(averagezook), ';  <zik> = ',num2str(averagezik)))
#plt.hold(False)
#
#
#


## Step 2.5 : Find the relative z_stage movement and z stamp excluding the
## frames to be ignored in the zook

zstamp_res_adjusted = np.zeros((shape_kymo_img[0],1))  #This is to store
# the relative z_stage movement excluding the frames to be ignored in the zook
zstamp = np.zeros((shape_kymo_img[0],1))  #This is to store the actual
# stamp of the frames

#for i in range(len(minimapoints)):# goes from 0:length(minimapoints)-2
#    zstamp[minimapoints[i]+
#        ignore_startzook : maximapoints[i]-ignore_endzook+1,0] = \
#        z_stage[minimapoints[i]+ ignore_startzook : maximapoints[i]-
#          ignore_endzook+1] - z_stage[0]*np.ones((maximapoints[i]
#                    -ignore_endzook) - (minimapoints[i]+ ignore_startzook)+1)
#
#    zstamp_res_adjusted[minimapoints[i]+
#       ignore_startzook : maximapoints[i]-ignore_endzook+1] = \
#      np.round((zstamp[minimapoints[i]+ ignore_startzook : maximapoints[i]-
#                 ignore_endzook+1])*BF_resolution)
#

##### create determiistic z_stamp
## create deterministic minima and maximapoints


minimapoints_det = np.arange(0,np.floor(total_frames_BF/ZookZikPeriod))*ZookZikPeriod
maximapoints_det = minimapoints_det + ZookZikPeriod  - 1
zstamp_det = np.zeros((shape_kymo_img[0],1))  #This is to store the actual
# stamp of the frames
zstamp_det_unrounded = np.zeros((shape_kymo_img[0],1))  #This is to store the actual
# stamp of the frames
zstamp_det_resized = np.zeros((shape_kymo_img[0],1))

minimapoints = minimapoints_det.astype(int)
maximapoints = maximapoints_det.astype(int)
#pixelengthzook = stats.mode(z_stage[maximapoints]-z_stage[minimapoints])
#zooklengthadjustment = (np.max(z_stage[maximapoints]-z_stage[minimapoints])-np.min(z_stage[maximapoints]-z_stage[minimapoints])) # always
##check to make sure that the zebrafish bondary doesn't move much between the
##last frame of one zook and first frame of next zook
#pixelengthzook = pixelengthzook[0]
#pixelstart = stats.mode(z_stage[minimapoints])
#pixelstart = pixelstart[0]
#pixelmotionperframe = pixelengthzook/ZookZikPeriod
#
#pixelmotionperframe = pixelengthzook/ZookZikPeriod
#for i in range(len(minimapoints)):# goes from 0:length(minimapoints)-2
#    zstamp_det[minimapoints_det[i]: maximapoints_det[i]+1,0] = \
#        np.round(np.arange(0,ZookZikPeriod)*pixelmotionperframe)
#
#
#for i in range(len(minimapoints)):# goes from 0:length(minimapoints)-2
#    zstamp_det_unrounded[minimapoints_det[i]: maximapoints_det[i]+1,0] = \
#        np.arange(0,ZookZikPeriod)*pixelmotionperframe
#
#for i in range(len(minimapoints)):# goes from 0:length(minimapoints)-2
#    zstamp_det_resized[minimapoints_det[i]: maximapoints_det[i]+1,0] = \
#        np.round((np.arange(0,ZookZikPeriod)*pixelmotionperframe)*resampling_factor)
#
#
##    zstamp_res_adjusted[minimapoints[i]+
##       ignore_startzook : maximapoints[i]-ignore_endzook+1] = \
##      np.round((zstamp[minimapoints[i]+ ignore_startzook : maximapoints[i]-
##                 ignore_endzook+1])*BF_resolution)
##

#pixels_to_move = 5
#num_steps = int(pixels_to_move*resampling_factor)

##create zstamp moved by 1 pixel to the right
#zstamp_det_resized_p1 = zstamp_det_resized + num_steps
#
##create zstamp moved by 1 pixel to the left
#zstamp_det_resized_m1 = zstamp_det_resized - num_steps

#
#x_axis3 = np.array([])
#x_axis4 = np.array([])
#for x in range(len(zstamp_res_adjusted)):
#    x_axis3 = np.append(x_axis3,x+1)
#for x in range(len(zstamp)):
#    x_axis4 = np.append(x_axis4,x+1)
#
#plt.figure(253)
#plt.plot(x_axis3,zstamp_res_adjusted,'b',linewidth=1)
##axis tight, AxesLabel( 'Frame number ','z-stamp (\mum)','pixels moved in kymograph')
#
#plt.figure(254)
#plt.plot(x_axis4,zstamp,'r',linewidth=1)
#
##AxesLabel( 'Frame number ','z-stamp (pixel)',strcat('After ignoring ',num2str(ignore_startzook),' frames at start & ',num2str(ignore_endzook),' frames at end' ))
#
#
###Step 2.6 : Store the relevant statistics
##Sheet 1 : Store the z_stage movement
##Sheet 2 : maxima points (end of zook)
##Sheet 3 : minima points (start of zook)
##Sheet 4 : zstamp_res_adjusted
##Sheet 5 : zstamp
#
##xlswrite(xlsname_zstamp,z_stage,1);
##xlswrite(xlsname_zstamp,maximapoints,2);
##xlswrite(xlsname_zstamp,minimapoints,3);
##xlswrite(xlsname_zstamp,zstamp_res_adjusted,4);
##xlswrite(xlsname_zstamp,zstamp,5);
#
### NEEDS TO BE REWRITTEN savefig(strcat(figurenames,'Step02_TurningPoints.fig'));
#
#print('            ')
#print('average # zook frames is  :' + str(averagezook))
#print('average # zik frames is  :' + str(averagezik))
#print('mode # zook frames is  :' + str(modezook))
#print('mode # zik frames is  :' + str(modezik))
#print('maximum # zook frames is  :' + str(maxzook))
#print('maximum # zik frames is  :' + str(maxzik))
print('            ')
print('For the step number 2')
toc = time.time()
print(toc-tic)
print('            ')
print('##################################################')








##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
#%%

#Step 3

tic = time.time()   # start counting time

#from __future__ import division
from os import makedirs
import numpy as np
from scipy.signal import argrelextrema
from scipy import stats,misc
import math
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skimage.external.tifffile as tff
import pandas
from skimage.transform import resize
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('            ')
print(' Step 03 : Import drawing box and mask for images')




#rectfile_name = 'rectfile.csv'
#rect_file_path = location + '/'+ rectfile_name
#
#rect_file = pandas.read_csv(rect_file_path)
#x_start = rect_file['Var2'][0]-1  # the -1 is because there is a one pixel shift from matlab to python
#x_end = rect_file['Var2'][1]-1  # the -1 is because there is a one pixel shift from matlab to python
#y_end = rect_file['Var2'][2]-1   # the -1 is because there is a one pixel shift from matlab to python
#width = rect_file['Var2'][3]
#y_left = y_end - width
#
#
#
#
#x_start_resized = x_start*resampling_factor
#x_end_resized = x_end*resampling_factor
#y_end_resized = y_end*resampling_factor
#width_resized = width*resampling_factor
#y_left_resized = y_end_resized - width_resized
#
#


mask_file_name = 'mask.tif'
mask_file_path = location + '/' + mask_file_name

mask = tff.imread(mask_file_path)
mask_size = mask.shape
#mask_resized = resize(mask, (mask_size[0]*resampling_factor,mask_size[1]*resampling_factor),preserve_range=True)
#mask_resized[mask_resized>0] = 1

#arr[arr > 255] = x
print('For the step number 3')
toc = time.time()
print(toc-tic)
print('            ')
print('##################################################')

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
##%%
#
##Step 3.1
#
#tic = time.time()   # start counting time
#
##from __future__ import division
#from os import makedirs
#import numpy as np
#from scipy.signal import argrelextrema
#from scipy import stats,misc
#import math
#import matplotlib.pyplot as plt
#import time
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import skimage.external.tifffile as tff
#import pandas
#from skimage.transform import resize
#print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#print('            ')
#print(' Step 03.1 : Import drawing box and mask for zstamping images')
#
#rectregfile_name = 'rectfilereg.csv'
#rectreg_file_path = location + '/'+ rectregfile_name
#
#rectreg_file = pandas.read_csv(rectreg_file_path)
#x_start_reg = rectreg_file['Var2'][0]-1  # the -1 is because there is a one pixel shift from matlab to python
#x_end_reg = rectreg_file['Var2'][1]-1  # the -1 is because there is a one pixel shift from matlab to python
#y_end_reg = rectreg_file['Var2'][2]-1   # the -1 is because there is a one pixel shift from matlab to python
#width_reg = rectreg_file['Var2'][3]
#y_left_reg = y_end_reg - width_reg
#
#
#
#x_start_reg_resized = x_start_reg*resampling_factor
#x_end_reg_resized = x_end_reg*resampling_factor
#y_end_reg_resized = y_end_reg*resampling_factor
#width_reg_resized = width_reg*resampling_factor
#y_left_reg_resized = y_end_reg_resized - width_reg_resized
#
#
##arr[arr > 255] = x
#print('For the step number 3.1')
#toc = time.time()
#print(toc-tic)
#print('            ')
#print('##################################################')
#
#
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
#%%
# Step03: Z stamp using reference ROI
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('            ')
print(' Step 03.2 : Z stamp using reference ROI')

#from func_3_2 import find_zstamp_corr

#zstamp_corr = find_zstamp_corr(location,x_start_reg,x_end_reg,y_end_reg,width_reg,y_left_reg,zstamp_det,minimapoints,ignore_startzook,maximapoints,ignore_endzook)
tic = time.time()   # start counting time

#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skimage.external.tifffile as tff
from tiffseriesimport import importtiff, writetiff
from PIL import Image
from skimage.transform import resize
from img_proccessing import corr2, ssim2, corr2_masked, ZookZik_1X_v1_MaximumCorrelationCluster, ZookZik_1X_v1_MaximumCorrelationCluster_vectorized
from filter_function import butter_lowpass_filtfilt
from medpyimage import mutual_information, mutual_information_normalized_masked
import pywt


actual_frame_index = np.arange(0,total_frames_BF)
new_frame_index = np.arange(0,total_frames_BF)
new_frame_index[:] = np.NAN
new_avg_frame_index = np.arange(0,total_frames_BF)
new_avg_frame_index[:] = np.NAN
new_frame_index_for_j = np.arange(0,total_frames_BF)
new_frame_index_for_j[:] = np.NAN

folder_BF_resampled = 'Phase_Side1_crop_seq_bfr_resampled'
prefix_BF_resampled = 'FRAMEX'    # this is the name of the Brightfeild tiff file
folderpath_BF_resampled = location + '/' + folder_BF_resampled
makedirs(folderpath_BF_resampled)

####################


newfolder_wavelet = location+folder_otherdata+'/StageMotionCorrectedWavelet'
makedirs(newfolder_wavelet)
newname= 'FRAMEX' # Name of the stage motion corrected new image data


newfolder_avg = location+folder_otherdata+'/TimeAveraged'
makedirs(newfolder_avg)

newfolder_avg_interpolated = location+folder_otherdata+'/TimeAveragedInterpolated'
makedirs(newfolder_avg_interpolated)




num_zooks_for_test = 3
num_zooks_for_numrepframes = 24


####################

#A = importtiff(folderpath_BF,FirstRefFrame,prefix=prefix_BF,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
#A_ref = A[x_start:x_end,y_left:y_end]


goodzookendframenumber = np.array([]) #This stores the framenumber where the good zook frames end in the new image
goodzookendframenumber_avg = np.array([])
goodzookendframenumber_for_numrepframes = np.array([])
countframesinnewimage = 0
counttotalframesinnewimage = 0
counttotalframesinnewimage_2 = 0
countframesinnewimage_for_numrepframes = 0
countframesinZ2 = 0

A1 = importtiff(folderpath_BF,FirstRefFrame,prefix=prefix_BF,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
A1_size = A1.shape
#A1_resized =resize(A1, (A1_size[0]*resampling_factor,A1_size[1]*resampling_factor),preserve_range=True)


(c1A1, (c1H1, c1V1, c1D1)) = pywt.dwt2(A1,'db1', mode='sym')
Z = np.zeros(((c1A1.shape[0],c1A1.shape[1],ZookZikPeriod)))
#y_right_reg_resized = np.absolute(y_end_reg_resized+ zstamp_det_resized[FirstRefFrame])
#y_left_reg_resized = np.absolute(y_right_reg_resized - width_reg_resized)
##A1slide = A1[x_start_reg:x_end_reg+1,y_left_reg:y_right_reg+1]*mask

#A1slide_resized_unmasked = A1_resized[x_start_reg_resized:x_end_reg_resized+1,y_left_reg_resized:y_right_reg_resized+1]


#slide_range = 5  ##always check if the calc_shift stays within the slide range and doesn't go to the edges of the slide range
#slide_extent=slide_range*resampling_factor
#corMat_coarse = np.zeros((total_frames_BF,2*slide_extent+1))
#si = np.array([]) #SI = similarity index, Corr, or MI or SADWaC

#calc_shift =  np.zeros(shape_kymo_img[0])
#calc_shift_resampled =  np.zeros(shape_kymo_img[0])
zstamp_corr =  np.zeros(shape_kymo_img[0])
#zstamp_corr_resized =  np.zeros(shape_kymo_img[0])
#best_zstamp_corr = np.zeros(shape_kymo_img[0])
#zstamp_corr_resized_p1 =  np.zeros(shape_kymo_img[0])
#zstamp_corr_resized_m1 =  np.zeros(shape_kymo_img[0])

#for i in range(len(minimapoints)): # This is to just use apx ~0.15% of the zooks for storing reference images
#for i in range(num_zooks_for_test): # This is to just use apx ~0.15% of the zooks for storing reference images
for i in np.arange(ignore_zooks_at_start,len(minimapoints)):
    print(i)
    countframesinZ = 0

    countframesinZ2_resampled = 0
    countframesinthiszook = 0
    for framenumber in np.arange(minimapoints[i]+ignore_startzook,maximapoints[i]-ignore_endzook+1):

#        print('framenumber is', framenumber)
        A = importtiff(folderpath_BF,framenumber,prefix=prefix_BF,index_start_number=index_start_number_BF,num_digits=num_digits_BF)

        Aslide_resized = A*mask # A_slide
        # contains the multiplication of the image in the sliding window
        # with mask that was drawn by the user
        (c1A1, (c1H1, c1V1, c1D1)) = pywt.dwt2(Aslide_resized,'db1', mode='sym')
        Z[:,:,countframesinthiszook] = c1A1
        new_frame_index_for_j[countframesinthiszook] = framenumber
        countframesinthiszook = countframesinthiszook + 1

        if(framenumber>minimapoints[i]+ignore_startzook+ignore_end_waveletframe-1 and framenumber<maximapoints[i]-ignore_endzook+1-ignore_end_waveletframe):
#            print(framenumber)
#            print(countframesinthiszook-1)
#            print(counttotalframesinnewimage)
            c1A1_round = np.round(c1A1)
            c1A1_int = c1A1_round.astype(int)


            if(i < num_zooks_for_numrepframes+ignore_zooks_at_start):
                writetiff(c1A1_int,newfolder_wavelet,counttotalframesinnewimage,prefix=newname,index_start_number=0,num_digits=5)

            new_frame_index[framenumber] = counttotalframesinnewimage

            counttotalframesinnewimage = counttotalframesinnewimage+1


            if(i<num_zooks_for_numrepframes+ignore_zooks_at_start):
                countframesinnewimage_for_numrepframes = countframesinnewimage_for_numrepframes +1

            if(i<num_zooks_for_test+ignore_zooks_at_start):
                countframesinnewimage = countframesinnewimage +1 # this keeps count
                # of the numbr of images in the file that will be stored as newname
#        print(countframesinnewimage)


    Z = Z[:,:,:countframesinthiszook]
    Z2 = np.zeros(((Z.shape[0],Z.shape[1],int(Z.shape[2]/4))))

    for j in np.arange(0,Z.shape[2],time_resampling_factor):
#    print(i)
#    B3 = (Z[:,:,i] + Z[:,:,i+1] + Z[:,:,i+2] + Z[:,:,i+3])/4
#    Z2[:,:,int(i/4)] = B3

        B3 = Z[:,:,j+1]
        Z2[:,:,int(j/4)] = B3

        new_avg_frame_index[countframesinZ2] = new_frame_index_for_j[j+1] #stores at jth position the actual frame number in the original file that this average frame corrresponds to

        B3_round = np.round(B3)
        B3_int =B3_round.astype(int)
        writetiff(B3_int,newfolder_avg,countframesinZ2,prefix=newname,index_start_number=0,num_digits=5)
        countframesinZ2 = countframesinZ2 +1
#    corr_A3 = corr2_masked(A3,B3,mask)

#    corr_A3_vect[int(i/4)] = corr_A3

    Z2_size = Z2.shape

    if(i < num_zooks_for_numrepframes+ignore_zooks_at_start):
        Z2_resized =resize(Z2, (Z2_size[0],Z2_size[1],(Z2.shape[2])*time_resampling_factor),order=5,mode='constant',clip=False,preserve_range=True)


        for framenumber in np.arange(minimapoints[i]+ignore_startzook,maximapoints[i]-ignore_endzook+1):

#        print('framenumber is', framenumber)
            if(framenumber>minimapoints[i]+ignore_startzook+ignore_end_waveletframe-1 and framenumber<maximapoints[i]-ignore_endzook+1-ignore_end_waveletframe):
#            print(framenumber)
#            print(countframesinZ2_resampled)
#            print(counttotalframesinnewimage_2)
#        print(framenumber)
                B = Z2_resized[:,:,countframesinZ2_resampled]

                B_round = np.round(B)
                B_int =B_round.astype(int)

                writetiff(B_round,newfolder_avg_interpolated,counttotalframesinnewimage_2,prefix=newname,index_start_number=0,num_digits=5)
                counttotalframesinnewimage_2 =counttotalframesinnewimage_2 + 1

            countframesinZ2_resampled = countframesinZ2_resampled + 1

        #########
#        print('framenumber is ', framenumber)
#        calc_shift[framenumber] = np.argmax(corMat)- slide_extent


    goodzookendframenumber_avg = np.append(goodzookendframenumber_avg,countframesinZ2-1)
    goodzookendframenumber = np.append(goodzookendframenumber,countframesinnewimage-1) # this stores the frame number of the
    # last frame in each of the three zooks that we used in this m file
    goodzookendframenumber_for_numrepframes = np.append(goodzookendframenumber_for_numrepframes,countframesinnewimage_for_numrepframes-1)

#plt.hold(False)
#plt.figure(326)
#plt.hold(True)
#plt.plot(np.arange(0,len(zstamp_corr)),zstamp_corr,c='r')
#plt.plot(np.arange(0,len(zstamp_corr)),zstamp_det_unrounded,c='k')
#plt.hold(False)
#
#plt.figure(327)
#plt.plot(np.arange(0,len(zstamp_corr)),zstamp_det_unrounded[:,0]-zstamp_corr,c='r')
#
#plt.figure(328)
#plt.plot(np.arange(0,len(best_zstamp_corr)),best_zstamp_corr,c='r')
#
#plt.figure(329)
#plt.plot(np.arange(0,len(calc_shift_resampled)),calc_shift_resampled,c='r')
#

#
##arr[arr > 255] = x
print('For the step number 3.2')
toc = time.time()
print(toc-tic)
print('            ')
print('##################################################')

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##%%
#
#
#
#
##Step 3.3
#
#tic = time.time()   # start counting time
#
##from __future__ import division
#from os import makedirs
#import numpy as np
#from scipy.signal import argrelextrema
#from scipy import stats,misc
#import math
#import matplotlib.pyplot as plt
#import time
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import skimage.external.tifffile as tff
#import pandas
#from skimage.transform import resize
#print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#print('            ')
#print(' Step 03.3 : Fit line to the curve')
#
#
#
#
#zstamp_curvefit = np.zeros(shape_kymo_img[0])
#
#
#for i in range(len(minimapoints)): # This is to just use apx ~0.15% of the zooks for storing reference images
#    y1 = zstamp_corr[minimapoints[i]+ignore_startzook:maximapoints[i]-ignore_endzook+1]
#    x1 = np.arange(minimapoints[i]+ignore_startzook,maximapoints[i]-ignore_endzook+1)
#    slope, intercept, r_value, p_value, std_err = stats.linregress(x1,y1)
#    y1_prime = slope*x1 + intercept
#
#    zstamp_curvefit[minimapoints[i]+ignore_startzook:maximapoints[i]-ignore_endzook+1] = y1_prime
#
#
##y = zstamp_corr[0:191]
##x = np.arange(0,191)
##
##slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
##
##y_prime = slope*x + intercept
#
#
#
#plt.hold(False)
#plt.figure(331)
#plt.hold(True)
#plt.plot(np.arange(0,len(zstamp_corr)),zstamp_corr,c='r')
#plt.plot(np.arange(0,len(zstamp_corr)),zstamp_curvefit,c='k')
#plt.hold(False)
#
#
#plt.figure(332)
#plt.plot(np.arange(0,len(zstamp_corr)),zstamp_curvefit-zstamp_corr,c='r')
#
###histogram
#zstamp_diff = zstamp_curvefit-zstamp_corr
#
#plt.figure(333)
#plt.hist(zstamp_diff, bins = (np.arange(-5.1,5.1,0.2)))
#
#
#plt.figure(334)
#plt.hist(zstamp_diff, bins = (np.arange(-5.05,5.05,0.1)))
#
#mean_zstamp_diff = np.mean(zstamp_diff)
#std_zstamp_diff = np.std(zstamp_diff)
#
#
##arr[arr > 255] = x
#print('For the step number 3.3')
#toc = time.time()
#print(toc-tic)
#print('            ')
#print('##################################################')
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
##%%
#
##Step 4
#
#tic = time.time()   # start counting time
#
##from __future__ import division
#from os import makedirs
#import numpy as np
#from scipy.signal import argrelextrema
#from scipy import stats
#import math
#import matplotlib.pyplot as plt
#import time
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import skimage.external.tifffile as tff
#import pandas
#from tiffseriesimport import importtiff, writetiff
#print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#print('            ')
#print(' Step 04 : Stage Motion Subtraction Started')
#
#
#
#folder_BF = 'Phase_Side1_crop_seq_bfr'
#prefix_BF = 'FRAMEX'    # this is the name of the Brightfeild tiff file
#folderpath_BF = location + '/' + folder_BF
#index_start_number_BF = 0
#num_digits_BF = 5
#total_frames_BF = 57696
#
#
#A = importtiff(folderpath_BF,FirstRefFrame,prefix=prefix_BF,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
#A1_size = A1.shape
#A1_resized =resize(A1, (A1_size[0]*resampling_factor,A1_size[1]*resampling_factor),preserve_range=True)
#y_right_resized = np.absolute(y_end_resized+ zstamp_corr_resized[ref_frame])
#y_left_resized = np.absolute(y_right_resized - width_resized)
#A_ref = A[x_start_resized:x_end_resized,y_left_resized:y_end_resized]
#
#newfolder = location+folder_otherdata+'/StageMotionCorrected'
#makedirs(newfolder)
#newname= 'FRAMEX' # Name of the stage motion corrected new image data
#newfolder_unmasked = location+folder_otherdata+'/StageMotionCorrected_Unmasked'
#makedirs(newfolder_unmasked)
#
#goodzookendframenumber = np.array([]) #This stores the framenumber where the good zook frames end in the new image
#countframesinnewimage = 0
#
#num_zooks_for_test = 3
#
#for i in range(num_zooks_for_test): # This is to just use apx ~0.15% of the zooks for storing reference images
#
#    for framenumber in np.arange(minimapoints[i]+ignore_startzook,maximapoints[i]-ignore_endzook+1):
#
#        A = importtiff(folderpath_BF,framenumber,prefix=prefix_BF,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
#        A_size = A.shape
#        A_resized =resize(A, (A_size[0]*resampling_factor,A_size[1]*resampling_factor),preserve_range=True)
#
#
#        y_right_resized = np.absolute(y_end_resized+ zstamp_corr_resized[ref_frame]) #y_right is now the
#        # position of the column number on the right side of the slide
#        # window for the current frame A
#        y_left_resized = np.absolute(y_right_resized - width_resized)  # This decides the position of the
#        # column number on the left side of the slide window for the current
#        # frame A
#        Aslide_resized = A_resized[x_start_resized:x_end_resized+1,y_left_resized:y_right_resized+1]*mask_resized; # A_slide
#        # contains the multiplication of the image in the sliding window
#        # with mask that was drawn by the user
##        fig=plt.figure()
##        if i<201: # This is just to display a few stage motion corrected frames
##            ax=fig.add_subplot(111)
##            ax.imshow(Aslide,cmap = plt.get_cmap('gray'))
##            plt.hold(True)
#        Aslide_unmasked_resized = A[x_start_resized:x_end_resized+1,y_left_resized:y_right_resized+1] # This is to store unmasked images for first 2 zooks
#        writetiff(Aslide_resized,newfolder,countframesinnewimage,prefix=newname,index_start_number=0,num_digits=5)
#
#
#        writetiff(Aslide_unmasked_resized,newfolder_unmasked,countframesinnewimage,prefix=newname,index_start_number=0,num_digits=5)
#        countframesinnewimage = countframesinnewimage +1 # this keeps count
#        # of the numbr of images in the file that will be stored as newname
#        print(countframesinnewimage)
#    goodzookendframenumber = np.append(goodzookendframenumber,countframesinnewimage-1) # this stores the frame number of the
#    # last frame in each of the three zooks that we used in this m file
#
#
#
#print('For the step number 4')
#toc = time.time()
#print(toc-tic)
#print('            ')
#print('##################################################')
#
#
#
#
#




















################################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
#%%

#Step 5

tic = time.time()   # start counting time

#from __future__ import division
from os import makedirs
import numpy as np
from scipy.signal import argrelextrema
from scipy import stats, ndimage, misc, signal
import math
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skimage.external.tifffile as tff
import pandas
from tiffseriesimport import importtiff, writetiff
from img_proccessing import corr2, ssim2, corr2_masked
from filter_function import butter_lowpass_filtfilt
from medpyimage import mutual_information
from itertools import cycle

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('            ')
print(' Step 05 : Find mean period Started')



numberof_referenceframes = np.ceil(goodzookendframenumber_for_numrepframes[0]/10)
Fre = np.zeros(numberof_referenceframes)
#Fre_SSIM = np.zeros(numberof_referenceframes,)

count_measurements = 0

#goodzookendframenumber = np.zeros(200)
#Cor = []
#SSIM = []
color=iter(plt.cm.rainbow(np.linspace(0,1,int(numberof_referenceframes*10))))


for i in range(int(numberof_referenceframes)):
#for i in np.arange(8,9):
    print(i)
    A1 = importtiff(newfolder_avg_interpolated,i,prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
    start_frame = 0+ignore_startzook
    sizeA1 = A1.shape

#    for k in range(num_zooks_for_test):
    for k in range(1):
        count = 0
        sad = np.array([])
#        SSIM = np.array([])
        for j in np.arange(int(start_frame),int(goodzookendframenumber_for_numrepframes[num_zooks_for_numrepframes-1]+1)):
            A = importtiff(newfolder_avg_interpolated,j,prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            sad =np.append(sad,np.sum(np.absolute(A1-A)))
#            SSIM = np.append(SSIM,mutual_information(A1, A, bins=256))
            count = count+1

        start_frame = goodzookendframenumber_for_numrepframes[k]+1+ignore_startzook

        cutoff = 30
        smooth_sad = butter_lowpass_filtfilt(sad, cutoff, BF_fps)
#        smooth_SSIM = butter_lowpass_filtfilt(SSIM, cutoff, BF_fps)

        f, Pxx = signal.welch(smooth_sad, BF_fps, nperseg=30000)
        Pxx[f<0.3] = 0  ##remove the possibility of less than 1 heartbeats per second.
        ##This means that this algorithm cannot work for phase stamping of hearts beating with less than 2 heartbeats per second
        final_f = BF_fps/f
        max_pxx = np.argmax(Pxx)



#        f_SSIM, Pxx_SSIM = signal.welch(smooth_SSIM, BF_fps)
#        final_f_SSIM = BF_fps/f_SSIM
#        max_pxx_SSIM = np.argmax(Pxx_SSIM[2:])

        c=next(color)

        plt.figure(511)
        plt.plot(final_f,Pxx,c=c)
        plt.hold(True)

#        plt.figure(512)
#        plt.plot(final_f_SSIM[2:],Pxx_SSIM[2:],c=c)
#        plt.hold(True)

    Fre[count_measurements] = final_f[max_pxx]
#    Fre_SSIM[count_measurements] = final_f_SSIM[max_pxx_SSIM+2]

    count_measurements = count_measurements+1


meanperiodframes_sad = np.round(stats.mstats.mode(Fre))[0]
#meanperiodframes_SSIM = np.round(stats.mstats.mode(Fre_SSIM))[0]


print(' The mean period of frames using Corr is')
print(meanperiodframes_sad)

#print(' The mean period of frames using SSIM is')
#print(meanperiodframes_SSIM)

print('For the step number 5')
toc = time.time()
print(toc-tic)
print('            ')
print('##################################################')




#%%



#############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
#%%

#Step 6

tic = time.time()   # start counting time

#from __future__ import division
from os import makedirs
import numpy as np
from scipy.signal import argrelextrema
from scipy import stats, ndimage, misc, signal
import math
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skimage.external.tifffile as tff
import pandas
from tiffseriesimport import importtiff, writetiff
from img_proccessing import corr2, ssim2, corr2_masked, ZookZik_1X_v1_MaximumCorrelationCluster, ZookZik_1X_v1_MaximumCorrelationCluster_vectorized
from filter_function import butter_lowpass_filtfilt
from medpyimage import mutual_information
from itertools import cycle
from multiprocessing import Pool
import copy
from itertools import repeat
#from multicoreSM import multifunc

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('            ')
print(' Step 06 : Finding Representative Images started')


## Section to switch between Corr and MI
meanperiodframes =  meanperiodframes_sad

repfolder_wavelet = location+folder_otherdata+'/Representative Phases Wavelet'
makedirs(repfolder_wavelet)
repname= 'FRAMEX' # Name of the stage motion corrected new image data

repfolder_avg_interpolated = location+folder_otherdata+'/Representative Phases Avg Interpolated'
makedirs(repfolder_avg_interpolated)


## Step 6.1 Calculating the cluster index of images for every continuous set of

start_frame = 0
sizeA1 = np.asarray(np.shape(importtiff(newfolder_avg_interpolated,0,prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)))
#sizeA1 is the size of the first frame in the stage motion corrected image file


SadMatrix = np.zeros((countframesinnewimage,countframesinnewimage))

for i in np.arange(0,countframesinnewimage):
#    print(i)
#    for j in range(num_zooks_for_test): #For 3 zooks
    for k in np.arange(start_frame,goodzookendframenumber[num_zooks_for_test-1]+1):
#    for k in np.arange(start_frame,goodzookendframenumber[2-1]+1):

        if(k>=i):
            Z1 = importtiff(newfolder_avg_interpolated,int(k),prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            Z2 = importtiff(newfolder_avg_interpolated,int(i),prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            sadval = np.sum(np.absolute(Z1-Z2))
            SadMatrix[k,i] = sadval
            SadMatrix[i,k] = sadval

##
mean_clustered_frames = np.ones(goodzookendframenumber[num_zooks_for_test-1]-meanperiodframes+2-start_frame)*meanperiodframes
std_clustered_frames = np.ones(goodzookendframenumber[num_zooks_for_test-1]-meanperiodframes+2-start_frame)*meanperiodframes*100
mean2std_clustered_frames = np.zeros(goodzookendframenumber[num_zooks_for_test-1]-meanperiodframes+2-start_frame)

ClusteredFramesAll = []
MatchMatAll = []
for i in range(num_zooks_for_test):    #For 3 zooks
#for i in range(2):
    for j in np.arange(start_frame,goodzookendframenumber[i]-meanperiodframes+2):
#    for j in np.arange(start_frame,start_frame+2):
        Matchingframes = SadMatrix[j:j+meanperiodframes,0:countframesinnewimage]

        ArgMin = np.argmin(Matchingframes,axis=0)

        MatchMat = np.zeros(Matchingframes.shape)

        for k in range(len(ArgMin)):
            # TODO: Add threshold check here for min threshold value of Matchingframes[ArgMin[k],k] < threshold
            MatchMat[ArgMin[k],k] = 1

        ClusteredFrames = np.sum(MatchMat,axis=1)
        mean_clustered_frames[j] = np.mean(ClusteredFrames)
        std_clustered_frames[j] = np.std(ClusteredFrames)
        mean2std_clustered_frames[j] = mean_clustered_frames[j]/std_clustered_frames[j]


        ClusteredFramesAll.append(ClusteredFrames)
        MatchMatAll.append(MatchMat)
    start_frame = goodzookendframenumber[i]+1


ClusteredFramesAll = np.asarray(ClusteredFramesAll)
MatchMatAll = np.asarray(MatchMatAll)

MatchMatAll = MatchMatAll.astype(int)
ClusteredFramesAll = ClusteredFramesAll.astype(int)

MatchMatAll = MatchMatAll.astype('int8')
ClusteredFramesAll = ClusteredFramesAll.astype('int8')

##
## Plotting the results and finding the best representastive image
startframe_for_repimages = np.argmax(mean2std_clustered_frames)

plt.figure(601)
plt.bar(range(len(mean_clustered_frames)),mean_clustered_frames,width=1,bottom=0,yerr=std_clustered_frames,color='y')



#%%
##Store the representative image
j=0
for i in np.arange(startframe_for_repimages,startframe_for_repimages+meanperiodframes):
    R = importtiff(newfolder_wavelet,int(i),prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
    writetiff(R,repfolder_wavelet,j,prefix=repname,index_start_number=0,num_digits=5)
    j = j+1


#%%
##Store the representative image
j=0
for i in np.arange(startframe_for_repimages,startframe_for_repimages+meanperiodframes):
    R = importtiff(newfolder_avg_interpolated,int(i),prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
    writetiff(R,repfolder_avg_interpolated,j,prefix=repname,index_start_number=0,num_digits=5)
    j = j+1

#%%


toc = time.time()
print(toc-tic)
print('            ')
print('##################################################')




############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
#%%

#Step 7

tic = time.time()   # start counting time

#from __future__ import division
from os import makedirs
import numpy as np
from scipy.signal import argrelextrema
from scipy import stats, ndimage, misc, signal
import math
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skimage.external.tifffile as tff
import pandas
from tiffseriesimport import importtiff, writetiff
from img_proccessing_2 import ZookZik_1X_v1_MinimumSADCluster
from filter_function import butter_lowpass_filtfilt
from medpyimage import mutual_information
from itertools import cycle
from multiprocessing import Pool
import copy
from itertools import repeat
#from multicoreSM import multifunc

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('            ')
print(' Step 07 : Phase Stamping for all frames started')


#maskedBFfolder = location+folder_otherdata+'/Masked BF Frames'
#makedirs(maskedBFfolder)
#maskedname= 'FRAMEX' # Name of the stage motion corrected new image data
#
#

#%%
##Read the representative frames from the actual wavelet file
R = np.zeros(((sizeA1[0],sizeA1[1],meanperiodframes)))
for b in np.arange(index_start_number_BF,index_start_number_BF+meanperiodframes):
    R[:,:,b-index_start_number_BF]= importtiff(repfolder_wavelet,int(b),prefix=repname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            #R reads the frame number b from the stagemotioncorrected files.

#%%
##Read the representative frames from the avg interpolated wavelet file
R1 = np.zeros(((sizeA1[0],sizeA1[1],meanperiodframes)))
for b in np.arange(index_start_number_BF,index_start_number_BF+meanperiodframes):
    R1[:,:,b-index_start_number_BF]= importtiff(repfolder_avg_interpolated,int(b),prefix=repname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            #R reads the frame number b from the stagemotioncorrected files.







#%% Phase stamp every single frame
start_frame = 0
pstamp_wav = np.ones(total_frames_BF)*total_frames_BF ##using a large number for frames that will be ignored during phase stamping
pstamp_wav_sad = np.ones(total_frames_BF)*total_frames_BF ##using a large number for frames that will be ignored during phase stamping

pstamp_avg_interpolated = np.ones(total_frames_BF)*total_frames_BF ##using a large number for frames that will be ignored during phase stamping
pstamp_avg_interpolated_sad = np.ones(total_frames_BF)*total_frames_BF ##using a large number for frames that will be ignored during phase stamping


for i in np.arange(ignore_zooks_at_start,len(minimapoints)):
#for i in range(len(minimapoints)):    #For all zooks   imapoints[i]-ignore_endzook):
#for i in range(2):
    for j in np.arange(start_frame,goodzookendframenumber_avg[i]+1):

#    for j in np.arange(minimapoints[i]+ignore_startzook+numberofframesatbeginningforcoordination,max
#        current_frame_number_in_unmasked_file = new_frame_index[j]
        A = importtiff(newfolder_avg,int(j),prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)

#        y_right = np.absolute(y_end + zstamp[j]) #y_right is now the
#        # position of the column number on the right side of the slide
#        # window for the current frame A
#        y_left = np.absolute(y_right - width)  # This decides the position of the
#        # column number on the left side of the slide window for the current
#        # frame A

#        if (y_left>0):
#            size_of_frame = np.asarray(np.shape(A[x_start:x_end+1,y_left:y_right+1]))

#            if (np.array_equal(size_of_frame,np.asarray(np.shape(mask)))== True):
#        Aslide = A[x_start:x_end+1,y_left:y_right+1]*mask # A_slide
#                writetiff(Aslide,maskedBFfolder,j,prefix=maskedname,index_start_number=0,num_digits=5)
        [min_sadframe,min_sadvalue] =  ZookZik_1X_v1_MinimumSADCluster(R,A)
        pstamp_wav[j] = min_sadframe
        pstamp_wav_sad[j] = min_sadvalue

        [min_sadframe,min_sadvalue] =  ZookZik_1X_v1_MinimumSADCluster(R1,A)
        pstamp_avg_interpolated[j] = min_sadframe
        pstamp_avg_interpolated_sad[j] = min_sadvalue

        start_frame = goodzookendframenumber_avg[i]+1



diff1 = pstamp_wav - pstamp_avg_interpolated

slope_pstamp_wav = pstamp_wav[1:] - pstamp_wav[0:len(pstamp_wav)-1]
slope_pstamp_avg_interpolated = pstamp_avg_interpolated[1:] - pstamp_avg_interpolated[0:len(pstamp_avg_interpolated)-1]

slope_pstamp_wav[slope_pstamp_wav<-80] = 10
slope_pstamp_avg_interpolated[slope_pstamp_avg_interpolated<-80] = 10



plt.hold(False)
plt.figure(701)
plt.hold(True)
plt.plot(np.arange(0,len(pstamp_wav)),pstamp_wav,'r')
plt.plot(np.arange(0,len(pstamp_avg_interpolated)),pstamp_avg_interpolated,'b')
plt.hold(False)

plt.hold(False)
plt.figure(702)
plt.hold(True)
plt.plot(np.arange(0,len(pstamp_wav)),pstamp_wav,'r')
plt.hold(False)

plt.hold(False)
plt.figure(703)
plt.hold(True)
plt.plot(np.arange(0,len(pstamp_avg_interpolated)),pstamp_avg_interpolated,'b')
plt.hold(False)


plt.hold(False)
plt.figure(704)
plt.hist(diff1[:j],bins=np.arange(-4.5,5.5,1))
plt.hold(True)


plt.hold(False)
plt.figure(705)
fig, ax = plt.subplots()
ax.scatter(np.arange(0,len(slope_pstamp_wav)),slope_pstamp_wav,c='r')
plt.hold(False)

plt.hold(False)
plt.figure(707)
fig, ax = plt.subplots()
ax.scatter(np.arange(0,len(slope_pstamp_avg_interpolated)),slope_pstamp_avg_interpolated,c ='b')
plt.hold(False)


plt.hold(False)
plt.figure(708)
plt.hist(slope_pstamp_wav[:goodzookendframenumber_avg[len(goodzookendframenumber_avg)-1]],bins=np.arange(-4.5,13.5,1))
plt.hold(True)

plt.hold(False)
plt.figure(709)
plt.hist(slope_pstamp_avg_interpolated[:goodzookendframenumber_avg[len(goodzookendframenumber_avg)-1]],bins=np.arange(-4.5,13.5,1))
plt.hold(True)

#
toc = time.time()
print(toc-tic)
print('            ')
print('##################################################')





##%%
#
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
#################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#####%%
####
#####Step 7
####
####tic = time.time()   # start counting time
####
#####from __future__ import division
####from os import makedirs
####import numpy as np
####from scipy.signal import argrelextrema
####from scipy import stats, ndimage, misc, signal
####import math
####import matplotlib.pyplot as plt
####import time
####from mpl_toolkits.mplot3d import Axes3D
####import matplotlib.pyplot as plt
####import skimage.external.tifffile as tff
####import pandas
####from tiffseriesimport import importtiff, writetiff
####from img_proccessing import corr2, ssim2, ZookZik_1X_v1_MaximumCorrelationCluster, ZookZik_1X_v1_MaximumCorrelationCluster_vectorized, ZookZik_1X_v1_MaximumCorrelationCluster_masked
####from filter_function import butter_lowpass_filtfilt
####from medpyimage import mutual_information
####from itertools import cycle
####from multiprocessing import Pool
####import copy
####from itertools import repeat
####from multicoreSM import multifunc
####
####print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
####print('            ')
####print(' Step 07 : Phase Stamping for all frames started')
####
####
####maskedBFfolder = location+folder_otherdata+'/Masked BF Frames'
####makedirs(maskedBFfolder)
####maskedname= 'FRAMEX' # Name of the stage motion corrected new image data
####
####
#####%%
######Read the representative frames
####R = np.zeros(((sizeA1[0],sizeA1[1],meanperiodframes)))
####for b in np.arange(index_start_number_BF,index_start_number_BF+meanperiodframes):
####    R[:,:,b-index_start_number_BF]= (importtiff(repfolder,int(b),prefix=repname,index_start_number=index_start_number_BF,num_digits=num_digits_BF))*mask
####            #R reads the frame number b from the stagemotioncorrected files.
####
#####%% Phase stamp every single frame
####start_frame = 0
####pstamp = np.zeros(total_frames_BF)
####pstamp_correlation = np.zeros(total_frames_BF)
####for i in range(len(minimapoints)):    #For all zooks
#####for i in range(1):
####    for j in np.arange(minimapoints[i]+ignore_startzook+numberofframesatbeginningforcoordination,maximapoints[i]-ignore_endzook):
#####    for j in np.arange(start_frame,start_frame+2):
####        A = importtiff(folderpath_BF,int(j),prefix=prefix_BF,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
####
####        y_right = np.absolute(y_end + zstamp[j]) #y_right is now the
####        # position of the column number on the right side of the slide
####        # window for the current frame A
####        y_left = np.absolute(y_right - width)  # This decides the position of the
####        # column number on the left side of the slide window for the current
####        # frame A
####
####        if (y_left>0):
####            size_of_frame = np.asarray(np.shape(A[x_start:x_end+1,y_left:y_right+1]))
####
####            if (np.array_equal(size_of_frame,np.asarray(np.shape(mask)))== True):
####                Aslide = A[x_start:x_end+1,y_left:y_right+1]*mask # A_slide
####                writetiff(Aslide,maskedBFfolder,j,prefix=maskedname,index_start_number=0,num_digits=5)
####                [max_correlatedframe,max_correlationvalue] = ZookZik_1X_v1_MaximumCorrelationCluster_masked(R,Aslide,mask)
####                pstamp[j] = max_correlatedframe
####                pstamp_correlation[j] = max_correlationvalue
####
####
####
####toc = time.time()
####print(toc-tic)
####print('            ')
####print('##################################################')
####
####
####
####
####
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#%%

#Step 8

tic = time.time()   # start counting time

#from __future__ import division
from os import makedirs
import numpy as np
from scipy.signal import argrelextrema
from scipy import stats, ndimage, misc, signal
import math
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skimage.external.tifffile as tff
import pandas
from tiffseriesimport import importtiff, writetiff
from img_proccessing import corr2, ssim2, ZookZik_1X_v1_MaximumCorrelationCluster, ZookZik_1X_v1_MaximumCorrelationCluster_vectorized
from filter_function import butter_lowpass_filtfilt
from medpyimage import mutual_information
from itertools import cycle
from multiprocessing import Pool
import copy
#from itertools import repeat
#from multicoreSM import multifunc

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('            ')
print(' Step 08 : ZP table assignment started')


zstamp_new = np.zeros(goodzookendframenumber_avg[len(goodzookendframenumber_avg)-1]+1)


start_frame = 0

for i in np.arange(ignore_zooks_at_start,len(minimapoints)):
#for i in range(len(minimapoints)):    #For all zooks   imapoints[i]-ignore_endzook):
#for i in range(2):
    count_z_stamp_new = 0
    for j in np.arange(start_frame,goodzookendframenumber_avg[i]+1):

        zstamp_new[j] = count_z_stamp_new
        count_z_stamp_new = count_z_stamp_new+1
        start_frame = goodzookendframenumber_avg[i]+1


zpmatrix = np.zeros((goodzookendframenumber_avg[0]+1,meanperiodframes_sad))
zpmatrix_avg_interpolated = np.zeros((goodzookendframenumber_avg[0]+1,meanperiodframes_sad))

for zstamped in np.arange(np.min(zstamp_new),np.max(zstamp_new)):
#    print(zstamped)
    for pstamped in np.arange(0,meanperiodframes_sad):

        count_num_frames = 1
        count_num_frames_avg_interpolated = 1
        for index in range(len(zstamp_new)):

            if ((zstamp_new[index] == zstamped) and (pstamp_wav[index] == pstamped)):

                zpmatrix[zstamped,pstamped] = count_num_frames
                count_num_frames = count_num_frames+1

            if ((zstamp_new[index] == zstamped) and (pstamp_avg_interpolated[index] == pstamped)):

                zpmatrix_avg_interpolated[zstamped,pstamped] = count_num_frames_avg_interpolated
                count_num_frames_avg_interpolated = count_num_frames_avg_interpolated+1


num_frames_per_pstamp = np.sum(zpmatrix,axis=0)
num_frames_per_pstamp_avg_interpolated = np.sum(zpmatrix_avg_interpolated,axis=0)


plt.hold(False)
plt.figure(802)
plt.hold(True)
plt.plot(np.arange(0,len(num_frames_per_pstamp)),num_frames_per_pstamp,'r')
plt.hold(False)


plt.hold(False)
plt.figure(803)
plt.hold(True)
plt.plot(np.arange(0,len(num_frames_per_pstamp_avg_interpolated)),num_frames_per_pstamp_avg_interpolated,'r')
plt.hold(False)

#
toc = time.time()
print(toc-tic)
print('            ')
print('##################################################')


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
