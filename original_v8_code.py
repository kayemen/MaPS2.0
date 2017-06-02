# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 17:20:29 2016

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
### v8 - Uses Correlation, but removes the zeros from the masked mages before comparing them and it uses zstamp_using_correlation for zstamping.
### And it finds the zstamp with +/- 0.2 pixel accuracy by interpolation. And it uses code from Zstamp_using_correlation_supersampled_faster.py
### This version subsamples the supersampled frames after removing the z stage motion, so as to make the phase stamping process faster. And it is
### validated against the phase stamping done with the supersampled frames.
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
#sys.path.append('C:\\Users\\saram_000\\Documents\\WinPython-64bit-2.7.9.2\\python-2.7.9.amd64\\Lib\\site-packages\\libtiff')

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
print('%%% v8 - Uses Correlation, but removes the zeros from the masked mages before comparing them and it uses zstamp_using_correlation for zstamping. %%%')
print('%%% And it finds the zstamp with +/- 0.2 pixel accuracy by interpolation. And it uses code from Zstamp_using_correlation_supersampled_faster.py %%%')
print('%%% This version subsamples the supersampled frames after removing the z stage motion, so as to make the phase stamping process faster. And it is %%%')
print('%%% validated against the phase stamping done with the supersampled frames. %%%')





## Step 01 : Take the inputs for the data set


location='E:/1USC/Dr. Fraser/Projects/Zebrafish Heart Atlas/150428-0502_PSRun/TEST_MISREG_Z_1_PIX_WAVE/150429_3a_homo_54hpf_sample1\Side 1/Recon alg and files/Test1_zstamp_using_corr'
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
folder_BF = 'Phase_Side1_crop_seq_bfr'
prefix_BF = 'FRAMEX'    # this is the name of the Brightfeild tiff file
folderpath_BF = location + '/' + folder_BF
index_start_number_BF = 0
num_digits_BF = 5
total_frames_BF = 57696
#frame = importtiff(folderpath_BF,frame_number,prefix='FRAMEX',index_start_number=1,num_digits=5)

name_kymo_temp ='kymo_final_XZ_41.tif' # this is the name of the kymograph image
name_kymo = location + '/' + name_kymo_temp

folder_FL = 'Fluorescent_Side1_crop_seq_bfr'
prefix_FL = 'FRAMEX'    # this is the name of the Brightfeild tiff file
folderpath_FL = location + '/' + folder_FL
index_start_number_FL = 0
num_digits_FL = 5
total_frames_FL = 57696

format_all_files = 'tif' # this is the format for all the image files we use

'''
###############################################################################
###############################################################################
#This is a way to upload tiff stacks and then save tiff stacks in python
# and here us the weblink
# http://cmci.embl.de/documents/110816pyip_cooking

#i did pip install libtiff but this isn't working
#import libtiff
#t = libtiff.TiffFile(name)
#tt = t.get_tiff_array()

# then I did pip install tifffile and that worked
import tifffile as tff
tiffimg = tff.TIFFfile(name)
img = tiffimg.asarray()
tiffimg
type(img)
img10 = tiffimg[10].asarray()
type(img10)
plt.imshow(img10, cmap=plt.cm.gray, hold = 'on')
## this section above imports a tiff stack and then displays the 11th image
# out of these images
tff.imsave(name_kymo, img10)

#importing single tif file using tifffile library
tiffimg = tff.TIFFfile(name_kymo)
img_new_10 = tiffimg.asarray()

###############################################################################
###############################################################################

toc = time.time()
print toc-tic
'''


'''
info=imfinfo(name);
info_kymo = imfinfo(name_kymo);
info_FL_1 = imfinfo(name_FL_1);
info_FL_2 = imfinfo(name_FL_2);
info_FL_3 = imfinfo(name_FL_3);
'''

#tiffBF = tff.imread(name) # this line imports the brightfiled tif file
#BFimg = tiffBF.asarray() # this line creates a numpy 3D array using the
##brightfeild tif file


#shape_BF_image = np.asarray(np.shape(BFimg))
# this variable stores the shape of the BFimg variable

TotalBFimages = total_frames_BF # total number of BF images is equal to the
# zeroth variable in the shape=BF_image variable
Tstart = index_start_number_BF                # this needs to be described
Tend = TotalBFimages     # this needs to be described

framestobedisplayed = TotalBFimages  # this is the number of frames to be
#displayed in the first stage motion plot. One can chose to see fewer
#time points int he stage motion plot if they want to zoom into the plot better

ignore_zooks_at_start = 2  ##this the number of zooks in the beginning of the image collection that are to be ignored
ignore_startzook = 7#clarify what this is
ignore_endzook = 3  #clarify what this is
BF_resolution = 0.6296 #clarify what this is
BF_fps = 90    #clarify what this is
FL_fps = 90    #clarify what this is
stepsizeforBFvsFL = 1   # This can be 1, 2 or 3 #clarify what this is
numberofframesatbeginningforcoordination = 0 # This can be 0 or 1 #clarify what
# this is

FirstMaximapoint = 191  # this is the first maxima point for the stage motion
# this means tha if we acquire 130 frames per zook, then the first maxima point
#is 130
FirstMinimapoint = 0 #this is the first minima point. This tells us which was
#the first frame acquired in the first zook . here it is frame number 1
ZookZikPeriod = 192 # number of frames in a zook

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


tiff_kymo = tff.imread(name_kymo) # import the kymograph tif file
kymo_img = np.asarray(tiff_kymo) # create a numpy 3D array using the
#kymograph tif file
plt.figure(250)
plt.imshow(kymo_img, cmap=plt.cm.gray, hold = 'on')

shape_kymo_img = np.asarray(np.shape(kymo_img))

matchingvalue =0; #this is the value for the whitest pixel

## Step 2.1 : Find the last (rightmost) white pixel in every row of
## kymograph

z_stage = np.zeros(shape_kymo_img[0])
for i in range(shape_kymo_img[0]):
    k = kymo_img[i,:]
    a = np.where(k == matchingvalue)
    a = np.asarray(a)
    z_stage[i]=a[0,-1] #This denotes the pixel index of the rightmost
    #black pixel in kymograph


## Step 2.2 : Find the maxima points

# maximapoints = FirstMaximapoint:ZookZikPeriod:size(A,1);
# m = z_stage(maximapoints);
frame_number = np.zeros(framestobedisplayed)
for x in range(framestobedisplayed):
    frame_number[x] = x

#
#maximapoints = argrelextrema(z_stage, np.greater)
#maximapoints = np.asarray(maximapoints)
#maximapoints = maximapoints.transpose()
#maximapoints = maximapoints.flatten()
maximapoints = np.transpose(np.asarray(np.where((z_stage[1:] - z_stage[:57695])<-10)))
maximapoints = maximapoints.flatten()
m = z_stage[maximapoints]
plt.figure(251)
plt.hold(True)
plt.plot(frame_number,z_stage[0:framestobedisplayed]) #,'b','linewidth',2)
#AxesLabel( 'Frame Number','z-stage motion (in pixels)','Zook-Zik from kymograph')




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

minimapoints = np.asarray(FirstMinimapoint) # This is assuming
# that the start of BF image sequence oincides with start of zook
minimapoints = minimapoints.astype(np.int64)
lo = np.asarray([z_stage[FirstMinimapoint]])
for i in range(len(maximapoints)-1):
#    print(i)
    g = np.argmin(z_stage[maximapoints[i]:maximapoints[i+1]])
#    print(g)
    minimapoints = np.append(minimapoints,maximapoints[i]+g)

    n = z_stage[minimapoints[-1]]
    lo = np.append(lo,n)

    if maximapoints[i] < framestobedisplayed:
        plt.plot(maximapoints[i],m[i],'ro')
        plt.hold(True)

    if minimapoints[i] < framestobedisplayed:
        plt.plot(minimapoints[i],lo[i],'bo')
        plt.hold(True)



plt.hold(False)

## Step 2.4 :Find the statistics on zook and zik lengths and plot
## normalized values for all based on the average
averagezik = np.round(np.mean(minimapoints[1:len(minimapoints)-1] -
                maximapoints[0:len(minimapoints)-2]))


averagezook = np.round(np.mean(maximapoints[0:len(maximapoints)] -
                    minimapoints[0:len(maximapoints)]+1))


modezik = np.round(stats.mode(minimapoints[1:len(minimapoints)-1] -
                maximapoints[0:len(minimapoints)-2]))


modezook = np.round(stats.mode(maximapoints[0:len(maximapoints)] -
                minimapoints[0:len(maximapoints)]+1))

maxzik = np.max(minimapoints[1:len(minimapoints)-1] -
                maximapoints[0:len(minimapoints)-2])

maxzook = np.max(maximapoints[0:len(maximapoints)] -
                minimapoints[0:len(maximapoints)]+1)


# some_value = lengths of all ziks in the kymo (number of frames)
some_value = minimapoints[1:len(minimapoints)-1] - \
                maximapoints[0:len(minimapoints)-2]
# av_some_value: normalized zik length
av_some_value = some_value/averagezik



# same as above for zook
another_value = maximapoints[0:len(maximapoints)] - \
                minimapoints[0:len(maximapoints)]+1
av_another_value = another_value/averagezook


x_axis1 = np.array([])
x_axis2 = np.array([])
for x in range(len(minimapoints)-2):
    x_axis1 = np.append(x_axis1,x)
for x in range(len(maximapoints)):
    x_axis2 = np.append(x_axis2,x)



plt.figure(252)
plt.hold(True)
plt.plot(x_axis1,av_some_value,'ro-.',linewidth=2)
plt.plot(x_axis2,av_another_value,'go-.',linewidth=2)
#AxesLabel( 'Stage Ramping index ','Normalized lengths',strcat('<zook> = ',num2str(averagezook), ';  <zik> = ',num2str(averagezik)))
plt.hold(False)





## Step 2.5 : Find the relative z_stage movement and z stamp excluding the
## frames to be ignored in the zook

zstamp_res_adjusted = np.zeros((shape_kymo_img[0],1))  #This is to store
# the relative z_stage movement excluding the frames to be ignored in the zook
zstamp = np.zeros((shape_kymo_img[0],1))  #This is to store the actual
# stamp of the frames

for i in range(len(minimapoints)):# goes from 0:length(minimapoints)-2
    zstamp[minimapoints[i]+
        ignore_startzook : maximapoints[i]-ignore_endzook+1,0] = \
        z_stage[minimapoints[i]+ ignore_startzook : maximapoints[i]-
          ignore_endzook+1] - z_stage[0]*np.ones((maximapoints[i]
                    -ignore_endzook) - (minimapoints[i]+ ignore_startzook)+1)

    zstamp_res_adjusted[minimapoints[i]+
       ignore_startzook : maximapoints[i]-ignore_endzook+1] = \
      np.round((zstamp[minimapoints[i]+ ignore_startzook : maximapoints[i]-
                 ignore_endzook+1])*BF_resolution)


##### create determiistic z_stamp
## create deterministic minima and maximapoints


minimapoints_det = np.arange(0,len(minimapoints))*ZookZikPeriod
maximapoints_det = minimapoints_det + ZookZikPeriod  - 1
zstamp_det = np.zeros((shape_kymo_img[0],1))  #This is to store the actual
# stamp of the frames
zstamp_det_unrounded = np.zeros((shape_kymo_img[0],1))  #This is to store the actual
# stamp of the frames
zstamp_det_resized = np.zeros((shape_kymo_img[0],1))
pixelengthzook = stats.mode(z_stage[maximapoints]-z_stage[minimapoints])
zooklengthadjustment = (np.max(z_stage[maximapoints]-z_stage[minimapoints])-np.min(z_stage[maximapoints]-z_stage[minimapoints])) # always
#check to make sure that the zebrafish bondary doesn't move much between the
#last frame of one zook and first frame of next zook
pixelengthzook = pixelengthzook[0]
pixelstart = stats.mode(z_stage[minimapoints])
pixelstart = pixelstart[0]
pixelmotionperframe = pixelengthzook/ZookZikPeriod

pixelmotionperframe = pixelengthzook/ZookZikPeriod

# Not in maps 2
for i in range(len(minimapoints)):# goes from 0:length(minimapoints)-2
    zstamp_det[minimapoints_det[i]: maximapoints_det[i]+1,0] = \
        np.round(np.arange(0,ZookZikPeriod)*pixelmotionperframe)


# Not in maps 2
for i in range(len(minimapoints)):# goes from 0:length(minimapoints)-2
    zstamp_det_unrounded[minimapoints_det[i]: maximapoints_det[i]+1,0] = \
        np.arange(0,ZookZikPeriod)*pixelmotionperframe

# In maps 2
for i in range(len(minimapoints)):# goes from 0:length(minimapoints)-2
    zstamp_det_resized[minimapoints_det[i]: maximapoints_det[i]+1,0] = \
        np.round((np.arange(0,ZookZikPeriod)*pixelmotionperframe)*resampling_factor)


#    zstamp_res_adjusted[minimapoints[i]+
#       ignore_startzook : maximapoints[i]-ignore_endzook+1] = \
#      np.round((zstamp[minimapoints[i]+ ignore_startzook : maximapoints[i]-
#                 ignore_endzook+1])*BF_resolution)
#


#create zstamp moved by 1 pixel to the right
zstamp_det_resized_p1 = zstamp_det_resized + 1

#create zstamp moved by 1 pixel to the left
zstamp_det_resized_m1 = zstamp_det_resized - 1


x_axis3 = np.array([])
x_axis4 = np.array([])
for x in range(len(zstamp_res_adjusted)):
    x_axis3 = np.append(x_axis3,x+1)
for x in range(len(zstamp)):
    x_axis4 = np.append(x_axis4,x+1)

plt.figure(253)
plt.plot(x_axis3,zstamp_res_adjusted,'b',linewidth=1)
#axis tight, AxesLabel( 'Frame number ','z-stamp (\mum)','pixels moved in kymograph')

plt.figure(254)
plt.plot(x_axis4,zstamp,'r',linewidth=1)

#AxesLabel( 'Frame number ','z-stamp (pixel)',strcat('After ignoring ',num2str(ignore_startzook),' frames at start & ',num2str(ignore_endzook),' frames at end' ))


##Step 2.6 : Store the relevant statistics
#Sheet 1 : Store the z_stage movement
#Sheet 2 : maxima points (end of zook)
#Sheet 3 : minima points (start of zook)
#Sheet 4 : zstamp_res_adjusted
#Sheet 5 : zstamp

#xlswrite(xlsname_zstamp,z_stage,1);
#xlswrite(xlsname_zstamp,maximapoints,2);
#xlswrite(xlsname_zstamp,minimapoints,3);
#xlswrite(xlsname_zstamp,zstamp_res_adjusted,4);
#xlswrite(xlsname_zstamp,zstamp,5);

## NEEDS TO BE REWRITTEN savefig(strcat(figurenames,'Step02_TurningPoints.fig'));

print('            ')
print('average # zook frames is  :' + str(averagezook))
print('average # zik frames is  :' + str(averagezik))
print('mode # zook frames is  :' + str(modezook))
print('mode # zik frames is  :' + str(modezik))
print('maximum # zook frames is  :' + str(maxzook))
print('maximum # zik frames is  :' + str(maxzik))
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




rectfile_name = 'rectfile.csv'
rect_file_path = location + '/'+ rectfile_name

rect_file = pandas.read_csv(rect_file_path)
x_start = rect_file['Var2'][0]-1  # the -1 is because there is a one pixel shift from matlab to python
x_end = rect_file['Var2'][1]-1  # the -1 is because there is a one pixel shift from matlab to python
y_end = rect_file['Var2'][2]-1   # the -1 is because there is a one pixel shift from matlab to python
width = rect_file['Var2'][3]
y_left = y_end - width




x_start_resized = x_start*resampling_factor
x_end_resized = x_end*resampling_factor
y_end_resized = y_end*resampling_factor
width_resized = width*resampling_factor
y_left_resized = y_end_resized - width_resized




mask_file_name = 'mask.tif'
mask_file_path = location + '/' + mask_file_name

mask = tff.imread(mask_file_path)
mask_size = mask.shape
mask_resized = resize(mask, (mask_size[0]*resampling_factor,mask_size[1]*resampling_factor),preserve_range=True)
mask_resized[mask_resized>0] = 1

#arr[arr > 255] = x
print('For the step number 3')
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

#Step 3.1

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
print(' Step 03.1 : Import drawing box and mask for zstamping images')

rectregfile_name = 'rectfilereg.csv'
rectreg_file_path = location + '/'+ rectregfile_name

rectreg_file = pandas.read_csv(rectreg_file_path)
x_start_reg = rectreg_file['Var2'][0]-1  # the -1 is because there is a one pixel shift from matlab to python
x_end_reg = rectreg_file['Var2'][1]-1  # the -1 is because there is a one pixel shift from matlab to python
y_end_reg = rectreg_file['Var2'][2]-1   # the -1 is because there is a one pixel shift from matlab to python
width_reg = rectreg_file['Var2'][3]
y_left_reg = y_end_reg - width_reg



x_start_reg_resized = x_start_reg*resampling_factor
x_end_reg_resized = x_end_reg*resampling_factor
y_end_reg_resized = y_end_reg*resampling_factor
width_reg_resized = width_reg*resampling_factor
y_left_reg_resized = y_end_reg_resized - width_reg_resized


#arr[arr > 255] = x
print('For the step number 3.1')
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

actual_frame_index = np.arange(0,total_frames_BF)
new_frame_index = np.arange(0,total_frames_BF)
new_frame_index[:] = np.NAN


folder_BF_resampled = 'Phase_Side1_crop_seq_bfr_resampled'
prefix_BF_resampled = 'FRAMEX'    # this is the name of the Brightfeild tiff file
folderpath_BF_resampled = location + '/' + folder_BF_resampled
makedirs(folderpath_BF_resampled)

####################

folder_BF = 'Phase_Side1_crop_seq_bfr'
prefix_BF = 'FRAMEX'    # this is the name of the Brightfeild tiff file
folderpath_BF = location + '/' + folder_BF
index_start_number_BF = 0
num_digits_BF = 5
total_frames_BF = 57696

newfolder = location+folder_otherdata+'/StageMotionCorrected'
makedirs(newfolder)
newname= 'FRAMEX' # Name of the stage motion corrected new image data
newfolder_unmasked = location+folder_otherdata+'/StageMotionCorrected_Unmasked'
makedirs(newfolder_unmasked)


num_zooks_for_test = 3



####################

#A = importtiff(folderpath_BF,FirstRefFrame,prefix=prefix_BF,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
#A_ref = A[x_start:x_end,y_left:y_end]


goodzookendframenumber = np.array([]) #This stores the framenumber where the good zook frames end in the new image
countframesinnewimage = 0
counttotalframesinnewimage = 0

A1 = importtiff(folderpath_BF,FirstRefFrame,prefix=prefix_BF,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
A1_size = A1.shape
A1_resized =resize(A1, (A1_size[0]*resampling_factor,A1_size[1]*resampling_factor),preserve_range=True)

y_right_reg_resized = np.absolute(y_end_reg_resized+ zstamp_det_resized[FirstRefFrame])
y_left_reg_resized = np.absolute(y_right_reg_resized - width_reg_resized)
#A1slide = A1[x_start_reg:x_end_reg+1,y_left_reg:y_right_reg+1]*mask

A1slide_resized_unmasked = A1_resized[x_start_reg_resized:x_end_reg_resized+1,y_left_reg_resized:y_right_reg_resized+1]


slide_range = 5   ##always check if the calc_shift stays within the slide range and doesn't go to the edges of the slide range
slide_extent=slide_range*resampling_factor
corMat_coarse = np.zeros((total_frames_BF,2*slide_extent+1))
si = np.array([]) #SI = similarity index, Corr, or MI or SADWaC

calc_shift =  np.zeros(shape_kymo_img[0])
calc_shift_resampled =  np.zeros(shape_kymo_img[0])
zstamp_corr =  np.zeros(shape_kymo_img[0])
zstamp_corr_resized =  np.zeros(shape_kymo_img[0])
best_zstamp_corr = np.zeros(shape_kymo_img[0])



#for i in range(len(minimapoints)): # This is to just use apx ~0.15% of the zooks for storing reference images
#for i in range(1): # This is to just use apx ~0.15% of the zooks for storing reference images
for i in np.arange(ignore_zooks_at_start,len(minimapoints)):

    for framenumber in np.arange(minimapoints[i]+ignore_startzook,maximapoints[i]-ignore_endzook+1):

#        print('framenumber is', framenumber)
        A = importtiff(folderpath_BF,framenumber,prefix=prefix_BF,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
        A_size = A.shape
        A_resized =resize(A, (A_size[0]*resampling_factor,A_size[1]*resampling_factor),preserve_range=True)

#        writetiff(A_resized,folderpath_BF_resampled,framenumber,prefix=prefix_BF_resampled,index_start_number=0,num_digits=5)

        # for the actual zstamp
        y_right_reg_resized = np.absolute(y_end_reg_resized + zstamp_det_resized[framenumber]) #y_right is now the
        # position of the column number on the right side of the slide
        # window for the current frame A
        y_left_reg_resized = np.absolute(y_right_reg_resized - width_reg_resized)  # This decides the position of the
        # column number on the left side of the slide window for the current
        # frame A
        #Aslide = A[x_start:x_end+1,y_left:y_right+1]*mask; # A_slide
        # contains the multiplication of the image in the sliding window
        # with mask that was drawn by the user


        for slideamount in np.arange(-slide_extent,slide_extent+1,resampling_factor):
#            print('slideamount is ', slideamount)
            y_right_reg_resized_shift = y_right_reg_resized +slideamount
            y_left_reg_resized_shift = y_left_reg_resized +slideamount

            Aslide_resized_unmasked = A_resized[x_start_reg_resized:x_end_reg_resized+1,y_left_reg_resized_shift:y_right_reg_resized_shift+1] # This is to store unmasked images for first 2 zooks

            corMat_coarse[framenumber,slideamount+slide_extent] = corr2(A1slide_resized_unmasked,Aslide_resized_unmasked)
#            print('slideamount+slide_extent', slideamount+slide_extent)
#            si =np.append(si,corr2(A1slide_unmasked,Aslide_unmasked))
#            corMat[slideamount+slide_extent] = corr2(A1slide_unmasked,Aslide_unmasked)


        best_slideamount = np.argmax(corMat_coarse[framenumber,:]) - slide_range*resampling_factor

        isconvex = 0
        ##check for convexity


        if best_slideamount in np.arange((-slide_range+1)*resampling_factor,(slide_range)*resampling_factor,resampling_factor):
            best_corr = corMat_coarse[framenumber, int(best_slideamount+slide_range*resampling_factor)]
            nexttobest_corr = corMat_coarse[framenumber,int(best_slideamount+slide_range*resampling_factor+resampling_factor)]
            lasttobest_corr = corMat_coarse[framenumber, int(best_slideamount+slide_range*resampling_factor-resampling_factor)]
            if((best_corr>nexttobest_corr) and (best_corr>lasttobest_corr)):
               isconvex = 1


        if(isconvex==1):
            for slideamount in np.arange(int(best_slideamount)-resampling_factor+1,int(best_slideamount)+resampling_factor):
#            print('slideamount is ', slideamount)
                y_right_reg_resized_shift = y_right_reg_resized +slideamount
                y_left_reg_resized_shift = y_left_reg_resized +slideamount

                Aslide_resized_unmasked = A_resized[x_start_reg_resized:x_end_reg_resized+1,y_left_reg_resized_shift:y_right_reg_resized_shift+1] # This is to store unmasked images for first 2 zooks

                corMat_coarse[framenumber,(slideamount+slide_extent)] = corr2(A1slide_resized_unmasked,Aslide_resized_unmasked)
#               print('slideamount+slide_extent', slideamount+slide_extent)
#               si =np.append(si,corr2(A1slide_unmasked,Aslide_unmasked))
#               corMat[slideamount+slide_extent] = corr2(A1slide_unmasked,Aslide_unmasked)


        if(isconvex==0):
            for slideamount in np.arange(-slide_extent,slide_extent+1):
#            print('slideamount is ', slideamount)
                y_right_reg_resized_shift = y_right_reg_resized +slideamount
                y_left_reg_resized_shift = y_left_reg_resized +slideamount

                Aslide_resized_unmasked = A_resized[x_start_reg_resized:x_end_reg_resized+1,y_left_reg_resized_shift:y_right_reg_resized_shift+1] # This is to store unmasked images for first 2 zooks

                corMat_coarse[framenumber,(slideamount+slide_extent)] = corr2(A1slide_resized_unmasked,Aslide_resized_unmasked)
#               print('slideamount+slide_extent', slideamount+slide_extent)
#               si =np.append(si,corr2(A1slide_unmasked,Aslide_unmasked))
#               corMat[slideamount+slide_extent] = corr2(A1slide_unmasked,Aslide_unmasked)

        calc_shift[framenumber] = np.argmax(corMat_coarse[framenumber,:],axis=0) - slide_extent
        calc_shift_resampled[framenumber] = calc_shift[framenumber]/resampling_factor
        zstamp_corr[framenumber] = (zstamp_det_resized[framenumber,0]/resampling_factor) + calc_shift_resampled[framenumber]
        zstamp_corr_resized[framenumber] = zstamp_corr[framenumber]*resampling_factor
        best_zstamp_corr[framenumber] = np.max(corMat_coarse[framenumber,:],axis=0)

        #########

        y_right_resized = np.absolute(y_end_resized+ zstamp_corr_resized[framenumber]) #y_right is now the
        # position of the column number on the right side of the slide
        # window for the current frame A
        y_left_resized = np.absolute(y_right_resized - width_resized)  # This decides the position of the
        # column number on the left side of the slide window for the current
        # frame A
        Aslide_resized = A_resized[x_start_resized:x_end_resized+resampling_factor,y_left_resized:y_right_resized+resampling_factor]*mask_resized; # A_slide
        # contains the multiplication of the image in the sliding window
        # with mask that was drawn by the user
        Aslide_resized_size = Aslide_resized.shape
        Aslide_resized_back =resize(Aslide_resized, (Aslide_resized_size[0]/resampling_factor,Aslide_resized_size[1]/resampling_factor),preserve_range=True)


        Aslide_unmasked_resized = A_resized[x_start_resized:x_end_resized+resampling_factor,y_left_resized:y_right_resized+resampling_factor] # This is to store unmasked images for first 2 zooks
        Aslide_unmasked_resized_size = Aslide_unmasked_resized.shape
        Aslide_unmasked_resized_back =resize(Aslide_unmasked_resized, (Aslide_unmasked_resized_size[0]/resampling_factor,Aslide_unmasked_resized_size[1]/resampling_factor),preserve_range=True)



        writetiff(Aslide_resized_back,newfolder,counttotalframesinnewimage,prefix=newname,index_start_number=0,num_digits=5)


        writetiff(Aslide_unmasked_resized_back,newfolder_unmasked,counttotalframesinnewimage,prefix=newname,index_start_number=0,num_digits=5)


        new_frame_index[framenumber] = counttotalframesinnewimage
        counttotalframesinnewimage = counttotalframesinnewimage + 1

        #########
#        print('framenumber is ', framenumber)
#        calc_shift[framenumber] = np.argmax(corMat)- slide_extent

        if(i<num_zooks_for_test+ignore_zooks_at_start):
            countframesinnewimage = countframesinnewimage +1 # this keeps count
        # of the numbr of images in the file that will be stored as newname
#        print(countframesinnewimage)

    goodzookendframenumber = np.append(goodzookendframenumber,counttotalframesinnewimage-1) # this stores the frame number of the
    # last frame in each of the three zooks that we used in this m file


plt.hold(False)
plt.figure(326)
plt.hold(True)
plt.plot(np.arange(0,len(zstamp_corr)),zstamp_corr,c='r')
plt.plot(np.arange(0,len(zstamp_corr)),zstamp_det_unrounded,c='k')
plt.hold(False)

plt.figure(327)
plt.plot(np.arange(0,len(zstamp_corr)),zstamp_det_unrounded[:,0]-zstamp_corr,c='r')

plt.figure(328)
plt.plot(np.arange(0,len(best_zstamp_corr)),best_zstamp_corr,c='r')


plt.figure(329)
plt.plot(np.arange(0,len(calc_shift_resampled)),calc_shift_resampled,c='r')

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
#%%

#Step 3.3

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
print(' Step 03.3 : Fit line to the curve')




zstamp_curvefit = np.zeros(shape_kymo_img[0])


for i in range(len(minimapoints)): # This is to just use apx ~0.15% of the zooks for storing reference images
    y1 = zstamp_corr[minimapoints[i]+ignore_startzook:maximapoints[i]-ignore_endzook+1]
    x1 = np.arange(minimapoints[i]+ignore_startzook,maximapoints[i]-ignore_endzook+1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x1,y1)
    y1_prime = slope*x1 + intercept

    zstamp_curvefit[minimapoints[i]+ignore_startzook:maximapoints[i]-ignore_endzook+1] = y1_prime


#y = zstamp_corr[0:191]
#x = np.arange(0,191)
#
#slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#
#y_prime = slope*x + intercept



plt.hold(False)
plt.figure(331)
plt.hold(True)
plt.plot(np.arange(0,len(zstamp_corr)),zstamp_corr,c='r')
plt.plot(np.arange(0,len(zstamp_corr)),zstamp_curvefit,c='k')
plt.hold(False)


plt.figure(332)
plt.plot(np.arange(0,len(zstamp_corr)),zstamp_curvefit-zstamp_corr,c='r')

##histogram
zstamp_diff = zstamp_curvefit-zstamp_corr

plt.figure(333)
plt.hist(zstamp_diff, bins = (np.arange(-5,5.2,0.2)))

mean_zstamp_diff = np.mean(zstamp_diff)
std_zstamp_diff = np.std(zstamp_diff)


#arr[arr > 255] = x
print('For the step number 3.3')
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
###############################################################################################
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



numberof_referenceframes = np.ceil(goodzookendframenumber[0]/10)
Fre = np.zeros(numberof_referenceframes)
Fre_SSIM = np.zeros(numberof_referenceframes,)

count_measurements = 0

#goodzookendframenumber = np.zeros(200)
#Cor = []
#SSIM = []
color=iter(plt.cm.rainbow(np.linspace(0,1,500)))


for i in range(int(numberof_referenceframes)):
#for i in np.arange(8,9):
#    print(i)
    A1 = importtiff(newfolder_unmasked,i,prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
    start_frame = 0

    for k in range(num_zooks_for_test):
#    for k in range(1):
        count = 0
        Cor = np.array([])
#        SSIM = np.array([])
        for j in np.arange(int(start_frame),int(goodzookendframenumber[k]+1)):
            A = importtiff(newfolder_unmasked,j,prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            Cor =np.append(Cor,corr2_masked(A1,A,mask))
#            SSIM = np.append(SSIM,mutual_information(A1, A, bins=256))
            count = count+1

        start_frame = goodzookendframenumber[k]+1+ignore_startzook

        cutoff = 30
        smooth_corr = butter_lowpass_filtfilt(Cor, cutoff, BF_fps)
#        smooth_SSIM = butter_lowpass_filtfilt(SSIM, cutoff, BF_fps)

        f, Pxx = signal.welch(smooth_corr, BF_fps)
        Pxx[f<1] = 0  ##remove the possibility of less than 1 heartbeats per second.
        ##This means that this algorithm cannot work for phase stamping of hearts beating with less than 2 heartbeats per second
        final_f = BF_fps/f
        max_pxx = np.argmax(Pxx[2:])



#        f_SSIM, Pxx_SSIM = signal.welch(smooth_SSIM, BF_fps)
#        final_f_SSIM = BF_fps/f_SSIM
#        max_pxx_SSIM = np.argmax(Pxx_SSIM[2:])

        c=next(color)

        plt.figure(511)
        plt.plot(final_f[2:],Pxx[2:],c=c)
        plt.hold(True)

#        plt.figure(512)
#        plt.plot(final_f_SSIM[2:],Pxx_SSIM[2:],c=c)
#        plt.hold(True)

    Fre[count_measurements] = final_f[max_pxx+2]
#    Fre_SSIM[count_measurements] = final_f_SSIM[max_pxx_SSIM+2]

    count_measurements = count_measurements+1


meanperiodframes_corr = np.round(stats.mstats.mode(Fre))[0]
#meanperiodframes_SSIM = np.round(stats.mstats.mode(Fre_SSIM))[0]


print(' The mean period of frames using Corr is')
print(meanperiodframes_corr)

#print(' The mean period of frames using SSIM is')
#print(meanperiodframes_SSIM)

print('For the step number 5')
toc = time.time()
print(toc-tic)
print('            ')
print('##################################################')




#%%



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
from multicoreSM import multifunc

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('            ')
print(' Step 06 : Finding Representative Images started')


## Section to switch between Corr and MI
meanperiodframes =  meanperiodframes_corr

repfolder = location+folder_otherdata+'/Representative Phases'
makedirs(repfolder)
repname= 'FRAMEX' # Name of the stage motion corrected new image data


## Step 6.1 Calculating the cluster index of images for every continuous set of

start_frame = 0
sizeA1 = np.asarray(np.shape(importtiff(newfolder_unmasked,0,prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)))
#sizeA1 is the size of the first frame in the stage motion corrected image file


CorrelationMatrix = np.zeros((countframesinnewimage,countframesinnewimage))

for i in np.arange(0,countframesinnewimage):
#    print(i)
#    for j in range(num_zooks_for_test): #For 3 zooks
    for k in np.arange(start_frame,goodzookendframenumber[num_zooks_for_test-1]+1):

        if(k>=i):
            Z1 = importtiff(newfolder_unmasked,int(k),prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            Z2 = importtiff(newfolder_unmasked,int(i),prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            corrval = corr2_masked(Z1,Z2,mask)
            CorrelationMatrix[k,i] = corrval
            CorrelationMatrix[i,k] = corrval

##
mean_clustered_frames = np.ones(goodzookendframenumber[num_zooks_for_test-1]-meanperiodframes+2-start_frame)*meanperiodframes
std_clustered_frames = np.ones(goodzookendframenumber[num_zooks_for_test-1]-meanperiodframes+2-start_frame)*meanperiodframes*100
mean2std_clustered_frames = np.zeros(goodzookendframenumber[num_zooks_for_test-1]-meanperiodframes+2-start_frame)

ClusteredFramesAll = []
MatchMatAll = []
for i in range(num_zooks_for_test):    #For 3 zooks
#for i in range(1):
    for j in np.arange(start_frame,goodzookendframenumber[i]-meanperiodframes+2):
#    for j in np.arange(start_frame,start_frame+2):
        Matchingframes = CorrelationMatrix[j:j+meanperiodframes,0:countframesinnewimage]

        ArgMax = np.argmax(Matchingframes,axis=0)

        MatchMat = np.zeros(Matchingframes.shape)

        for k in range(len(ArgMax)):
            MatchMat[ArgMax[k],k] = 1

        ClusteredFrames = np.sum(MatchMat,axis=1)
        mean_clustered_frames[j] = np.mean(ClusteredFrames)
        std_clustered_frames[j] = np.std(ClusteredFrames)
        mean2std_clustered_frames[j] = mean_clustered_frames[j]/std_clustered_frames[j]


        ClusteredFramesAll.append(ClusteredFrames)
        MatchMatAll.append(MatchMat)
    start_frame = goodzookendframenumber[i]+1
##
## Plotting the results and finding the best representastive image
startframe_for_repimages = np.argmax(mean2std_clustered_frames)

plt.figure(601)
plt.bar(range(len(mean_clustered_frames)),mean_clustered_frames,width=1,bottom=0,yerr=std_clustered_frames,color='y')



#%%
##Store the representative image
j=0
for i in np.arange(startframe_for_repimages,startframe_for_repimages+meanperiodframes):
    R = importtiff(newfolder_unmasked,int(i),prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
    writetiff(R,repfolder,j,prefix=repname,index_start_number=0,num_digits=5)
    j = j+1

#%%


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
from img_proccessing import corr2, ssim2, ZookZik_1X_v1_MaximumCorrelationCluster, ZookZik_1X_v1_MaximumCorrelationCluster_vectorized, ZookZik_1X_v1_MaximumCorrelationCluster_masked
from filter_function import butter_lowpass_filtfilt
from medpyimage import mutual_information
from itertools import cycle
from multiprocessing import Pool
import copy
from itertools import repeat
from multicoreSM import multifunc

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('            ')
print(' Step 07 : Phase Stamping for all frames started')


maskedBFfolder = location+folder_otherdata+'/Masked BF Frames'
makedirs(maskedBFfolder)
maskedname= 'FRAMEX' # Name of the stage motion corrected new image data



#%%
##Read the representative frames
R = np.zeros(((sizeA1[0],sizeA1[1],meanperiodframes)))
for b in np.arange(index_start_number_BF,index_start_number_BF+meanperiodframes):
    R[:,:,b-index_start_number_BF]= importtiff(repfolder,int(b),prefix=repname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
            #R reads the frame number b from the stagemotioncorrected files.

#%% Phase stamp every single frame
start_frame = 0
pstamp = np.ones(total_frames_BF)*total_frames_BF ##using a large number for frames that will be ignored during phase stamping
pstamp_correlation = np.ones(total_frames_BF)*total_frames_BF ##using a large number for frames that will be ignored during phase stamping

for i in np.arange(ignore_zooks_at_start,len(minimapoints)):
#for i in range(len(minimapoints)):    #For all zooks
#for i in range(1):
    for j in np.arange(minimapoints[i]+ignore_startzook+numberofframesatbeginningforcoordination,maximapoints[i]-ignore_endzook):
#    for j in np.arange(start_frame,start_frame+2):
        current_frame_number_in_unmasked_file = new_frame_index[j]
        A = importtiff(newfolder_unmasked,int(current_frame_number_in_unmasked_file),prefix=newname,index_start_number=index_start_number_BF,num_digits=num_digits_BF)

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
        [max_correlatedframe,max_correlationvalue] = ZookZik_1X_v1_MaximumCorrelationCluster_masked(R,A,mask)
        pstamp[j] = max_correlatedframe
        pstamp_correlation[j] = max_correlationvalue


plt.hold(False)
plt.figure(701)
plt.hold(True)
plt.plot(np.arange(0,len(pstamp)),pstamp,c='r')
plt.hold(False)


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
##Step 7
#
#tic = time.time()   # start counting time
#
##from __future__ import division
#from os import makedirs
#import numpy as np
#from scipy.signal import argrelextrema
#from scipy import stats, ndimage, misc, signal
#import math
#import matplotlib.pyplot as plt
#import time
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import skimage.external.tifffile as tff
#import pandas
#from tiffseriesimport import importtiff, writetiff
#from img_proccessing import corr2, ssim2, ZookZik_1X_v1_MaximumCorrelationCluster, ZookZik_1X_v1_MaximumCorrelationCluster_vectorized, ZookZik_1X_v1_MaximumCorrelationCluster_masked
#from filter_function import butter_lowpass_filtfilt
#from medpyimage import mutual_information
#from itertools import cycle
#from multiprocessing import Pool
#import copy
#from itertools import repeat
#from multicoreSM import multifunc
#
#print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#print('            ')
#print(' Step 07 : Phase Stamping for all frames started')
#
#
#maskedBFfolder = location+folder_otherdata+'/Masked BF Frames'
#makedirs(maskedBFfolder)
#maskedname= 'FRAMEX' # Name of the stage motion corrected new image data
#
#
##%%
###Read the representative frames
#R = np.zeros(((sizeA1[0],sizeA1[1],meanperiodframes)))
#for b in np.arange(index_start_number_BF,index_start_number_BF+meanperiodframes):
#    R[:,:,b-index_start_number_BF]= (importtiff(repfolder,int(b),prefix=repname,index_start_number=index_start_number_BF,num_digits=num_digits_BF))*mask
#            #R reads the frame number b from the stagemotioncorrected files.
#
##%% Phase stamp every single frame
#start_frame = 0
#pstamp = np.zeros(total_frames_BF)
#pstamp_correlation = np.zeros(total_frames_BF)
#for i in range(len(minimapoints)):    #For all zooks
##for i in range(1):
#    for j in np.arange(minimapoints[i]+ignore_startzook+numberofframesatbeginningforcoordination,maximapoints[i]-ignore_endzook):
##    for j in np.arange(start_frame,start_frame+2):
#        A = importtiff(folderpath_BF,int(j),prefix=prefix_BF,index_start_number=index_start_number_BF,num_digits=num_digits_BF)
#
#        y_right = np.absolute(y_end + zstamp[j]) #y_right is now the
#        # position of the column number on the right side of the slide
#        # window for the current frame A
#        y_left = np.absolute(y_right - width)  # This decides the position of the
#        # column number on the left side of the slide window for the current
#        # frame A
#
#        if (y_left>0):
#            size_of_frame = np.asarray(np.shape(A[x_start:x_end+1,y_left:y_right+1]))
#
#            if (np.array_equal(size_of_frame,np.asarray(np.shape(mask)))== True):
#                Aslide = A[x_start:x_end+1,y_left:y_right+1]*mask # A_slide
#                writetiff(Aslide,maskedBFfolder,j,prefix=maskedname,index_start_number=0,num_digits=5)
#                [max_correlatedframe,max_correlationvalue] = ZookZik_1X_v1_MaximumCorrelationCluster_masked(R,Aslide,mask)
#                pstamp[j] = max_correlatedframe
#                pstamp_correlation[j] = max_correlationvalue
#
#
#
#toc = time.time()
#print(toc-tic)
#print('            ')
#print('##################################################')
#
#
#
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
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##%%
#
##Step 8
#
#tic = time.time()   # start counting time
#
##from __future__ import division
#from os import makedirs
#import numpy as np
#from scipy.signal import argrelextrema
#from scipy import stats, ndimage, misc, signal
#import math
#import matplotlib.pyplot as plt
#import time
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import skimage.external.tifffile as tff
#import pandas
#from tiffseriesimport import importtiff, writetiff
#from img_proccessing import corr2, ssim2, ZookZik_1X_v1_MaximumCorrelationCluster, ZookZik_1X_v1_MaximumCorrelationCluster_vectorized
#from filter_function import butter_lowpass_filtfilt
#from medpyimage import mutual_information
#from itertools import cycle
#from multiprocessing import Pool
#import copy
#from itertools import repeat
#from multicoreSM import multifunc
#
#print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#print('            ')
#print(' Step 08 : ZP table assignment started')
#
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
