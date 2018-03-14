# -*- coding: utf-8 -*-
"""
Created on Wed May 25 06:36:17 2016

@author: smadaan
"""

import numpy as np


def corr2(a, b):
    import numpy as np

    # Getting shapes and prealocating the auxillairy variables
    k = np.shape(a)

    # Calculating mean values
    AM = np.mean(a)
    BM = np.mean(b)

    # calculate vectors
    c_vect = (a-AM)*(b-BM)
    d_vect = (a-AM)**2
    e_vect = (b-BM)**2

    # Formula itself
    r_out = np.sum(c_vect)/float(np.sqrt(np.sum(d_vect)*np.sum(e_vect)))

    return r_out


def ssim2(a, b, k1=0.01, k2=0.03):

    import numpy as np

    # the default values of k1 = 0.01 and k2 = 0.03
    # Getting shapes and prealocating the auxillairy variables
    k = np.asarray(np.shape(a))

    # Calculating the dynamic range
    mindata = np.min([np.min(a), np.min(b)])
    maxdata = np.max([np.max(a), np.max(b)])
    L = maxdata

    # Calculating mean values
    AM = np.mean(a)
    BM = np.mean(b)

    # Calculating variance values
    c_vect = ((a-AM)*(b-BM))/(k[0]*k[1])
    d_vect = ((a-AM)**2)/(k[0]*k[1])
    e_vect = ((b-BM)**2)/(k[0]*k[1])

    # Calculating the other variables
    c1 = (k1*L)**2
    c2 = (k2*L)**2

    num = ((2*np.sum(AM)*np.sum(BM))+c1)*((2*np.sqrt(np.sum(c_vect)))+c2)
    den = (((np.sum(AM))**2) + ((np.sum(BM))**2) + c1) * \
        (np.sum(d_vect)+np.sum(e_vect)+c2)
    ssim_out = num/den

    return ssim_out


def butter_lowpass(cutoff, fs, order=5):
    from scipy import signal
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    from scipy import signal
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def ZookZik_1X_v1_MaximumCorrelationCluster(R, RefImage):
    import numpy as np
    from img_proccessing import corr2, ssim2
    Cor = np.zeros(R.shape[2])
    for t in range(R.shape[2]):
        Cor[t] = corr2(RefImage, R[:, :, t])  # Correlation

    max_correlationvalue_intermediate = np.amax(Cor)
    max_correlatedframe_intermediate = np.argmax(Cor)

    max_correlatedframeandvalue = [
        max_correlatedframe_intermediate, max_correlationvalue_intermediate]

    # use this to use histogram but avoid the for loop in t
    # http://stackoverflow.com/questions/18851471/numpy-histogram-on-multi-dimensional-array
    ##
    return max_correlatedframeandvalue


def ZookZik_1X_v1_MaximumCorrelationCluster_vectorized(R, RefImage):
    import numpy as np
#    from img_proccessing import corr2, ssim2

    R_shape = np.asarray(np.shape(R))
    R_flattened = R.swapaxes(0, 2).reshape(R_shape[2], R_shape[0]*R_shape[1])

    RefImageflattened = RefImage.transpose().ravel()
#    RefImagetiled = np.tile(RefImageflattened,(R_shape[2],1))

    AA = np.transpose(R_flattened)
#    BB = np.transpose(RefImagetiled)

    # Calculating mean subtracted values
    AAM = AA - np.mean(AA, axis=0)
#    BBM=np.mean(BB,axis=0)
    BM = RefImageflattened - np.mean(RefImageflattened)

    # calculate vectors
#    CC_vect = (AA-AAM)*(BB-BBM)
#    DD_vect = (AA-AAM)**2
#    EE_vect = (BB-BBM)**2
    DD_vect = AAM**2
    E_vect = BM**2
    EE_vect = np.transpose(np.tile(np.transpose(E_vect), (R_shape[2], 1)))
    CC_vect = AAM*np.transpose(np.tile(BM, (R_shape[2], 1)))

    # Formula itself
    Cor = np.sum(CC_vect, axis=0)/np.sqrt((np.sum(DD_vect, axis=0)
                                           * np.sum(EE_vect, axis=0)).astype(float))

    max_correlationvalue_intermediate = np.amax(Cor)
    max_correlatedframe_intermediate = np.argmax(Cor)

    max_correlatedframeandvalue = [
        max_correlatedframe_intermediate, max_correlationvalue_intermediate]

    # use this to use histogram but avoid the for loop in t
    # http://stackoverflow.com/questions/18851471/numpy-histogram-on-multi-dimensional-array
    ##
    return max_correlatedframeandvalue


def ZookZik_1X_v1_ClusterindexforZook(J, MPF, Z):

    from img_proccessing import ZookZik_1X_v1_MaximumCorrelationCluster
    import numpy as np
    import time

    tic = time.time()
    R = Z[:, :, J:J+MPF]
    for k in np.arange(0, Z.shape[2]):
        A1 = Z[:, :, k]
        [max_correlatedframe,
            max_correlationvalue] = ZookZik_1X_v1_MaximumCorrelationCluster(R, A1)

    toc = time.time()
    print('Clustering dne using frames between #s  ' +
          str(J) + '  and  ' + str(J+MPF-1) + '  as reference')
    print('time taken is ' + str(toc-tic))

    return [max_correlatedframe, max_correlationvalue]


def ZookZik_1X_v1_MaximumMICluster(R, RefImage, bins=256):
    import numpy as np
    from medpyimage import mutual_information
    mi = np.zeros(R.shape[2])
    for t in range(R.shape[2]):
        mi[t] = mutual_information(RefImage, R[:, :, t], bins=bins)

    max_mivalue_intermediate = np.amax(mi)
    max_miframe_intermediate = np.argmax(mi)

    max_miframeandvalue = [max_miframe_intermediate, max_mivalue_intermediate]

    # use this to use histogram but avoid the for loop in t
    # http://stackoverflow.com/questions/18851471/numpy-histogram-on-multi-dimensional-array
    ##
    return max_miframeandvalue


def corr2_masked(a, b, mask):
    # Applying the mask to create new vectors
    a = a[np.where(mask == 255)]
    b = b[np.where(mask == 255)]
    # Getting shapes and prealocating the auxillairy variables
    k = np.shape(a)

    # Calculating mean values
    AM = np.mean(a)
    BM = np.mean(b)

    # calculate vectors
    c_vect = (a-AM)*(b-BM)
    d_vect = (a-AM)**2
    e_vect = (b-BM)**2

    # Formula itself
    r_out = np.sum(c_vect)/float(np.sqrt(np.sum(d_vect)*np.sum(e_vect)))

    return r_out


def ZookZik_1X_v1_MaximumCorrelationCluster_masked(R, RefImage, mask):
    import numpy as np
    from img_proccessing import corr2, ssim2, corr2_masked
    Cor = np.zeros(R.shape[2])
    for t in range(R.shape[2]):
        Cor[t] = corr2_masked(RefImage, R[:, :, t], mask)  # Correlation

    max_correlationvalue_intermediate = np.amax(Cor)
    max_correlatedframe_intermediate = np.argmax(Cor)

    max_correlatedframeandvalue = [
        max_correlatedframe_intermediate, max_correlationvalue_intermediate]

    # use this to use histogram but avoid the for loop in t
    # http://stackoverflow.com/questions/18851471/numpy-histogram-on-multi-dimensional-array
    ##
    return max_correlatedframeandvalue


def ZookZik_1X_v1_MinimumSADCluster(R, RefImage):
    import numpy as np

    Sad = np.zeros(R.shape[2])
    for t in range(R.shape[2]):
        Sad[t] = np.sum(np.absolute(RefImage-R[:, :, t]))

    min_sadvalue_intermediate = np.amin(Sad)
    min_sadframe_intermediate = np.argmin(Sad)

    min_sadframeandvalue = [
        min_sadframe_intermediate, min_sadvalue_intermediate]

    # use this to use histogram but avoid the for loop in t
    # http://stackoverflow.com/questions/18851471/numpy-histogram-on-multi-dimensional-array
    ##
    return min_sadframeandvalue
