# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 22:34:02 2016

@author: smadaan
"""

def butter_lowpass(cutoff, fs, order=5):
    from scipy import signal
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    from scipy import signal
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data,padlen=len(data)-1)
    return y