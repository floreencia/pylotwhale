# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:18:42 2016

@author: florencia
"""

#!/usr/bin/python

from __future__ import print_function
import numpy as np
import functools

### Audio feature modules
import librosa as lf # Librosa for audio
import features as psf # Librosa for audio

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
#import random
#import ast
from scipy.io import wavfile
#import scikits.audiolab as al
import sys
import os.path
import scipy.signal as sig

import pylotwhale.utils.annotationTools as annT
matplotlib.rcdefaults()
matplotlib.rcParams.update({'savefig.bbox' : 'tight'})

"""
    Module for waveform effects
    florencia @ 06.09.14
"""

###########################################################
#####           waveform manipulations                #####
###########################################################


    
#### WAVEFORM MANIPULATIONS    
    
def normalizeWF(waveform):
    return 1.0*waveform/np.max(np.abs(waveform))
    
def tileTillN(arr, N, n0=0):
    '''returns an arrray of size N (>0) from tiling of arr. n0 is the starting index'''
    #np.tile(arr, int(n/len(arr))+1)[:n]
    return arr[np.array([i for i in np.arange(n0, N + n0)%len(arr)])]
    
def addWhiteNoise(y, param=1.0):
    """
    adds white noise with amplitude 'param" to y
    """
    y_ns = np.random.random_sample(len(y))*2 - 1 # white noise
    return y + param*y_ns

    
def addToSignal(y1, y2, noiseIndex):
    '''
    adds y2 (noise) to the primary signal y1. Returns the sum, keeping the size of y1
    '''
    return y1 + tileTillN(y2, len(y1), noiseIndex)    
    

def freqshift(y, Fs, param=100):
    """Frequency shift the signal by param
    """
    x = np.fft.rfft(y)
    T = len(y)/float(Fs)
    df = 1.0/T
    nbins = int(param/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    z = np.fft.irfft(y)
    return z
    
    
    

def waveformEffectsDictionary(funName=None):
    '''
    Dictionary of waveform effects
    Parameters:
    -----------
        < funName : effect name (if None)
    Return:
    -------
        > effect function
            this functions take the waveform an alteration if the later
    '''
    D = {
        'addWhiteNoise' : addWhiteNoise,# param : amplitud of the white noise waveform 
        'addSignals' : addToSignal,
        'freqShift' : freqshift, # y, Fs, fshift=100
        'timeStreach' : lf.effects.time_stretch, # time_stretch(y, 1.5)
        'pitchshift' : lf.effects.pitch_shift
        }

    if funName == None: # retuns a list of posible feature names
        return D.keys()
    else:
        return D[funName]

    
#### waveform ensembles   
    
def generateWaveformEnsemble(y_template, effectName=None, generate_data_grid=None):#, **kwEffect):
    '''
    Generates an ensemble of signals from y_template 
    using the parameters in generate_data_grid
    !!! only works for white noise addition
    Params:
    -------
        y_template : waveform (np.array)
        effect : name of the effect to use (string)
        generate_data_grid : parameter grid for the effect (np.array)
        kwEffect : kwargs for the effect
    Returns:
    -------
        Y : waveform ensemble (np.array), one waveform per row
    '''
    
    if generate_data_grid is None: generate_data_grid = np.ones(1)
    if effectName is None: 
        effectFun = waveformEffectsDictionary("addWhiteNoise")
    else:
        effectFun = waveformEffectsDictionary(effectName)
    
    Y = np.zeros((len(generate_data_grid), len(y_template)))
        
    for i in range(len(generate_data_grid)):
        Y[i,:] = effectFun(y_template, param=generate_data_grid[i])#, **kwEffect)
    return(Y)
        

def generateAddEnsemble(y_template, y_add, intensity_grid=None, normalize=True):
    '''
    generate an ensemble of y_template-singnals adding y_add
    normalizes both signals and adds different amplitudes of y_add to y_template
    Returns:
    Y : a matrix, with the sum of y_template and y_add in each row
    '''
    if intensity_grid is None:
        intensity_grid = np.linspace(0.1, 10, 10)
     
    #print(len(intensity_grid), len(y_template))
    Y = np.zeros((len(intensity_grid), len(y_template)))
    if normalize:
        y_template = normalizeWF(y_template)
        y_add = normalizeWF(y_add)
        
    for i in range(len(intensity_grid)):
        Y[i,:] = addToSignal(y_template, intensity_grid[i]*y_add, np.random.randint(0,len(y_template)))
        #y_template + intensity_grid[i]*tileTillN(y_add, len(y_template), np.random.randint(0,len(y_template)))
    
    return Y    
    
def generatePitchShiftEnsemble(y_template, fs, shift_grid=None):
    '''
    generate an ensemble of y_template-singnals shifting the pitch of the original signal
    normalizes both signals and adds different amplitudes of y_add to y_template
    Parameters:
    -----------
        shift_grid : 12 steps per octave
    Returns:
        Y : a matrix, with the sum of y_template and y_add in each row
    '''
    if shift_grid is None:
        shift_grid = np.linspace(-2, 2, 5)
     
    #print(len(intensity_grid), len(y_template))
    Y = np.zeros((len(shift_grid), len(y_template)))
    for i in range(len(shift_grid)):
        Y[i,:] = lf.effects.pitch_shift(y_template, fs, shift_grid[i])
        #y_template + intensity_grid[i]*tileTillN(y_add, len(y_template), np.random.randint(0,len(y_template)))
    
    return Y    
    
def generateTimeStreachEnsemble(y_template, streach_grid=None):
    '''
    generate an ensemble of y_template-singnals adding y_add
    normalizes both signals and adds different amplitudes of y_add to y_template
    Returns:
    Y : a matrix, with the sum of y_template and y_add in each row
    '''
    if streach_grid is None:
        streach_grid = np.linspace(0.8, 1.2, 5)
     
    #print(len(intensity_grid), len(y_template))
    Y = []#np.zeros((len(streach_grid), len(streach_grid)))
    for i in range(len(streach_grid)):
        Y.append(lf.effects.time_stretch(y_template, streach_grid[i]))
    
    return Y 
    
    

##### wav files


