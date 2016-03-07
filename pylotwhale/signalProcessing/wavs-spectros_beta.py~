#!/usr/bin/python

import numpy as np
import pylab as pl
import sys
import pandas as pd
import random
import ast

"""
    Module for for the preprocessing of features
    florencia @ 12.08.14

    
"""


###########################################################
#####                     plotting                    #####
###########################################################


###########################################################
#####              ON spectral matrices               #####
###########################################################

# functions

def reeScale_E(M, spec_factor = 1.0/3.0):
    """
    Zeroes the noise by taking only the part of the spectrum with the higest energy.
    * M, log (spectrogram)
    """
    
    cutE = (np.max(M) + np.min(M))*spec_factor
    nx, ny = np.shape(M)
    M_tr = np.copy(M)

    for i in range(nx):
        for j in range(ny):
            if M_tr[i,j] < cutE: M_tr[i, j] = cutE
                
    return M_tr

def selectBand(M, fr_0 = 0, fr_f = 24000, v0_cut = 1.0*1000, vf_cut = 20.0*1000):
    """
    selects a band on frquencies from the matrix M
    fr_0, initial frequency of the matrix
    fr_f, finnla frequency of the matrix
    cutting frequencies
    v0_cut
    vf_cut
    """
    ny, nx = np.shape(M)
    n0_cut = int( ny*v0_cut/( fr_f - fr_0 ) )
    nf_cut = int( ny*vf_cut/( fr_f - fr_0 ) )
    print n0_cut, nf_cut, nx, ny
    return M[ n0_cut:nf_cut, : ]

def allPositive_andNormal(M):
    """
    normalizes the matrices, so that all it's values lay in (0,1)
    """
    if np.min(M) < 0:
        M = M - np.min(M)
    M = 1.0*M/np.max(M)
    return M

def reeSize_t(M, h_size = 3938):
    ny, nt = np.shape(M)
    if h_size - nt > 0:
        d = h_size - nt
        reM = np.concatenate((np.zeros((ny, d/2)), M), axis = 1)
        
        return np.concatenate((reM, np.zeros((ny, h_size - np.shape(reM)[1]))), axis = 1)
    else:
        print "WARNING: matrix is to large", nt
    
