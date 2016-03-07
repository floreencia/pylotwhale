# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:45:10 2015
@author: florencia
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os

import pylotwhale.signalProcessing.signalTools_beta as sT
import pylotwhale.utils.whaleFileProcessing as fp
import pylotwhale.utils.annotationTools as annT



### annotations validity

def isCronologicalyOrdered(t):
    return not (t[1:]<t[:-1]).any()

def isLast_tf_theLargest(t):
    return np.max(t) == t[-1]

def whereOverlaps(t0, tf):
    '''returns an np.array with the line indexs, starting from 0'''
    tbin = np.zeros(len(tf), dtype=bool)
    for i in range(len(tf)-1):
        tbin[i+1] = (t0[i+1] < tf[:i+1]).any()
    return np.arange(len(tf))[tbin]

def whereIsContinuous(t0, tf):
    '''returns an np.array with the line indexes, starting from 0'''
    cn = t0[1:] == tf [:-1]
    return np.arange(len(tf))[cn]

class annotationsValidity():
    def __init__(self, annFi):
        self.dat = np.loadtxt(annFi, usecols =[0,1])
        self.t0 = self.dat[:,0]
        self.tf = self.dat[:,1]
    
    def isCronologicalyOrdered(self, t):
        return isCronologicalyOrdered(t)
     
    def whereIsContinuous(self, t0=None, tf=None):
        if t0 is None or tf is None:
            tf=self.tf
            t0=self.t0
        return whereIsContinuous(t0, tf)
        
    def whereOverlaps(self, t0=None, tf=None):
        if t0 is None or tf is None:
            tf=self.tf
            t0=self.t0
        return whereOverlaps(t0, tf)
    
    def isLast_tf_theLargest(self):
        return isLast_tf_theLargest(self.tf)  
        
    def printStatus(self):
        print("chronological: t0 {}, tf {}".format( isCronologicalyOrdered(self.tf), 
                                            isCronologicalyOrdered(self.tf)))
        print("continuous:", whereIsContinuous(self.t0, self.tf))
        print("overlaps:", whereOverlaps(self.t0, self.tf))
        print("isLast_tf_theLargest:", isLast_tf_theLargest(self.tf))
    


### plot


def plotAnnotatedSpectro(wavFi, annFi, outDir, callAsTitle=True, figsize=None): 
    '''
    plots the spectrogram with it's annotations
    Parameters
    ----------    
    wavAnnCollection  : collecion of paths to wavs and annotations files
    outDir : dir where the plots will be saved
    '''
    ## wav file
    try:
        waveForm, fs = sT.wav2waveform(wavFi)
    except:
        return False
    M, ff, tf, _ = sT.spectralRep(waveForm, fs)#plt.specgram(waveForm, Fs=fs)
    plt.ioff()
    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(M.T, aspect='auto', origin='bottom', interpolation='nearest', 
              extent=[0, tf, 0, fs/2000.])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequecy (KHz)')
    ## read annotations
    annD = annT.parseAupFile(annFi)
    annLabel = ''
    for item in annD: # plot call segments - black line
        ax.hlines(10, float(item['startTime']), float(item['endTime']))
        annLabel += '{}  '.format(item['label']) #read labels
        
    if callAsTitle:
        ax.set_title('{}'.format(annLabel))
        
    #if callAsTitle: ax.set_title('{}'.format(fp.parseHeikesNameConv(wavFi)['call']))
    
        
    
    outPl = os.path.join( outDir, os.path.basename(wavFi).replace('wav', 'png'))
    plt.savefig(outPl)
    del fig, ax
    
    
def annWavColl2annotatedSpectroPlots( wavAnnCollection, outDir, callAsTitle=True, figsize=None):
    '''
    plots the spectros with it's annotations calling plotAnnotatedSpectro
    Parameters
    ----------    
    wavAnnCollection  : collection of paths to wavs and annotations files
    outDir : dir where the plots will be saved
    '''
    for wF, annF in wavAnnCollection:
        plotAnnotatedSpectro(wF, annF, outDir, callAsTitle=callAsTitle, figsize=figsize)


class annotationsDF():
    def __init__(self, df, names=None):
        """
        class for processing annotations
        ict (gaps), call length
        Parametrs:
        ----------
            df : dataframe with the annotations info
            names : names of the df columns
                default = ['t0', 'tf', 'l']
        """
        ### load data     
        self.df = df
        if names is None: names=['t0', 'tf', 'l']
        self.Nsections =len(self.df)
        ### read times and stamps
        self.t0 = self.df['t0'].values # initial times
        self.tf = self.df['tf'].values # final times
        self.labels = self.df['l'].values # section label - sound - call type
        ### sounds and silences
        self.ict = self.t0[1:] - self.tf[:-1] # inter-call times
        self.calls_lengths = self.tf[:] - self.t0[:] # sound lengths
        ###
        if self.Nsections > 1: # only make sense if there is more than one section
            self.shortestCall = np.min(self.calls_lengths)
            self.longestCall = np.max(self.calls_lengths)
            self.shortestGap = np.min(self.ict)
            self.longestGap = np.max(self.ict)
        else:
            self.shortestCall = None
            self.longestCall = None
            self.shortestGap = None
            self.longestGap = None
            
    def barpl(self, x, y, yl, xl=None, figSize = (14,4), width=None, legLabel=None, outFig=None):
        """
        bar plot
        """
        fig, ax = plt.subplots(figsize=figSize)

        ax.set_ylabel(yl)
        ax.set_xlabel(xl)
        ax.bar(x, y, width=width, edgecolor = "none", label = legLabel)  # color=[(0.5, 0.5, 0.5)],
        ax.legend()
        
        if outFig: fig.savefig(outFig)
        
    def pl_ict(self, width=0.1, outFig=None):
        """
        plot interval times (gaps)
        """
        self.barpl(self.tf[:-1], self.ict, yl="ict [s]", xl = 'tf [s]', width = width)
        
    def pl_call_legths(self, width=0.1, outFig=None):
        """
        plot sound lengths (call length)
        """
        self.barpl(self.t0, self.calls_lengths, yl="call length [s]", xl = 'tf [s]', width = width)
        
        
class file2annotationsDF(annotationsDF):
    def __init__(self, path2file, names=None):
        """
        Class for processing annotations
        reads a text file into a pandas dataframe
        ict (gaps), call length
        Parameters
        -----------
            path2file : path to a somple (eg. csv) text file
            names : names of the columns
                default = ['t0', 'tf', 'l']
        """
        ### load data     
        if names is None: names=['t0', 'tf', 'l']
        self.annotations_file = path2file
        df = pd.read_csv(self.annotations_file, sep ='\t', names=names )
        annotationsDF.__init__(self, df, names)           