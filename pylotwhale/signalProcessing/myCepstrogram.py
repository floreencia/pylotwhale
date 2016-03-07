#!/usr/bin/python

import numpy as np
import pylab as pl
import scipy.signal as sig
import scikits.audiolab as al
import argparse
import os
import plSpecgrams as pspec # for the plots

parser = argparse.ArgumentParser() #takes the input and interprets it
parser.add_argument("wavFN", type=str, help = "wav file name")
parser.add_argument("-o", "--overlap", type = float, default = 0.5, help = "overlap of the NFTT")
parser.add_argument("-w", "--winPow", type = int, default = 9, help = "Exponent of the FFT window size, NFFT = 2^{windowPow}")
parser.add_argument("-s", "--saveDat", type = int, default = 0, help = "do you whant to save the cepstrum data? (0,1)")

# check input
assert( parser.parse_args().overlap >= 0 and parser.parse_args().overlap < 1 ) #overlap \in [0,1)
assert( parser.parse_args().winPow < 15 ) #overlap \in [0,1) 

###########################################################################
#############################   FUNCIONS   ################################
###########################################################################

def computeCeps(x, sRate, pw, over):
    """
    generates the cepstrogram of a waveform given the:
    waveform - the waveform, 
    sRate - the sampling rate
    pw - the power of the NFFT window
    over - overlap
    
    RETURNS: complex cepstrum (Cxy), final_time (tf), final_qfreq (qf), number_coeffis (numCeps)
    """
    
    ## fft settings
    NFFT = 2**pw
    pad_to = NFFT
    noverlap = int(NFFT*over)
    numFreqs = pad_to //2 # int division 
    winN = 'hamming'
    windowVals =  sig.get_window(winN, NFFT)# returns the fft window with the sieze of NFFT
    scaling_factor = 2
    step = NFFT - noverlap
    ind = np.arange(0, len(x) - NFFT + 1, step) # prepare the indexes
    qf = 1.0*NFFT/(2.0*sRate) #compute the final qfreq
    numCeps = numFreqs//2
    windowC = sig.get_window('hamming',numFreqs)
    #tf = 1.0*(len(waveform)-0.1)/sRate #final time
    tf = 1.0*(len(x)-0.1)/sRate #final time
    
    #print "singal", len(x), "NFFT", NFFT, "overlap", noverlap, "num freqs", numFreqs, "Nceps", numCeps, "final quefreq", qf

    n = len(ind)

    Pxy = np.zeros((numFreqs, n), np.complex_)
    Cxy = np.zeros((numCeps, n), np.complex_)

    for i in range(n):
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals * pl.detrend(thisX) #flo: window the singal, detrend desplaces the signal to cero mean
        #thisX = windowVals * thisX #+flo: window the singal
        #fx = np.fft.fft(thisX, n=pad_to) -flo
        fx = ( np.fft.fft(thisX, n=pad_to)) #+flo: fourier transform
        Px = np.conjugate(fx[:numFreqs]) * fx[:numFreqs] #+flo: abs
        Px /= (np.abs(windowVals)**2).sum() #+flo: scale
        Px[1:-1] *= scaling_factor #+flo
        Cx = ( np.fft.fft(windowC*np.log(Px))) #+flo: fourier transform
        Cxy[:,i] = np.conjugate(Cx[:numCeps]) * Cx[:numCeps] #flo: power scpectrum: 1) tae
    
    return Cxy, tf, qf, numCeps # complex cepstrum, final time, final qfreq, number of cepstral coeffis


def plotCeps(C, tf, qf, baseN):
    pl.shape(Cxy)
    C = np.abs(Cxy)
    
    fig = pl.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    cax = ax.imshow(np.log10(np.abs(C)), extent=[ 0, tf, 0, 1000*qf], origin='lower', aspect = 'auto')#, cmap=pl.cm.gray_r)
    
    # labels 
    pl.xlabel('time [s]', fontsize=16)
    pl.ylabel('quefrency [ms]', fontsize=16)
    ax.tick_params(axis='both', labelsize='x-large') 
    
    # cbar    
    ( ll,ml,ul ) = pspec.cbarLabels( np.log10(C).min(), np.log10(C).max() )
    cbar = fig.colorbar(cax, ticks=[ll, ml, ul])
    cbar.ax.set_yticklabels(['10$^{%d}$'%ll,'10$^{%d}$'%ml,'10$^{%d}$'%ul], size='x-large')# vertically oriented colorbar
    
#pl.ylim(0,20)

    outF = baseN+'-queCeps-noNormalized.png'
    pl.savefig(outF)
    print "out:", outF


def plNumCeps( Cxy, tf, numCeps, baseN ):
    pl.shape(Cxy)
    C = np.abs(Cxy)

    fig = pl.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    cax = ax.imshow(np.log10(np.abs(C)), extent=[ 0, tf, 1, numCeps], origin='lower', aspect = 'auto')#, cmap=pl.cm.gray_r)

    # labels
    pl.xlabel('time [s]', fontsize=16)
    pl.ylabel('cepstral coefficient', fontsize=16)
    ax.tick_params(axis='both', labelsize='x-large') 

    # cbar
    ( ll, ml, ul ) = pspec.cbarLabels( np.log10(C).min(), np.log10(C).max() ) # returns the labels of the cbar
    cbar = fig.colorbar(cax, ticks=[ll, ml, ul])
    cbar.ax.set_yticklabels(['10$^{%d}$'%ll,'10$^{%d}$'%ml,'10$^{%d}$'%ul], size='x-large')# vertically oriented colorbar

    # pl.ylim(0,20)

    outF = baseN+'.png'
    pl.savefig(outF)
    print "out:", outF


###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
###########################################################################


##### ASSIGMENTS #####
args = parser.parse_args()
fileN = args.wavFN ## file name
NFFTpow = args.winPow ## power
overlap = args.overlap ## overlap
saveD = args.saveDat ## save data

##### FILE HANDLING #####
outDN = os.path.dirname(fileN)+'-cepst'
fileBN = os.path.basename(fileN).split('.')[0] # take the base name and remove the extension
fileBN = fileBN + '-ceps_p%d_o%d'%(NFFTpow, int(100*overlap))
Bname = outDN+'/'+fileBN
print "base name:" ,Bname

## Check dir or create
if not os.path.isdir(outDN):
    print "creating out dir:", outDN 
    os.mkdir(outDN)


##### COMPUTATIONS #####
## get waveform
waveForm, sRate, m = al.wavread(fileN)

## compute cepstrum
myCeps, tf, qf, Nceps  = computeCeps( waveForm, sRate, NFFTpow, overlap) # compute the cesptrum

## plot the cesptrum
plNumCeps(myCeps, tf, Nceps, Bname) 

## save cepstrum data
if( saveD ):
    outFN = Bname+'.dat'
    print "saving ceps data", outFN, np.shape(myCeps)
    pl.savetxt( outFN, abs(myCeps) )
