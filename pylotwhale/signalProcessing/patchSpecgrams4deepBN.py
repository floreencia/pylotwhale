#!/usr/bin/python

import pylab as pl
import numpy as np
import os
import scikits.audiolab as al


def cbarLabels_f(minV, maxV):
    """
    give me the maximum and the minimum values of color bar and I will retun 3 labels
    the lables returned are float type
    """
    minV = np.ceil(minV)
    maxV = np.floor(maxV)
    ml = (minV+maxV)/2.
    #print minV, maxV,"ml", ml
    D = min(abs(ml-maxV),abs(minV-ml))
    ul = ml - D
    ll = ml + D
    #print D, ul, ll
    return (ll, ml, ul)

def cbarLabels(minV, maxV):
    """
    give me the maximum and the minimum values of color bar and I will retun 3 label
    the lables returned are int type. ment for exponents.
    """
    minV = int(np.ceil(minV))
    maxV = int(np.floor(maxV))
    ml = (minV+maxV)/2
    #print minV, maxV,"ml", ml
    D = min(abs(ml-maxV),abs(minV-ml))
    ul = ml - D
    ll = ml + D
    #print D, ul, ll
    return (ll, ml, ul)

def specgramWav(wav_fileN, powerOfWinLen = 12, overlap = 0.5, saveData = 0): #, max_freq = 8000):
    '''
    this function creates spectrograms form a wav file
    wav_fileN = name of the wavfile we want to plot
    max_freq = the maximum frequency we whant ot display [KHz]
    powerOfWinLen = fft works optimally with widows sizes of the power of 2
    '''

    # read wav file
    waveform, sRate, bla = al.wavread(wav_fileN)
    print "sampling rate:",sRate
    
    # settings
    nPoints = 2**powerOfWinLen
    win = pl.mlab.window_hanning
    tf = 1.0*(len(waveform)-0.1)/sRate
    ff = sRate/2.0
    
    # SPECTROGRAM
    spec = pl.specgram(waveform, NFFT=nPoints, Fs=sRate, window=win, noverlap=int(nPoints*overlap))
    M = spec[0]

    # SAVING DATA
    if(saveData):
        outD = os.path.splitext(wav_fileN)[0]+'-%.1fHann%d.dat'%(overlap, powerOfWinLen)
        np.savetxt(outD,M)
        print 'out data:', outD

    return M, tf, ff


def saveSpecgram(wav_fileN, powerOfWinLen = 12, overlap = 0.9, freqFrac = 1.0): #, max_freq = 8000):
    '''
    this function creates spectrograms form a wav file
    wav_fileN = name of the wavfile we want to plot
    max_freq = the maximum frequency we whant ot display [KHz]
    powerOfWinLen = fft works optimally with widows sizes of the power of 2
    freqFrac = fraction of the max frequency 
    '''
    # settings

    M, tf, ff = specgramWav(wav_fileN, powerOfWinLen = powerOfWinLen, overlap = overlap)

    # ploting

    outF = os.path.splitext(wav_fileN)[0]+'-%dHann%dSpecZoom.png'%(100*overlap, powerOfWinLen)
    print outF
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow( np.log10(M), extent=[0,tf, 0, ff/1000], origin='lower', aspect='auto', interpolation = 'nearest')
    pl.ylim([0, ff*freqFrac/1000])

    # cbar
    ( ll, ml, ul ) = cbarLabels( np.log10(M).min(), np.log10(M).max() )
    cbar = fig.colorbar(cax, ticks=[ll, ml, ul])
    cbar.ax.set_yticklabels(['10$^{%d}$'%ll,'10$^{%d}$'%ml,'10$^{%d}$'%ul], size='x-large')# vertically oriented colorbar

    ax.set_xlabel( 'time [s]', fontsize=16 )
    ax.set_ylabel( 'frequency [KHz]' , fontsize=16)
    ax.tick_params(axis='both', labelsize='x-large')  #set_xticks(ticks, fontsize=24)
    pl.savefig(outF)


def saveBIGplot(wav_fileN, powerOfWinLen = 12, overlap = 0.9): #, max_freq = 8000):
    '''
    this function creates spectrograms form a wav file
    wav_fileN = name of the wavfile we want to plot
    max_freq = the maximum frequency we whant ot display [KHz]
    powerOfWinLen = fft works optimally with widows sizes of the power of 2
    '''
    M, tf, ff  = specgramWav(wav_fileN, powerOfWinLen = powerOfWinLen, overlap = overlap)

    # ploting 
    outF = os.path.splitext(wav_fileN)[0]+'-%dHann%dnoLogSpec.png'%(100*overlap, powerOfWinLen)
    print "spec mat size", np.shape(M), "\nout:", outF
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow( M, extent=[ 0, tf, 0, ff], origin='lower', interpolation = 'nearest', aspect = 'auto')
    fig.colorbar(cax)
    ax.set_xlabel( 'time [s]' )
    ax.set_ylabel( 'frequency Hz]' )
    ax.tick_params(axis='both', labelsize='x-large')  #set_xticks(ticks, fontsize=24)	
    #pl.ylim([0, max_freq*1000])
    pl.savefig(outF)

    outF = os.path.splitext(wav_fileN)[0]+'-%dHann%dSpecZoom.png'%(100*overlap, powerOfWinLen)
    print outF
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow( np.log(M), extent=[0,tf, 0, ff], origin='lower', aspect='auto')#, interpolation = 'nearest')
    pl.ylim([0, ff/2])
    fig.colorbar(cax)
    ax.set_xlabel( 'time [s]' )
    ax.set_ylabel( 'frequency [Hz]' )
    ax.tick_params(axis='both', labelsize='large')  #set_xticks(ticks, fontsize=24)
    pl.savefig(outF)
