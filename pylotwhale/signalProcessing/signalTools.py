#!/usr/bin/python

from __future__ import print_function, division
import functools

import os.path
import scipy.signal as sig

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#import pandas as pd
from sklearn.preprocessing import scale, maxabs_scale, minmax_scale

from scipy.io import wavfile
import librosa  # Librosa for audio


"""
    Tools for manipulating processing audio signals (waveforms) 
    extracting features with librosa
"""

def waveformPreprocessingFun(funName=None):
    '''
    Dictionary of feature extracting functions
    maps method_name --> estimator (callable)
    returns a callable
    ------
    > feature names (if None)
    > feature function
        this functions take the waveform and return an instantiated feature matrix
        m (instances) - rows
        n (features) - columns
    '''
    D = {#'welch' : welchD,
        #'bandEnergy' : bandEnergy, ## sum of the powerspectrum within a band
        'band_pass_filter': butter_bandpass_filter,
        'scale': scale2range,
        'maxabs_scale': maxAbsScale,
        'median_scale': None,
        'whiten': whiten        
        }

    if funName in D.keys(): # returns a list of possible feature names
        return D[funName]
    else:
        return D


def audioFeaturesFun(funName=None):
    '''
    Dictionary of feature extracting functions
    returns a callable
    ------
    > feature names (if None)
    > feature function
        this functions take the waveform and return an instantiated feature matrix
        m (instances) - rows
        n (features) - columns
    '''
    D = {#'welch' : welchD,
        #'bandEnergy' : bandEnergy, ## sum of the powerspectrum within a band
        'spectral' : spectral,
        'spectralDelta' : spectral_nDelta,
        'cepstrum' : cepstral,
        'MFCC' : mfcepstral,
        'MFcepsDelta' : functools.partial(mfcepstral_nDdelta, order=1), # MFCC and delta-MFCC
        'MFcepsDeltaDelta' : functools.partial(mfcepstral_nDdelta, order=2),
        'chroma' : chromogram,
        'melspectroDelta' : melSpectral_nDelta,
        'melspectro' : functools.partial(melSpectral_nDelta, order=0),
        'rms': librosa.feature.rms
        }

    if funName in D.keys(): # returns a list of possible feature names
        return D[funName]
    else:
        return D # returns function name of the asked feature

###########################################################
#####           waveform manipulations                #####
###########################################################

#### WAVEFORM MANIPULATIONS
########### moved to effects.py
    
def standardise(y, axis=0):
    """standardises array along axis
    centres and translates array so that mu = 0 and std = 1"""
    return scale(y, axis=axis)
    
def scale2range(y, feature_range=(-1, 1), axis=0):
    """scales array to range"""
    return minmax_scale(y, feature_range=feature_range, axis=axis)
     
def maxAbsScale(y, axis=0):
    """normalises array dividing by the max abs value"""
    return maxabs_scale(y, axis=axis)
    

def tileTillN(arr, N, n0=0):
    """returns an array of size N (>0) from tiling of arr. n0 is the starting index"""
    #np.tile(arr, int(n/len(arr))+1)[:n]
    return arr[np.array([i for i in np.arange(n0, N + n0)%len(arr)])]
    
def addToSignal(y1, y2, noiseIndex):
    """
    adds y2 (noise) to the primary signal y1. Returns the sum, keeping the size of y1
    """
    return y1 + tileTillN(y2, len(y1), noiseIndex)    
    
def generateAddEnsemble(y_template, y_add , intensity_grid=None):
    """
    generate an ensemble of y_template-signals adding y_add
    normalizes both signals and adds different amplitudes of y_add to y_template
    Returns:
    Y : a matrix, with the sum of y_template and y_add in each row
    """
    if intensity_grid is None:
        intensity_grid = np.linspace(0.1, 10, 10)
     
    #print(len(intensity_grid), len(y_template))
    Y = np.zeros((len(intensity_grid), len(y_template)))
    y_template = normalizeWF(y_template)
    y_add = normalizeWF(y_add)
    for i in range(len(intensity_grid)):
        Y[i,:] = addToSignal(y_template, intensity_grid[i]*y_add, np.random.randint(0,len(y_template)))
        #y_template + intensity_grid[i]*tileTillN(y_add, len(y_template), np.random.randint(0,len(y_template)))
    
    return Y    
    
def generatePitchShiftEnsemble(y_template, fs, shift_grid=None):
    """
    generate an ensemble of y_template-signals shifting the pitch of the original signal
    normalizes both signals and adds different amplitudes of y_add to y_template
    Parameters:
    -----------
        shift_grid : 12 steps per octave
    Returns:
        Y : a matrix, with the sum of y_template and y_add in each row
    """
    if shift_grid is None:
        shift_grid = np.linspace(-2, 2, 5)

    Y = np.zeros((len(shift_grid), len(y_template)))
    for i in range(len(shift_grid)):
        Y[i,:] = librosa.effects.pitch_shift(y_template, fs, shift_grid[i])

    
    return Y    
    
def generateTimeStretchEnsemble(y_template, stretch_grid=None):
    """
    generate an ensemble of y_template-signals adding y_add
    normalizes both signals and adds different amplitudes of y_add to y_template
    Returns:
    Y : a matrix, with the sum of y_template and y_add in each row
    """
    if stretch_grid is None:
        stretch_grid = np.linspace(0.8, 1.2, 5)
     
    #print(len(intensity_grid), len(y_template))
    Y = []#np.zeros((len(stretch_grid), len(stretch_grid)))
    for i in range(len(stretch_grid)):
        Y.append(librosa.effects.time_stretch(y_template, stretch_grid[i]))
    
    return Y 

def freqshift(data, Fs, fshift=100):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data)/float(Fs)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    z = np.fft.irfft(y)
    return z
    

##### wav files

def write_wavfile(filename,fs,data):
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename,int(fs), d)

###########################################################
#####                     plotting                    #####
###########################################################

def wavs2spectros(files, dirN='', outFig = '', title = '', winPow = 9,
                  over = 0.5, axTitle = True, fc0 = 0, fcf=120,
                  aspect = 'auto', figScale=1, spec_factor = '0.6'):
    """
    < files, an array with the wav file names
    * spec_factor, 0 = nothing, 1 = all
    """
    n = len(files[:12])
    nC, nR = fitNinSqr(n) # print "size ", nR, nC
    fig, axes  = plt.subplots(nrows=nR, ncols=nC, figsize = (int(nR)*4*figScale, int(nC)*2*figScale) )
    counter = 0

    for axR in axes:
        for ax in axR:
            thisFile = os.path.join(dirN, files[counter]) if dirN else files[counter]
            M0, tf, ff = specgramWav('%s'%thisFile, powerOfWinLen=winPow, overlap=over)
            M = reeScale_E(np.log(M0), spec_factor = spec_factor)
            M = selectBand(M, v0_cut=fc0, vf_cut=fcf)
            ax.imshow(M, extent=[0,tf, fc0/1000.0, fcf/1000.0], origin='lower', aspect = aspect, cmap=plt.cm.bone_r)
            if axTitle : ax.set_title(' '.join(files[counter].split('-')[3:5]), size=8 )
            counter+=1

    if title : fig.suptitle(title)
    if outFig : fig.savefig(outFig, bbox_inches='tight')

##### WAVFILE LOADING ######        

def wav2waveform(wavF, fs=None, **kwargs):
    """reads wave file and returns (waveform, fs)
    for **kwargs, see librosa.core.load"""
    return _wav2waveform(wavF, sr=fs, **kwargs)


def _wav2waveform(wavF, **kwargs):
    "read wavfile and return fs, waveform"
    assert os.path.isfile(wavF), " ---> {}\n\tFile not found!".format(wavF)
    y, sr = librosa.core.load(wavF, **kwargs)

    return y, sr


def plWave(wavFi, dirN='', outFig='', title='', figsize=None, normalize=True):
    """
    files is an array with the files names files
    wavFi : path to wav file
    * spec_factor, 0 = nothing, 1 = all
    """
    waveform, fs = wav2waveform(wavFi, normalize=normalize)

    tf = 1.0*(len(waveform)-0.1)/fs
    #print "sampling rate:", fs
    fig, ax =  plt.subplots(figsize=figsize)

    ax.plot(np.linspace(0, tf, len(waveform)), waveform)
    ax.set_xlabel('time (s)')
    ax.set_ylim(-np.abs(np.max(waveform)),np.abs(np.max(waveform)))


    if title : fig.suptitle(title)
    if outFig : fig.savefig(outFig, bbox_inches='tight')


###########################################################
#####              ON spectral matrices               #####
###########################################################

# functions


def fitNinSqr(N):
    """
    Returns the number of columns and rows to optimally plot N images 
    together, used by plotFiles
    """
    nC = np.ceil(np.sqrt(N))
    if nC*(nC-1)>=N:
        return int(nC), int(nC-1)
    else:
        return int(nC), int(nC)

def reeScale_E(M, spec_factor = 1.0/3.0):
    """
    Zeroes the noise by taking only the part of the spectrum with the highest energy.
    - spec_factor \in [0,1],
    --- 0 - max cutting energy (we don't see anything)
    --- 1 - min cutting energy (returns M without doing anything )
    * M, log (spectrogram)
    """

    assert(spec_factor >= 0 and spec_factor <= 1)
    cutE = (np.min(M) - np.max(M))*spec_factor + np.max(M)
    nx, ny = np.shape(M)
    M_tr = np.copy(M)

    for i in range(nx):
        for j in range(ny):
            if M_tr[i,j] < cutE: M_tr[i, j] = cutE

    return M_tr

def selectBand(M, fr_0 = 0, fr_f = 24000, v0_cut = 1.0*1000, vf_cut = 20.0*1000):
    """
    selects a band on frequencies from the matrix M
    fr_0, initial frequency of the matrix
    fr_f, final frequency of the matrix, sampR/2
    cutting frequencies
    v0_cut
    vf_cut
    """
    ny, nx = np.shape(M)
    n0_cut = int( ny*v0_cut/( fr_f - fr_0 ) )
    nf_cut = int( ny*vf_cut/( fr_f - fr_0 ) )
    #print n0_cut, nf_cut, nx, ny
    return M[ n0_cut:nf_cut, : ]

def allPositive_andNormal(M):
    """
    normalises the matrices, so that all it's values lay in (0,1)
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
        print( "WARNING: matrix is to large", nt)


def myBinarize(rawData, Nbits = 7):
    """
    This function binarises a matrix preserving the number of columns (time wins)
    """
    a = np.min(rawData)*1.0
    b = 1.0*np.max(rawData)-a

    Nf, Nt = np.shape(rawData)
    binData = np.zeros(( Nf*Nbits, Nt))

    for i in np.arange(Nt): # column loop - time
        binariSet = np.zeros((Nbits*Nf))

        for j in np.arange(Nf): #row loop -- frequency
            rawPoint = rawData[j,i]
            rawPoint = np.floor( ((rawPoint-a)/(1.0*b))*((2**(Nbits)-1)) ) # rescale the data

            for k in np.arange(Nbits): # binary loop
                binariSet[k+j*Nbits] = np.floor(np.mod( rawPoint/(2.0**(k)), 2))

        binData[:, i] = binariSet

    return binData


def flatPartition(nSlices, vec_size):
    """
    returns the indexes that slice an array of size vec_size into nSlices
    Parameters
    ----------
    nSlices : number of slices
    vec_size : size of the vector to slice
    """
    idx = np.linspace(0, np.arange(vec_size)[-1], nSlices)
    return np.array([int(item) for item  in idx])


##########################################
#####        AUDIO PROCESSING        #####
##########################################


def specgramWav(wav_fileN, NFFT=2**9, overlap=0.5, saveData=False): #, max_freq = 8000):
    """
    this function creates spectrograms form a wav file
    wav_fileN = name of the wavfile we want to plot
    max_freq = the maximum frequency we want to display [KHz]
    powerOfWinLen = fft works optimally with widows sizes of the power of 2
    > M, spectro matrix
    > tf, final time
    > ff, largest frequency
    > paramStr :  string with the used parameters
    """
    # read wav file
    fs, waveform = wavfile.read(wav_fileN)
    return spectralRep(waveform, fs, NFFT=2**10, overlap=0.5, outF=saveData), paramStr

def spectral(waveform, fs, NFFT=2**9, overlap=0.5,
                winN='hanning', outF=None, logSpec=True):
    """
    Extracts the power spectral features from a waveform
    < waveform :  numpy array
    < fs : sampling rate
    < powerOfWinLen : exponent of the fft window length in base 2
    < overlap : [0,1)
    < winN : win
    < logSpec : power spectrum in logarithmic scale
    --->
    > specM : spectral matrix (numpy array, m_instances x n_features )
    > s2f : names of the features. Central frequency of the bin (n,)
    > tf : final time (s2t[-1])
    > paramStr
    """
    # settings
    overlap=float(overlap)
    win = sig.get_window(winN, NFFT) #plt.mlab.window_hanning

    # SPECTROGRAM
    specM, s2f, s2t = mlab.specgram(waveform, NFFT=NFFT, Fs=fs, window=win,
                                    noverlap=int(NFFT*overlap))
																																				
    paramStr = 'NFFT%d-OV%d'%(NFFT, overlap*100)
    if logSpec: specM = 10. * np.log10(specM)

    return specM.T # transpose the matrix to have the (m x n) form


def spectral_nDelta(waveform, fs, NFFT=2**9, overlap=0.5, winN='hanning',
                order=1, logSc=True):
    # settings
    win = sig.get_window(winN, NFFT) #plt.mlab.window_hanning\
    overlap=float(overlap)
    order=int(order)

    ## SPECTROGRAM
    specM, featureNames, s2t, x = plt.specgram(waveform, NFFT=NFFT, Fs=fs, window=win,
                                    noverlap=int(NFFT*overlap))
    featureNames=list(featureNames)
    Mspec_log = np.log(specM)
    M = specM.copy()
    if logSc: M = np.log(M)

    for dO in np.arange(1,order+1):
        M = np.vstack((M, delta(Mspec_log, order=dO)))

    M = M.T

    return M


###########################################################
#####                 cepstrograms                    #####
###########################################################

def cepstral(waveform, fs, NFFT=2**9, overlap=0.5, Nceps=2**4, logSc=True,
             **kwargs):
    """Extracts cepstral features from a waveform
    Parameters
    ----------
    < waveform : numpy array
    < fs : sampling rate
    < NFFT : fft window length
    < overlap : [0,1)
    < Nceps : number of mfcc coefficients
    < logSc : return features in logarithmic scale
    Returns    
    ---
    > specM : spectral matrix (numpy array, ( m_instances x n_features ) )
    
    See also
    --------
        mfcepstral : cepstrogram
    """
    ## settings

    log_spec_y = librosa.core.logamplitude( librosa.core.spectrum._spectrogram(waveform, n_fft=NFFT, 
                                              hop_length=int(NFFT*(1-overlap), **kwargs))[0])

    dct_filt = librosa.filters.dct( Nceps, len(log_spec_y))
    cepsRep = np.dot(dct_filt, log_spec_y).T
    if logSc:
        cepsRep = librosa.core.logamplitude(cepsRep)
    return cepsRep

def mfcepstral(waveform, fs, NFFT=2**9, overlap=0.5, Nceps=2**4, logSc=True, n_mels=128,
             **kwargs):
    """
    Extracts MFCC from a waveform
    Parameters:
    ----------
    < waveform : numpy array
    < fs : sampling rate
    < NFFT : fft window length
    < overlap : [0,1)
    < Nceps : number of mfcc coefficients
    < logSc : return features in logarithmic scale
    --->
    > specM : spectral matrix (numpy array, ( m_instances x n_features ) )
    
    See also
    --------
        cepstral : Mel-frequency cepstrogram
    """
    ## settings																

    return mfcepstral_nDdelta(waveform, fs, NFFT=NFFT, overlap=overlap, Nceps=Nceps,
                logSc=logSc, order=0, n_mels=n_mels, **kwargs)

def mfcepstral_nDdelta(waveform, fs, NFFT=2**9, overlap=0.5, Nceps=2**4,
                order=1, logSc=True, n_mels=128, **kwargs):

    """
    mfcc feature matrix and the delta orders horizontally appended
    Parameters:
    ------------
        < waveform : numpy array
        < fs : samplig rate
        < NFFT : fft window length in base 2
        < overlap : [0,1)
        < Nceps : int,
            number of MFcepstral coefficients (n_mfcc) mel-spectral filters
        < logSc : return features in logarithmic scale
        < order : orders of the derivative 0->MFCC, 1->delta, 2-> delta-delta
    Returns:
    -------
        > M : mfcepstral feature matrix ( m_instances x n_features )
        > featureNames : list
        > tf : final time [s]
        > paramStr : settings string
    """
    ## settings																
    overlap = float(overlap)
    hopSz = int(NFFT*(1 - overlap))
    Nceps = int(Nceps)
    paramStr = '-Nceps%d'%(Nceps)
    ## CEPSTROGRAM
    M0 = librosa.feature.mfcc(waveform, sr=fs, n_mels=n_mels, n_mfcc=Nceps, 
                         n_fft=NFFT, hop_length=hopSz, **kwargs)#, hop_length=hopSz)
    m = np.shape(M0)[1]
    M=np.zeros((Nceps*(order+1), m))
    M[:Nceps, :] = M0
    ## deltas
    for dO in np.arange(1,order+1):
        #print(dO)
        M[Nceps*dO:Nceps*(dO+1), :] = delta(M0, order=dO)

    M=M.T
    tf = 1.*len(waveform) / fs

    return M


def melSpectral_nDelta(waveform, fs, NFFT=2**10, overlap=0.5, n_mels=2**4,
                       order=1, logSc=True, **kwargs):

    """
    melspectrum Feature Matrix and the delta orders horizontally appended
    Parameters:
    -----------
        < waveform : numpy array
        < fs : samplig rate
        < NFFT : fft window length in base 2
        < overlap : [0,1)
        < n_mels : number of mel filterbanks
        < logSc : return features in logarithmic scale
        < order : orders of the derivative 0->MFCC, 1->delta, 2-> delta-delta
    Returns:
    --------
        > M : mfcepstral feature matrix ( m_instances x n_features )
        > featureNames : list
        > tf : final time [s]
        > paramStr : settings string
    """
    ## settings																
    overlap = float(overlap)
    hopSz = int(NFFT*(1 - overlap) )
    n_mels = int(n_mels)			

    ## CEPSTROGRAM
    Mc = librosa.feature.melspectrogram(waveform, sr=fs, n_mels=n_mels,
                                   n_fft=NFFT, hop_length=hopSz, **kwargs)
    Mc_log = np.log(Mc)
    M = Mc.copy()
    if logSc: M = np.log(M)
    n = np.shape(M)[0]
    ## deltas
    for dO in np.arange(1, order + 1):
        #print(dO)
        M = np.vstack((M, delta(Mc_log, order=dO)))


    M = M.T
    tf = 1.*len(waveform) / fs

    return M



###########################################################
#####                   chroma                        #####
###########################################################

def chromogram(waveform, fs, C=None, hop_length=512, fmin=None,
			threshold=0.0, tuning=None, n_chroma=12, n_octaves=7,
			window=None, bins_per_octave=None, mode='full'):
														
    """
    Extracts the spectral features from a waveform
    < waveform : numpy array
    < fs : sampling rate
    < NFTTpow : exponent of the fft window length in base 2
    < overlap : [0,1)
    < Nceps : number of cepstral coefficients
    < logSc : return features in logarithmic scale
    --->
    > specM : spectral matrix (numpy array, n x m)
    > featureNames : names of the features. Central frequency of the bin (n,)
    > tf : final time (s2t[-1])
    > paramStr :  string with the used parameters
    """
    ## settings
								
    #y_harmonic, y_percussive = librosa.effects.hpss(y)
    C = librosa.feature.chroma_cqt(y=waveform, sr=fs, C=C, hop_length=hop_length,
			fmin=fmin, threshold=threshold, tuning=tuning,
			n_chroma=n_chroma, n_octaves=n_octaves, window=window,
			bins_per_octave=bins_per_octave, mode=mode)

    C=C.T

    return C


###########################################################
#####                   delta                        #####
###########################################################

def delta(M, width=9, order=1, axis=0, trim=True):
    """
    M :  feature matrix (m_instances x n_features)
    axis : int [scalar]
        the axis along which to compute deltas.
        Default is 0 (rows).
    trim      : bool
        set to `True` to trim the output matrix to the original size.
    """
    dM = librosa.feature.delta(M, width=width, order=order, axis=axis, trim=trim)
    return dM


########   Filters   ########

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = fs/2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(waveform, fs, lowcut, highcut, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, waveform)
    return y


def bandDecomposition(x, fs, intervals=None,
                      filFun = butter_bandpass_filter, order=3, Nbins=4):
    """
    band decomposition of a signal filter

    < x, input signal
    < fs, sampling rate
    < intervals, list of tuples with intervals of the bands.
                if None, then (Nbins) are defined logarithmic sizes
    < order, order of the butter filter
    < bins, used when the intervals are defined (default log)
    -->
    > yLi, dictionary of the band decomposition of x
    """

    if intervals == None:
        li = np.array([int(item) for item in np.logspace(np.log(1000), np.log(fs/2), num=Nbins, base=np.e)])
        intervals = []
        i0 = 100
        for item in li:
            intervals.append((i0, item))
            i0 = item
    assert type(intervals) == list, '! intervals must be a list'

    yLi = {}
    for (lcut, ucut) in intervals:
        print("input",lcut, ucut, np.shape(x))
        yLi[(lcut, ucut)] = butter_bandpass_filter(x, lcut, ucut, fs, order=order)

    return(yLi)

def whiten(waveForm, psd_whittener, Fs):
    """
    Noise whitening
    Divide by ASD in the frequency domain, and transform back.
    Parameters:
    ----------
        waveForm : numpy array
        psd_whitener : power spectrum with the frequencies to use for whitening
        Fs : sampling rate
    Returns:
    --------
        whitened waveform : np.array
    """
    Nt = len(waveForm)
    freqs = np.fft.rfftfreq(Nt, Fs)

    
    hf = np.fft.rfft(waveForm)
    white_hf = hf / (np.sqrt(psd_whittener(freqs) /Fs/2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht    


######    Spectral energy    ######

def bandEnergy(y, fs, f_band=None, nps=256, overl=None):
    """
    sum of the power spectral density within a freq interval using
    Welch's method

    < takes
    --------
    y : input signal
    fs : sampling rate of y
    f_band : frequencies interval, default (1000, fs/2)
    nps : nperseg

    > returns
    ----------
    sum (X_k), k in f_band
    """

    if f_band==None: f_band=(1000, fs/2)

    fr, Pxx_den = sig.welch(y, fs, nperseg=nps, noverlap=overl)
    ix = np.logical_and(fr > f_band[0], fr < f_band[1] )
    D = { str(f_band) : np.sum(Pxx_den[ix]) }

    #return( np.sum(Pxx_den[ix]) )
    return(D)

def welchD(y, fs, f_band=None, nps=256, overl=None):
    """
    sum of the power spectral density with in a freq interval using
    Welch's method

    < takes
    --------
    y : input signal
    fs : sampling rate of y
    f_band : frequencies interval, default (1000, fs/2)
    nps : nperseg

    > returns
    ----------
    sum (X_k), k in f_band
    """

    if f_band==None: f_band=(1000, fs/2)

    fr, Pxx_den = sig.welch(y, fs, nperseg=nps, noverlap=overl)
    ix = np.logical_and(fr > f_band[0], fr < f_band[1] )
    D = dict( zip( fr[ix], Pxx_den[ix] ) )

    return(D)

def tsBandenergy(y, fs, textureWS=0.1, textureWSsamps=0, overlap=0,
                 freqBand=None, normalize=True):
    """
    returns  a time series with the spectral energy every textureWS (seconds)
    overlap [0,1)
    < y : signal (waveform)
    < fs : sampling rate
    * optional
    < textureWS : texture window in seconds
    < textureWSsamps : texture window in number of samples (This has priority over textureWS)
    < normalize : the powerspectrum is normalized over the samples of y
    -->
    > time series of the powerspectral density
    > time interval of the texture window
    """

    ## set set (size of the texture window in samples)
    if textureWS: step = 2**np.max((2, int(np.log(fs*textureWS)/np.log(2)))) # as a power of 2
    if textureWSsamps: step = textureWSsamps

    overl = int(step*overlap)
    PSts = []
    ix0 = 0
    ix = step
    print("step:", step, "overlap", overl, "len signal", len(y), "step (s)", 1.0*step/fs)
    while ix < len(y):
        yi = y[ix0:ix] # signal section
        ix0 = ix - overl
        ix = ix0 + step

        PSts.append(bandEnergy(yi, fs, f_band=freqBand)) # append the energy

    PSts_arr = np.array(PSts)
    if normalize : PSts_arr = PSts_arr/np.max(PSts_arr)
    #sprint(len(PSts))
    return(PSts_arr, 1.0*step/fs)

