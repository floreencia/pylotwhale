from __future__ import print_function, division
import functools

import sys
import os.path
import scipy.signal as sig
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
from sklearn.preprocessing import scale, maxabs_scale, minmax_scale

from scipy.io import wavfile
import librosa as lf  # Librosa for audio

"""
Backup of signalTools
TODO: this module will be removed
"""


def audioFeaturesFun(funName=None):
    """
    Dictionary of feature extracting functions
    returns a callable
    ------
    > feature names (if None)
    > feature function
        this functions take the waveform and return an instantiated feature matrix
        m (instances) - rows
        n (features) - columns
    """
    D = {  #'welch' : welchD,
        #'bandEnergy' : bandEnergy, ## sum of the powerspectrum within a band
        "spectral": spectralRep,
        "spectralDelta": functools.partial(spectralDspecRep, order=1),
        "cepstral": cepstralRep,
        "cepsDelta": functools.partial(cepstralDcepRep, order=1),  # MFCC and delta-MFCC
        "cepsDeltaDelta": functools.partial(cepstralDcepRep, order=2),
        "chroma": chromaRep,
        "melspectroDelta": melSpecDRep,
        "melspectro": functools.partial(melSpecDRep, order=0),
    }

    if funName in D.keys():  # returns a list of possible feature names
        return D[funName]
    else:
        return D  # returns function name of the asked feature


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


def normalizeWF(waveform):
    warnings.warn("use maxAbsScale", DeprecationWarning)
    return 1.0 * waveform / np.max(np.abs(waveform))


def tileTillN(arr, N, n0=0):
    """returns an array of size N (>0) from tiling of arr. n0 is the starting index"""
    # np.tile(arr, int(n/len(arr))+1)[:n]
    return arr[np.array([i for i in np.arange(n0, N + n0) % len(arr)])]


def addToSignal(y1, y2, noiseIndex):
    """
    adds y2 (noise) to the primary signal y1. Returns the sum, keeping the size of y1
    """
    return y1 + tileTillN(y2, len(y1), noiseIndex)


def generateAddEnsemble(y_template, y_add, intensity_grid=None):
    """
    generate an ensemble of y_template-singnals adding y_add
    normalizes both signals and adds different amplitudes of y_add to y_template
    Returns:
    Y : a matrix, with the sum of y_template and y_add in each row
    """
    if intensity_grid is None:
        intensity_grid = np.linspace(0.1, 10, 10)

    # print(len(intensity_grid), len(y_template))
    Y = np.zeros((len(intensity_grid), len(y_template)))
    y_template = normalizeWF(y_template)
    y_add = normalizeWF(y_add)
    for i in range(len(intensity_grid)):
        Y[i, :] = addToSignal(
            y_template, intensity_grid[i] * y_add, np.random.randint(0, len(y_template))
        )
        # y_template + intensity_grid[i]*tileTillN(y_add, len(y_template), np.random.randint(0,len(y_template)))

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

    # print(len(intensity_grid), len(y_template))
    Y = np.zeros((len(shift_grid), len(y_template)))
    for i in range(len(shift_grid)):
        Y[i, :] = lf.effects.pitch_shift(y_template, fs, shift_grid[i])
        # y_template + intensity_grid[i]*tileTillN(y_add, len(y_template), np.random.randint(0,len(y_template)))

    return Y


def generateTimeStreachEnsemble(y_template, streach_grid=None):
    """
    generate an ensemble of y_template-signals adding y_add
    normalizes both signals and adds different amplitudes of y_add to y_template
    Returns:
    Y : a matrix, with the sum of y_template and y_add in each row
    """
    if streach_grid is None:
        streach_grid = np.linspace(0.8, 1.2, 5)

    # print(len(intensity_grid), len(y_template))
    Y = []
    for i in range(len(streach_grid)):
        Y.append(lf.effects.time_stretch(y_template, streach_grid[i]))

    return Y


def freqshift(data, Fs, fshift=100):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data) / float(Fs)
    df = 1.0 / T
    nbins = int(fshift / df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    z = np.fft.irfft(y)
    return z


##### wav files


def write_wavfile(filename, fs, data):
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename, int(fs), d)


###########################################################
#####                     plotting                    #####
###########################################################


def wavs2spectros(
    files,
    dirN="",
    outFig="",
    title="",
    winPow=9,
    over=0.5,
    axTitle=True,
    fc0=0,
    fcf=120,
    aspect="auto",
    figScale=1,
    spec_factor="0.6",
):
    """
    < files, an array with the wav file names
    * spec_factor, 0 = nothing, 1 = all
    """
    n = len(files[:12])
    nC, nR = fitNinSqr(n)  # print "size ", nR, nC
    fig, axes = plt.subplots(
        nrows=nR, ncols=nC, figsize=(int(nR) * 4 * figScale, int(nC) * 2 * figScale)
    )
    counter = 0

    for axR in axes:
        for ax in axR:
            thisFile = os.path.join(dirN, files[counter]) if dirN else files[counter]
            M0, tf, ff = specgramWav("%s" % thisFile, powerOfWinLen=winPow, overlap=over)
            M = reeScale_E(np.log(M0), spec_factor=spec_factor)
            M = selectBand(M, v0_cut=fc0, vf_cut=fcf)
            ax.imshow(
                M,
                extent=[0, tf, fc0 / 1000.0, fcf / 1000.0],
                origin="lower",
                aspect=aspect,
                cmap=plt.cm.bone_r,
            )
            if axTitle:
                ax.set_title(" ".join(files[counter].split("-")[3:5]), size=8)
            counter += 1

    if title:
        fig.suptitle(title)
    if outFig:
        fig.savefig(outFig, bbox_inches="tight")


def wav2waveform(wavF, normalize=False):
    """reads wave file and returns (waveform, sr)"""
    return _wav2waveform(wavF, normalize=normalize)


def _wav2waveform(wavF, normalize=False):
    "read wavfile and return sRate, waveform"
    try:
        sRate, waveform = wavfile.read(wavF)
        if normalize:
            waveform = normalizeWF(waveform)
        return waveform, sRate
    except IOError:
        print("Oops!  Couldn't read:\n %s " % wavF)
        return IOError


def plWave(wavFi, dirN="", outFig="", title="", figsize=None, normalize=True):
    """
    files is an array with the files names files
    wavFi : path to wav file
    * spec_factor, 0 = nothing, 1 = all
    """
    waveform, sRate = wav2waveform(wavFi, normalize=normalize)

    tf = 1.0 * (len(waveform) - 0.1) / sRate
    # print "sampling rate:", sRate
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(np.linspace(0, tf, len(waveform)), waveform)
    ax.set_xlabel("time (s)")
    ax.set_ylim(-np.abs(np.max(waveform)), np.abs(np.max(waveform)))

    if title:
        fig.suptitle(title)
    if outFig:
        fig.savefig(outFig, bbox_inches="tight")


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
    if nC * (nC - 1) >= N:
        return int(nC), int(nC - 1)
    else:
        return int(nC), int(nC)


def reeScale_E(M, spec_factor=1.0 / 3.0):
    """
    Zeroes the noise by taking only the part of the spectrum with the higest energy.
    - spec_factor \in [0,1],
    --- 0 - max cutting energy (we don't see anything)
    --- 1 - min cutting energy (returns M without doing anything )
    * M, log (spectrogram)
    """

    assert spec_factor >= 0 and spec_factor <= 1
    cutE = (np.min(M) - np.max(M)) * spec_factor + np.max(M)
    nx, ny = np.shape(M)
    M_tr = np.copy(M)

    for i in range(nx):
        for j in range(ny):
            if M_tr[i, j] < cutE:
                M_tr[i, j] = cutE

    return M_tr


def selectBand(M, fr_0=0, fr_f=24000, v0_cut=1.0 * 1000, vf_cut=20.0 * 1000):
    """
    selects a band on frequencies from the matrix M
    fr_0, initial frequency of the matrix
    fr_f, final frequency of the matrix, sampR/2
    cutting frequencies
    v0_cut
    vf_cut
    """
    ny, nx = np.shape(M)
    n0_cut = int(ny * v0_cut / (fr_f - fr_0))
    nf_cut = int(ny * vf_cut / (fr_f - fr_0))
    # print n0_cut, nf_cut, nx, ny
    return M[n0_cut:nf_cut, :]


def allPositive_andNormal(M):
    """
    normalizes the matrices, so that all it's values lay in (0,1)
    """
    if np.min(M) < 0:
        M = M - np.min(M)
    M = 1.0 * M / np.max(M)
    return M


def reeSize_t(M, h_size=3938):
    ny, nt = np.shape(M)
    if h_size - nt > 0:
        d = h_size - nt
        reM = np.concatenate((np.zeros((ny, d / 2)), M), axis=1)

        return np.concatenate((reM, np.zeros((ny, h_size - np.shape(reM)[1]))), axis=1)
    else:
        print("WARNING: matrix is to large", nt)


def myBinarize(rawData, Nbits=7):
    """
    This function binarises a matrix preserving the number of columns (time wins)
    """
    a = np.min(rawData) * 1.0
    b = 1.0 * np.max(rawData) - a

    Nf, Nt = np.shape(rawData)
    binData = np.zeros((Nf * Nbits, Nt))

    for i in np.arange(Nt):  # column loop - time
        binariSet = np.zeros((Nbits * Nf))

        for j in np.arange(Nf):  # row loop -- frequency
            rawPoint = rawData[j, i]
            rawPoint = np.floor(
                ((rawPoint - a) / (1.0 * b)) * ((2 ** (Nbits) - 1))
            )  # reescale the data

            for k in np.arange(Nbits):  # binary loop
                binariSet[k + j * Nbits] = np.floor(np.mod(rawPoint / (2.0 ** (k)), 2))

        binData[:, i] = binariSet

    return binData


def plspectro(
    waveform,
    sRate,
    outF="",
    N=2 ** 9,
    v0_cut=1000,
    vf_cut=20 * 1000,
    overFrac=0.5,
    winN="hanning",
    spec_fac=0.99999,
    plTitle="",
    plTitleFontSz=0,
    cmN="bone_r",
    figsize=None,
):

    # tf = float(N)/sRate
    tf = 1.0 * (len(waveform)) / sRate
    # ff = sRate/2.0
    N = int(N)
    win = sig.get_window(winN, N)
    noverlap = int(overFrac * N)
    A0 = plt.specgram(waveform, Fs=sRate, NFFT=N, noverlap=noverlap, window=win)[0]

    # Spectro edditing
    A = selectBand(A0, fr_f=sRate / 2, v0_cut=v0_cut, vf_cut=vf_cut)  # band filter
    A = reeScale_E(
        A, spec_factor=spec_fac
    )  # zero the the spectral energy smaller than 0.001% of <E>

    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)  # figsize=(max([3,int(tf/0.3)]), ff/8000))

    cax = ax.imshow(
        np.log(A),
        extent=[0, tf, v0_cut / 1000, vf_cut / 1000],
        origin="lower",
        aspect="auto",
        cmap=plt.cm.get_cmap(cmN),
    )  # , interpolation = 'nearest')

    # labels
    ax.set_xlabel("time [s]")  # , fontsize=16)
    ax.set_ylabel("frequency [KHz]")  # , fontsize=16)
    if plTitleFontSz:
        ax.set_title(plTitle, fontsize=plTitleFontSz)

    if tf < 1:
        plt.xticks(np.arange(0, tf, tf / 2.0))
    else:
        plt.xticks(np.arange(0, tf, 1.0))

    # cbar
    (ll, ml, ul) = cbarLabels(np.log(A).min(), np.log(A).max())
    cbar = fig.colorbar(cax, ticks=[ll, ml, ul])
    cbar.ax.set_yticklabels(["10$^{%d}$" % ll, "10$^{%d}$" % ml, "10$^{%d}$" % ul])

    # save
    #    outF = baseSpecN+'.jpg'
    if outF:
        print("out:", outF)
        fig.savefig(outF, bbox_inches="tight")


def flatPartition(nSlices, vec_size):
    """
    returns the indexes that slice an array of size vec_size into nSlices
    Parameters
    ----------
    nSlices : number of slices
    vec_size : size of the vector to slice
    """
    idx = np.linspace(0, np.arange(vec_size)[-1], nSlices)
    return np.array([int(item) for item in idx])


##########################################
#####        AUDIO PROCESSING        #####
##########################################


###########################################################
#####                 spectrograms                    #####
###########################################################


def cbarLabels(minV, maxV):
    """
    give me the maximum and the minimum values of colour bar and I will return 3 label
    the labels returned are int type. Meant for exponents.
    """
    minV = int(np.ceil(minV))
    maxV = int(np.floor(maxV))
    ml = (minV + maxV) / 2
    # print minV, maxV,"ml", ml
    D = min(abs(ml - maxV), abs(minV - ml))
    ul = ml - D
    ll = ml + D
    # print D, ul, ll
    return (ll, ml, ul)


def specgramWav(wav_fileN, NFFTpow=12, overlap=0.5, saveData=False):  # , max_freq = 8000):
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
    sRate, waveform = wavfile.read(wav_fileN)
    paramStr = "NFFT%d-OV%d" % (2 ** NFFTpow, overlap * 100)
    return spectralRep(waveform, sRate, NFFTpow=12, overlap=0.5, outF=saveData), paramStr


def spectralRep(waveform, sRate, NFFTpow=9, overlap=0.5, winN="hanning", outF=None, logSpec=True):
    """
    Extracts the power spectral features from a waveform
    < waveform :  numpy array
    < sRate : sampling rate
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
    NFFT = 2 ** int(NFFTpow)
    overlap = float(overlap)
    win = sig.get_window(winN, NFFT)  # plt.mlab.window_hanning

    # SPECTROGRAM
    specM, s2f, s2t = mlab.specgram(
        waveform, NFFT=NFFT, Fs=sRate, window=win, noverlap=int(NFFT * overlap)
    )

    paramStr = "NFFT%d-OV%d" % (NFFT, overlap * 100)
    if logSpec:
        specM = 10.0 * np.log10(specM)
    # specM = np.flipud(specM)

    return specM.T, s2f, s2t[-1], paramStr  # transpose the matrix to have the (m x n) form


def spectralDspecRep(waveform, sRate, NFFTpow=9, overlap=0.5, winN="hanning", order=1, logSc=True):
    # settings
    NFFT = 2 ** int(NFFTpow)
    win = sig.get_window(winN, NFFT)  # plt.mlab.window_hanning\
    overlap = float(overlap)
    order = int(order)

    ## SPECTROGRAM
    specM, featureNames, s2t, x = plt.specgram(
        waveform, NFFT=NFFT, Fs=sRate, window=win, noverlap=int(NFFT * overlap)
    )
    featureNames = list(featureNames)
    Mspec_log = np.log(specM)
    M = specM.copy()
    if logSc:
        M = np.log(M)
    paramStr = "NFFT%d-OV%d" % (NFFT, overlap * 100)

    # n = np.shape(M)[0]
    ## deltas
    for dO in np.arange(1, order + 1):
        # print(dO)
        M = np.vstack((M, delta(Mspec_log, order=dO)))
        featureNames += ["delta%s" % dO + str(cc) for cc in featureNames]
        paramStr += "-delta%s" % dO

    M = M.T
    tf = 1.0 * len(waveform) / sRate

    return M, featureNames, tf, paramStr


def saveSpecgram(
    wav_fileN, powerOfWinLen=12, overlap=0.9, freqFrac=1.0, outDir="", figsize=None, cmN="gray_r"
):  # , max_freq = 8000):
    """
    Draw spectrogram form a wav file
    Parameters:
    -----------
        wav_fileN : name of the wavfile we want to plot
        max_freq : the maximum frequency we want ot display [KHz]
        powerOfWinLen : log(NFFT)/log(2). fft works optimally with widows 
                        sizes of the power of 2
        freqFrac = fraction of the max frequency
        outDir='' # if no output dir is given the image is stored in the same dir
                as the wav file
    """
    # settings
    M, tf, ff = specgramWav(wav_fileN, powerOfWinLen=powerOfWinLen, overlap=overlap)

    # ploting
    outF = os.path.splitext(wav_fileN)[0] + "-%dHann%dSpecZoom.png" % (100 * overlap, powerOfWinLen)
    if outDir:
        outF = os.path.join(outDir, os.path.basename(outF))
    print(outF)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.imshow(
        np.log10(M),
        extent=[0, tf, 0, ff / 1000],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=plt.cm.get_cmap(cmN),
    )
    plt.ylim([0, ff * freqFrac / 1000])

    # cbar
    (ll, ml, ul) = cbarLabels(np.log10(M).min(), np.log10(M).max())
    cbar = fig.colorbar(cax, ticks=[ll, ml, ul])
    cbar.ax.set_yticklabels(
        ["10$^{%d}$" % ll, "10$^{%d}$" % ml, "10$^{%d}$" % ul]
    )  # , size='x-large')# vertically oriented colorbar

    ax.set_xlabel("time [s]", fontsize=16)
    ax.set_ylabel("frequency [KHz]", fontsize=16)
    # ax.tick_params(axis='both', labelsize='x-large')  #set_xticks(ticks, fontsize=24)
    plt.savefig(outF)


###########################################################
#####                 cepstrograms                    #####
###########################################################


def cepstralRep(waveform, sRate, NFFTpow=9, overlap=0.5, Nceps=2 ** 4, logSc=True, n_mels=128):
    """
    Extracts the spectral features from a waveform
    Parameters:
    ----------
    < waveform : numpy array
    < sRate : samplig rate
    < NFFTpow : exponent of the fft window lenght in base 2
    < overlap : [0,1)
    < Nceps : number of cepstral coefficients
    < logSc : return features in logarithmic scale
    --->
    > specM : spectral matrix (numpy array, ( m_instances x n_features ) )
    > featureNames : names of the features. Central frequency of the bin (n,)
    > tf : final time (s2t[-1])
    > paramStr :  string with the used parameters
    """
    ## settings

    return cepstralDcepRep(
        waveform,
        sRate,
        NFFTpow=NFFTpow,
        overlap=overlap,
        Nceps=Nceps,
        logSc=logSc,
        order=0,
        n_mels=n_mels,
    )


def cepstralDcepRep(
    waveform, sRate, NFFTpow=10, overlap=0.5, Nceps=2 ** 4, order=1, logSc=True, n_mels=128
):

    """
    cepstral feature matrix and the delta orders horizontally appended
    Parameters:
    ------------
        < waveform : numpy array
        < sRate : sampling rate
        < NFFTpow : exponent of the fft window lenght in base 2
        < overlap : [0,1)
        < Nceps : int,
            number of cepstral coefficients (n_mfcc) melspectral filters
        < logSc : return features in logarithmic scale
        < order : orders of the derivative 0->MFCC, 1->delta, 2-> delta-delta
    Returns:
    -------
        > M : cepstral feature matrix ( m_instances x n_features )
        > featureNames : list
        > tf : final time [s]
        > paramStr : settings string
    """
    ## settings
    NFFT = 2 ** int(NFFTpow)
    overlap = float(overlap)
    hopSz = int(NFFT * (1 - overlap))
    Nceps = int(Nceps)
    paramStr = "-Nceps%d" % (Nceps)
    ## CEPSTROGRAM
    M0 = lf.feature.mfcc(
        waveform, sr=sRate, n_mels=n_mels, n_mfcc=Nceps, n_fft=NFFT, hop_length=hopSz
    )  # , hop_length=hopSz)
    m = np.shape(M0)[1]
    M = np.zeros((Nceps * (order + 1), m))
    M[:Nceps, :] = M0
    featureNames = ["MFCC" + str(cc) for cc in range(Nceps)]
    ## deltas
    for dO in np.arange(1, order + 1):
        # print(dO)
        M[Nceps * dO : Nceps * (dO + 1), :] = delta(M0, order=dO)
        featureNames += ["delta%s" % dO + "MFCC" + str(cc) for cc in range(Nceps)]
        paramStr += "-delta%s" % dO

    M = M.T
    tf = 1.0 * len(waveform) / sRate

    return M, featureNames, tf, paramStr


def melSpecDRep(waveform, sRate, NFFTpow=10, overlap=0.5, n_mels=2 ** 4, order=1, logSc=True):

    """
    melspectrum Feature Matrix and the delta orders horizontally appended
    Parameters:
    -----------
        < waveform : numpy array
        < sRate : sampling rate
        < NFFTpow : exponent of the fft window lenght in base 2
        < overlap : [0,1)
        < n_mels : number of mel filterbanks
        < logSc : return features in logarithmic scale
        < order : orders of the derivative 0->MFCC, 1->delta, 2-> delta-delta
    Returns:
    --------
        > M : cepstral feature matrix ( m_instances x n_features )
        > featureNames : list
        > tf : final time [s]
        > paramStr : settings string
    """
    ## settings
    NFFT = 2 ** int(NFFTpow)
    overlap = float(overlap)
    hopSz = int(NFFT * (1 - overlap))
    n_mels = int(n_mels)
    paramStr = "NFFT%d-OV%d-Nmels%d" % (NFFT, overlap * 100, n_mels)

    ## CEPSTROGRAM
    Mc = lf.feature.melspectrogram(waveform, sr=sRate, n_mels=n_mels, n_fft=NFFT, hop_length=hopSz)
    Mc_log = np.log(Mc)
    M = Mc.copy()
    if logSc:
        M = np.log(M)
    n = np.shape(M)[0]
    featureNames = ["melSpec" + str(cc) for cc in range(n)]
    ## deltas
    for dO in np.arange(1, order + 1):
        # print(dO)
        M = np.vstack((M, delta(Mc_log, order=dO)))
        featureNames += ["delta%s" % dO + "melSpec" + str(cc) for cc in range(n)]
        paramStr += "-delta%s" % dO

    M = M.T
    tf = 1.0 * len(waveform) / sRate

    return M, featureNames, tf, paramStr


def wav2deltaCepsRep(waveform, sRate, NFFTpow=9, overlap=0.5, Nceps=2 ** 4, order=1):

    M, featureNames0, tf, paramStr0 = cepstralRep(
        waveform, sRate, NFFTpow=NFFTpow, overlap=overlap, Nceps=Nceps, logSc=True
    )
    dM = delta(M, order=order)
    featureNames = ["delta%s-" % order + featN for featN in featureNames0]
    paramStr = "delta%s-" % order + paramStr0
    return dM, featureNames, tf, paramStr


'''
def cepstralFeatures(waveform, sRate, analysisWS=0.025, analysisWStep=0.01,
                numcep=13, NFilt=26, NFFT=512, lFreq=0, hFreq=None,
                preemph=0.97, ceplifter=22):
    """
    Extracts the cepstral features from a waveform
    < waveform : numpy array
    < sRate : samplig rate
    < analysisWS : = 0.025 (seconds) ??!!! ...
    ...
    Returns:
    --------
    > specM : spectral matrix (numpy array)
    > s2f : names of the features. Central frquency of the bin
    > tf : final time (s2t[-1])
    """

    # cepstrogram
    cepsM = psf.mfcc(waveform, samplerate=sRate,
         winlen=analysisWS, winstep=analysisWStep, numcep=numcep,
          nfilt=NFilt, nfft=NFFT, lowfreq=lFreq, highfreq=hFreq,
          preemph=preemph, ceplifter=ceplifter, appendEnergy=True)
    #cepsM, s2f, s2t
    tf=len(waveform)/sRate
    featNames = np.arange(np.shape(cepsM)[1])

    #if logSpec : specM = np.log(specM)

    return cepsM, featNames, tf
 '''


def logfbankFeatures(
    waveform,
    sRate,
    analysisWS=0.025,
    analysisWStep=0.01,
    NFilt=26,
    NFFT=512,
    lFreq=0,
    hFreq=None,
    preemph=0.97,
):
    """
    Extracts the spectral features from a waveform
    < waveform : numpy array
    < sRate : sampling rate
    < analysisWS : = 0.025 (seconds) ??!!! ...
    ...
    --->
    > specM : spectral matrix (numpy array)
    > s2f : names of the features. Central frequency of the bin
    > tf : final time (s2t[-1])
    """

    # cepstrogram
    lbfM = psf.logfbank(
        waveform,
        samplerate=sRate,
        winlen=analysisWS,
        winstep=analysisWStep,
        nfilt=NFilt,
        nfft=NFFT,
        lowfreq=lFreq,
        highfreq=hFreq,
        preemph=preemph,
    )
    # cepsM, s2f, s2t
    tf = len(waveform) / sRate
    featNames = np.arange(np.shape(lbfM)[1])

    return lbfM, featNames, tf


###########################################################
#####                   chroma                        #####
###########################################################


def chromaRep(
    waveform,
    sRate,
    C=None,
    hop_length=512,
    fmin=None,
    threshold=0.0,
    tuning=None,
    n_chroma=12,
    n_octaves=7,
    window=None,
    bins_per_octave=None,
    mode="full",
):

    """
    Extracts the spectral features from a waveform
    < waveform : numpy array
    < sRate : samplig rate
    < NFTTpow : exponent of the fft window length in base 2
    < overlap : [0,1)
    < Nceps : number of cepstral coefficients
    < logSc : retunrn features in logarithmic scale
    --->
    > specM : spectral matrix (numpy array, n x m)
    > featureNames : names of the features. Central frequency of the bin (n,)
    > tf : final time (s2t[-1])
    > paramStr :  string with the used parameters
    """
    ## settings
    paramStr = "CHROMA-Hsz%d-chroma%d-octaves%d" % (hop_length, n_chroma, n_octaves)
    # Chromogram

    # y_harmonic, y_percussive = librosa.effects.hpss(y)
    C = lf.feature.chroma_cqt(
        y=waveform,
        sr=sRate,
        C=C,
        hop_length=hop_length,
        fmin=fmin,
        threshold=threshold,
        tuning=tuning,
        n_chroma=n_chroma,
        n_octaves=n_octaves,
        window=window,
        bins_per_octave=bins_per_octave,
        mode=mode,
    )

    C = C.T
    m, n = np.shape(C)
    featureNames = ["chroma" + str(cc) for cc in range(n)]
    tf = 1.0 * len(waveform) / sRate

    return C, featureNames, tf, paramStr


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
    dM = lf.feature.delta(M, width=width, order=order, axis=axis, trim=trim)
    return dM


########   Filters   ########


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(waveform, fs, lowcut, highcut, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, waveform)
    return y


def bandDecomposition(x, fs, intervals=None, filFun=butter_bandpass_filter, order=3, Nbins=4):
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
        li = np.array(
            [int(item) for item in np.logspace(np.log(1000), np.log(fs / 2), num=Nbins, base=np.e)]
        )
        intervals = []
        i0 = 100
        for item in li:
            intervals.append((i0, item))
            i0 = item
    assert type(intervals) == list, "! intervals must be a list"

    yLi = {}
    for (lcut, ucut) in intervals:
        print("input", lcut, ucut, np.shape(x))
        yLi[(lcut, ucut)] = butter_bandpass_filter(x, lcut, ucut, fs, order=order)

    return yLi


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
    white_hf = hf / (np.sqrt(psd_whittener(freqs) / Fs / 2.0))
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

    if f_band == None:
        f_band = (1000, fs / 2)

    fr, Pxx_den = sig.welch(y, fs, nperseg=nps, noverlap=overl)
    ix = np.logical_and(fr > f_band[0], fr < f_band[1])
    D = {str(f_band): np.sum(Pxx_den[ix])}

    # return( np.sum(Pxx_den[ix]) )
    return D


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

    if f_band == None:
        f_band = (1000, fs / 2)

    fr, Pxx_den = sig.welch(y, fs, nperseg=nps, noverlap=overl)
    ix = np.logical_and(fr > f_band[0], fr < f_band[1])
    D = dict(zip(fr[ix], Pxx_den[ix]))

    return D


def tsBandenergy(y, fs, textureWS=0.1, textureWSsamps=0, overlap=0, freqBand=None, normalize=True):
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
    if textureWS:
        step = 2 ** np.max((2, int(np.log(fs * textureWS) / np.log(2))))  # as a power of 2
    if textureWSsamps:
        step = textureWSsamps

    overl = int(step * overlap)
    PSts = []
    ix0 = 0
    ix = step
    print("step:", step, "overlap", overl, "len signal", len(y), "step (s)", 1.0 * step / fs)
    while ix < len(y):
        yi = y[ix0:ix]  # signal section
        ix0 = ix - overl
        ix = ix0 + step

        PSts.append(bandEnergy(yi, fs, f_band=freqBand))  # append the energy

    PSts_arr = np.array(PSts)
    if normalize:
        PSts_arr = PSts_arr / np.max(PSts_arr)
    # sprint(len(PSts))
    return (PSts_arr, 1.0 * step / fs)
