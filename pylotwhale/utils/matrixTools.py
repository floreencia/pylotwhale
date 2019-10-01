from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np

"""
this script contains functions frequently used in DSP
"""


def countMatrixEntrances(A):
    """
    returns the frequencies of a matrix
    with positive integer entrances (to use the indexes of the array)
    i.e.  how many elements are with value 0, with value 1, ... with value n
    """
    Nc, Nl = np.shape(A)
    maxA = np.max(A)
    counter = np.zeros(maxA + 1)
    for i in range(Nl):  #
        for j in range(Nc):
            counter[A[i, j]] += 1
    return counter


def colNormalization(A):
    return A / A.sum(axis=0)


def rowNormalization(A):
    return (A.T / A.sum(axis=1)).T


##############################################


def myHarmonic(f0, numHar, t, phase=0.0):
    """
    this function returns a oscillating digital signal (of t) with fundamental frequency f0, and numHar harmonics
    """
    signal = 0.0
    for n in np.arange(1, numHar + 1):
        signal = signal + np.sin(2.0 * np.pi * f0 * n * t + phase)
    # print n, numHar
    return signal / (numHar + 1)


def myStep(start, pointsPstep, mylenght=10000000):
    pStep = 1.0 / pointsPstep
    stepVec = np.round((np.arange(mylenght) * pStep) + 1.0) * start
    # print len(stepVec)
    plt.plot(stepVec)
    return stepVec


def chop(sig):
    """
    This function chops all first points of a signal whose slope is negative
    """
    i = 0
    while True:
        m = sig[i + 1] - sig[i]
        i = i + 1
        # print i, m
        if m >= 0:
            break
    newSig = sig[i - 1 :]
    x = np.arange(len(sig) - len(newSig), len(sig))
    return x, newSig


def chopMatIdx(M):
    """
    this function chops the first set of values with negative slope in the matrix. And returns 
    ! WARNING: Only apply this function over the region with calls. 
    """
    maxIdx = 0
    nc, nl = np.shape(M)
    # print nl

    for i in np.arange(nl):
        idx = chop(M[:, i])[0]
        if idx[0] > maxIdx:
            maxIdx = idx[0]
    # print i, maxIdx
    return np.arange(maxIdx, nc)


def myPitch(Y, X, idx_chop):
    """
	This function takes the cepstrum spectrum Y(X), and returns its pitch 
	by looking at the qfrequency (X) value at the maximum of Y

	TODO: generalize the function so you don't have to give the idx, by using the chopMatIdx
	"""
    pitchIdx = Y[idx_chop].argmax()
    return 1.0 / X[idx_chop[pitchIdx]]
