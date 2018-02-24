#!/usr/bin/python

from __future__ import print_function, division
from py.test import raises
import numpy as np
import pylotwhale.MLwhales.featureExtraction as fex

#import pylotwhale.signalProcessing.audioFeatures as auf

### artificial data
wf = np.sin(np.linspace(0, 100, 10000))
fs = 100

def test_Transformation_fun():
    audioFeatDi = {"NFFT":2**9, "overlap" : 0.5, "fs":fs }
    T1 = fex.Transformation('spectral', audioFeatDi)
    fun1 = T1.fun
    M1 = fun1(wf)
    
    summDict = {'n_textWS':5, 'normalise':False}
    T2 = fex.Transformation('walking', summDict)
    fun2 = T2.fun
    M2 = fun2(M1)
    
    tList = [('audio_features', T1), ('summ', T2)]
    Tpipe = fex.TransformationsPipeline(tList)
    Mpipe = Tpipe.fun(wf)
    
    np.testing.assert_almost_equal(M2, Mpipe, decimal=6)

    
    
