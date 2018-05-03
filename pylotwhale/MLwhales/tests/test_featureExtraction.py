#!/usr/bin/python

from __future__ import print_function, division
from py.test import raises
import os
import numpy as np
import pylotwhale.MLwhales.featureExtraction as fex

# import pylotwhale.signalProcessing.audioFeatures as auf

#### Data
scriptD = os.path.dirname(os.path.realpath(__file__))
wavF = os.path.join(scriptD, '../../tests/data/WAV_0111_001-48kHz_30-60sec.wav')
annF = os.path.join(scriptD, '../../tests/data/WAV_0111_001-48kHz_30-60sec.txt')

#### Artificial data
wf = np.sin(np.linspace(0, 100, 10000))
fs = 100

#### Transformations
## T1
audioFeatDi = {"NFFT": 2**9, "overlap": 0.5, "fs": fs }
T1 = fex.Transformation('spectral', audioFeatDi)
## T2
summDict1 = {'n_textWS':5, 'normalise': False}
T2 = fex.Transformation('walking', summDict1)
## T3
summDict2 = {'Nslices': 3, 'normalise': True}
T3 = fex.Transformation('splitting', summDict2)
## Tpipe1
tList1 = [('audio_features', T1), ('summ', T2)]
Tpipe1 = fex.TransformationsPipeline(tList1)
## Tpipe2
tList2 = [('audio_features', T1), ('summ', T3)]
Tpipe2 = fex.TransformationsPipeline(tList2)


def test_Transformation_fun():
    
	## feExFun via composition of transformations
    fun1 = T1.fun
    M1 = fun1(wf)
    fun2 = T2.fun
    M2 = fun2(M1)

    ## feExFun via pipeline
    Mpipe = Tpipe1.fun(wf)
    
    np.testing.assert_almost_equal(M2, Mpipe, decimal=6)


def test_wavAnn2annSecs_dataXy_names():
	datO = fex.wavAnn2annSecs_dataXy_names(wavF, annF, featExtFun=Tpipe2.fun)
	assert(datO.m_instances == 19)
    
    
