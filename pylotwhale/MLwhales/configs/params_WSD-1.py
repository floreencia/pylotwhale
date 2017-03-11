#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
SETTINGS
========

Parameters to be used in classification experiment:

runExperiment_prototypeCallType-clf.py

@author: florencia
"""

##############################
########   SETTINGS   ########
##############################


#### Experiment settings
# when random numbers are involved, repeat the experiment to get the stats
n_experiments = 3  # identical experiment repetitions

#### Feature extraction 
## preprocessing
prepro='maxabs_scale'
preproDict = {}

#### audio features
auD = {}
auD["sRate"] = 48000
NFFTpow = 10; auD["NFFT"] = 2**NFFTpow
overlap = 0.5; auD["overlap"] = overlap
audioF = 'spectral'#; featConstD["featExtrFun"]= featExtract
#T1 = fex.Transformation(featExtract, auD)

#### summ features
summDict = {'n_textWS': 3, 'normalise': False}
summType = 'walking'

##### clf
labs = ['b', 'c', 'echo', 'w']

metric='accuracy'
cv = 10

##### FILES
## INPUT -> collection files
collFi_train = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB_HeikesAnns.txt'
## OUTPUT -> DIR
oDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/experiments/test'





