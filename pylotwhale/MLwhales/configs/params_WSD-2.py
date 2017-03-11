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
lb = 1500; hb = 24000; order = 3 # None
wavPreprocessingFun = None  # functools.partial(sT.butter_bandpass_filter, lowcut=lb, highcut=hb, order=order)
#preproStr +=''#'bandfilter{}_{}'.format(lb, hb)

#### features dictionary
featConstD = {}
summDict = {'summarisation': 'walking', 'n_textWS':5, 'normalise':False}
featConstD['summariseDict'] = summDict
NFFTpow = 10; featConstD["NFFTpow"] = NFFTpow
overlap = 0.5; featConstD["overlap"] = overlap
#Nslices = 4; featConstD["Nslices"] = Nslices
#normalize = True; featConstD["normalize"] = normalize
#featExtract='spectral'; featConstD["featExtrFun"]= featExtract
n_mels = 20; featConstD["n_mels"]= n_mels; featExtract='melspectro'; featConstD["featExtrFun"] = featExtract
#Nceps=2**4; featConstD["Nceps"]= Nceps; featExtract='cepstral'; featConstD["featExtrFun"]= featExtract

##### clf
labs = ['b', 'c', 'echo', 'w']

metric='accuracy'
cv = 10

##### FILES
## INPUT -> collection files
collFi_train = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB_HeikesAnns.txt'
## OUTPUT -> DIR
oDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/experiments/test'





