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
n_experiments = 10  # identical experiment repetitions

#### Ensemble generation
ensembleSettingsD = {}
ensembleSettingsD['n_artificial_samples'] = 6 # number of artificial samples to generate for each amp
ensembleSettingsD['whiteNoiseAmp'] = 0.09

#### Feature extraction 
## preprocessing
lb = 1500; hb = 24000; order = 3 # None
wavPreprocessingFun = None  # functools.partial(sT.butter_bandpass_filter, lowcut=lb, highcut=hb, order=order)
#preproStr +=''#'bandfilter{}_{}'.format(lb, hb)

#### features dictionary
featConstD = {}
summDict = {'summarisation': 'splitting', 'Nslices':5, 'normalise':True}
featConstD['summariseDict'] = summDict
NFFTpow = 7; featConstD["NFFTpow"] = NFFTpow
overlap = 0.5; featConstD["overlap"] = overlap
#Nslices = 4; featConstD["Nslices"] = Nslices
#normalize = True; featConstD["normalize"] = normalize
featExtract='spectral'; featConstD["featExtrFun"]= featExtract
#n_mels = 64; featConstD["n_mels"]= n_mels; featExtract='melspectro'; featConstD["featExtrFun"]= featExtract
#Nceps=2**4; featConstD["Nceps"]= Nceps; featExtract='cepstral'; featConstD["featExtrFun"]= featExtract

##### clf
metric='accuracy'
cv = 6

#### Classes                
callSet = ['126i', '130', '127', '129', '128i', '131i', '093ii']

##### FILES
## INPUT -> collection files
collFi_train = '/home/florencia/whales/data/Vocal-repertoire-catalogue-Pilot-whales-Norway/flo/wavs/wavFiles-wavAnnCollection-prototypes.txt'
collFi_test = '/home/florencia/whales/MLwhales/callClassification/data/collections/grB-balanced-14-13-Filecollection.txt'
## OUTPUT -> DIR
oDir = '/home/florencia/whales/MLwhales/callClassification/data/experiments/trashtest/'





