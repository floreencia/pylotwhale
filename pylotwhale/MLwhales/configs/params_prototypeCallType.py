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
fs = 48000
T_settings = []

## preprocessing
filt='band_pass_filter'
filtDi={"fs":fs, "lowcut":0, "highcut":22000, "order":5}
#T_settings.append(('bandFilter', (filt, filtDi)))

#### features dictionary
auD = {}
auD["sRate"] = fs
NFFTpow = 7; auD["NFFT"] = 2**NFFTpow
overlap = 0.5; auD["overlap"] = overlap
#Nslices = 4; auD["Nslices"] = Nslices
audioF='spectral'#; auD["featExtrFun"]= featExtract
#n_mels = 64; auD["n_mels"]= n_mels; audioF='melspectro'; 
#Nceps=2**4; auD["Nceps"]= Nceps; audioF='cepstral'
T_settings.append(('Audio_features', (audioF, auD)))

summDict = {'Nslices': 5, 'normalise': True}
summType = 'splitting'
T_settings.append(('summ', (summType, summDict)))

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





