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

n_protoptypes=3
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

prepro='maxabs_scale'
preproDict = {}
T_settings.append(('normaliseWF', (prepro, preproDict)))

#### features dictionary
auD = {}
auD["sRate"] = fs
NFFTpow = 8; auD["NFFT"] = 2**NFFTpow
overlap = 0.5; auD["overlap"] = overlap
#Nslices = 4; auD["Nslices"] = Nslices
#audioF='spectral'#; auD["featExtrFun"]= featExtract
#n_mels = 64; auD["n_mels"]= n_mels; audioF='melspectro'; 
Nceps=2**4; auD["Nceps"]= Nceps; audioF='cepstral'
T_settings.append(('Audio_features', (audioF, auD)))

summDict = {'Nslices': 2, 'normalise': True}
summType = 'splitting'
T_settings.append(('summ', (summType, summDict)))

##### clf
metric='accuracy'
cv = 6

#### Classes                
callSet = ['126i', '130', '127', '129', '128i', '131i', '093ii']

##### FILES
## INPUT -> collection files
filesDi = {}
#collFi_train 
filesDi['train']= '/home/florencia/whales/data/Vocal-repertoire-catalogue-Pilot-whales-Norway/flo/annotations/callClassifier/collections/wavAnnColl_calltypes-{}SamplesTrainCollection.txt'.format(n_protoptypes)
#collFi_test 
filesDi['test'] = '/home/florencia/whales/data/Vocal-repertoire-catalogue-Pilot-whales-Norway/flo/annotations/callClassifier/collections/wavAnnColl_calltypes-6SamplesTestCollection.txt'
## OUTPUT -> DIR
#oDir = 
filesDi['outDir'] = '/home/florencia/whales/MLwhales/callClassification/data/experiments/{}prototypes/'.format(n_protoptypes)





