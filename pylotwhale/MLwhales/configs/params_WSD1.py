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
T_settings = []
## preprocessing

#### audio features
auD = {}
auD["sRate"] = 48000
NFFTpow = 10; auD["NFFT"] = 2**NFFTpow
overlap = 0; auD["overlap"] = overlap
n_mels = 10; auD["n_mels"]= n_mels;
fmin = 1000; auD["fmin"]= fmin;
audioF = 'melspectro'
T_settings.append(('Audio_features', (audioF, auD)))

#### summ features
summDict = {'n_textWS': 4, 'normalise': True}
summType = 'walking'
T_settings.append(('summ', (summType, summDict)))

##### clf
testFrac = 0.2
clf_labs = ['b', 'c', 'w']
labsHierarchy = ['c', 'w']

metric='accuracy'
cv = 5
from pylotwhale.MLwhales.clf_pool import random_forest_clf as clfSettings

###
pca_range = [ 6, 8, 10, 12, None]

estimators = [#('reduce_dim', PCA()),
              #('clf', SVC()),
              ('clf',  clfSettings.fun)]

paramsDi={'reduce_dim__n_components' : pca_range}
paramsDi.update(clfSettings.grid_params_di)
param_grid = [paramsDi] # clfSettings.grid_params #


##### FILES
## INPUT -> collection files
collFi_train = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB.txt'
collFi_test = '/home/florencia/whales/data/mySamples/whales/tapes/NPW/B/collections/wavAnnColl_grB_fullTapes.txt'
## OUTPUT -> DIR
oDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/experiments/test'
