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
fs = 48000
T_settings = []

## preprocessing
filt='band_pass_filter'
filtDi={"fs":fs, "lowcut":1100, "highcut":22000, "order":5}
T_settings.append(('bandFilter', (filt, filtDi)))

#### audio features
auD = {}
auD["sRate"] = fs
NFFTpow = 9; auD["NFFT"] = 2**NFFTpow
overlap = 0; auD["overlap"] = overlap
n_mels = 20; auD["n_mels"]= n_mels;
fmin = 1100; auD["fmin"]= fmin;
audioF = 'melspectro'
T_settings.append(('Audio_features', (audioF, auD)))

#### summ features
summDict = {'n_textWS': 10, 'normalise': True}
summType = 'walking'
T_settings.append(('summ', (summType, summDict)))

##### clf
testFrac = 0.2
clf_labs = ['b', 'c', 'w']
labsHierarchy = ['c', 'w']

metric='accuracy'
cv = 5

###
paramsDi={}
estimators=[]

##PCA
from sklearn.decomposition import PCA
pca_range = [ 6, 8, 10, 12]
#paramsDi.update{'reduce_dim__n_components' : pca_range}
#estimators.append(('reduce_dim',  PCA()))

### CLF
from pylotwhale.MLwhales.clf_pool import svc_rbf as clfSettings
estimators.append(('clf',  clfSettings.fun))
paramsDi.update(clfSettings.grid_params_di)

param_grid = [paramsDi] # clfSettings.grid_params #

##### FILES
## INPUT -> collection files
collFi_train = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB.txt'
collFi_test = '/home/florencia/whales/data/mySamples/whales/tapes/NPW/B/collections/wavAnnColl_grB_fullTapes.txt'
## OUTPUT -> DIR
oDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/test/experiments'
savePredictions = True
