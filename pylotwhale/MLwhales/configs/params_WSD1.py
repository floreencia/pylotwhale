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
overlap = 0.5; auD["overlap"] = overlap
n_mels = 20; auD["n_mels"]= n_mels;
#fmin = 1000; auD["fmin"]= fmin;
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

### parameters
#pca_range = [ 6, 8, 10, 12, None]
gamma_range = [ 0.1, 1.0]
pen_range = [ 1.0, 10.0, 100.0]

from sklearn.decomposition import PCA
from sklearn.svm import SVC

estimators = [#('reduce_dim', PCA()),
              ('clf', SVC())]
param_grid = [ {#'reduce_dim__n_components' : pca_range,
                'clf' : [SVC()],
                'clf__C': pen_range, 
                'clf__gamma': gamma_range, 
                'clf__kernel': ['rbf'] }]

##### FILES
## INPUT -> collection files
collFi_train = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB.txt'
## OUTPUT -> DIR
oDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/experiments/test'





