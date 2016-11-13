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
n_mels = 20; auD["n_mels"]= n_mels;
audioF = 'melspectro'
#T1 = fex.Transformation(featExtract, auD)

#### summ features
summDict = {'n_textWS': 5, 'normalise': True}
summType = 'walking'

##### clf
testFrac = 0.2
labs = ['b', 'c', 'echo', 'w']

metric='accuracy'
cv = 10

### parameters
pca_range = [ 6, 8, 10, 12, None]
gamma_range = [ 0.1, 1.0]
pen_range = [ 1.0, 10.0, 100.0]

from sklearn.decomposition import PCA
from sklearn.svm import SVC

estimators = [('reduce_dim', PCA()), ('clf', SVC())]


##### FILES
## INPUT -> collection files
collFi_train = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB.txt'
## OUTPUT -> DIR
oDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/experiments/test'





