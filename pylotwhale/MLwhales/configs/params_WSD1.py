#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
SETTINGS
========

Parameters to be used in classification WSD1 experiment

@author: florencia
"""

##############################
########   SETTINGS   ########
##############################

###### Experiment settings ######

#### Feature extraction 
fs = 48000
T_settings = []

## preprocessing
filt='band_pass_filter'
filtDi={"fs":fs, "lowcut":0, "highcut":22000, "order":5}
#T_settings.append(('bandFilter', (filt, filtDi)))

## audio features
auD = {}
auD["fs"] = fs
NFFTpow = 9; auD["NFFT"] = 2**NFFTpow
overlap = 0; auD["overlap"] = overlap
n_mels = 7; auD["n_mels"] = n_mels;
#fmin = 200; auD["fmin"]= fmin;
audioF = 'melspectro'
T_settings.append(('Audio_features', (audioF, auD)))

## summ features
summDict = {'n_textWS': 20, 'normalise': True}
summType = 'walking'
T_settings.append(('summ', (summType, summDict)))

##### clf
testFrac = 0.2
clf_labs = ['b', 'c', 'w']
labsHierarchy = ['c', 'w']

metric = 'f1c'
metricSettingsDi = {} # {'classTag':1}
cv = 5

### inicialise Clf settings
paramsDi={}
pipe_estimators=[]

### PCA
from sklearn.decomposition import PCA
pca_range = [ 6, 8, 10, 12]
#paramsDi.update{'reduce_dim__n_components' : pca_range}
#estimators.append(('reduce_dim',  PCA()))

### CLF
from pylotwhale.MLwhales.clf_pool import svc_l as clf_settings
pipe_estimators.append(('clf',  clf_settings.fun))
paramsDi.update(clf_settings.grid_params_di)
gs_grid = [paramsDi] # clfSettings.grid_params #

##### FILES
## INPUT -> collection files
collFi_train = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB.txt'
collFi_test = '/home/florencia/whales/data/mySamples/whales/tapes/NPW/B/collections/wavAnnColl_grB_fullTapes.txt'
## OUTPUT -> DIR
import pylotwhale.MLwhales.featureExtraction as fex
settings_str = fex.makeTransformationsPipeline(T_settings).string + clf_settings.clf_name
oDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/experiments/'\
		'WSD1/linearModelSVC/clf_{}/{}/{}/{}/'.format(''.join(clf_labs), audioF, metric, settings_str)#, auD["NFFT"])
oDir = '~/Desktop/TEST'
predictionsDir=False 
