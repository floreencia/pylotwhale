#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:05:03 2015
#!/usr/bin/python
@author: florencia

Runs call classification experiments generating artificial data and trying 
different parameters
"""

from __future__ import print_function, division
import numpy as np
import os
import sys

import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.MLwhales.experimentTools as exT
import pylotwhale.MLwhales.MLtools_beta as myML

from sklearn.utils import shuffle
from sklearn import grid_search
from sklearn.pipeline import Pipeline
from sklearn import svm
import time

#######################   SETTINGS   ######################################

####### Iter parameters

parameter = 'Nceps'
paramKey = 'Nceps'
N0 = 12
Ndelta = 1
N = 30
amp = np.arange(N0, N, Ndelta) # np.linspace(a0, a, n_amps) 

def updateParamInDict(paramDict, paramKey, param):
    paramDict['featExtFun'][paramKey] = param 
    return paramDict

updateTestSet = exT.updateParamTestSet

preproStr="{}_{}".format(parameter, '{}-{}'.format(N0,N))


##############################
####### FIX SETTINGS  ########
## experiment repetitions
# when random numbers are involved, repeat the experiment to get the stats
n_experiments = 10 # identical experiment repetitions
param_grid = np.repeat(amp, n_experiments) # repeat experiment

#n_artificial_samples = 10 # number of artificial samples to generate for each amp

#### Feature extraction 
## preprocessing
lb = 1500; hb = 24000; order = 3 # None
wavPreprocessingFun = None#functools.partial(sT.butter_bandpass_filter, lowcut=lb, highcut=hb, order=order)
#preproStr +=''#'bandfilter{}_{}'.format(lb, hb)

## features dictionary
featConstD = {}
NFFTpow = 10; featConstD["NFFTpow"] = NFFTpow
overlap = 0.5; featConstD["overlap"]= overlap
Nslices = 8; featConstD["Nslices"]= Nslices
normalize = True; featConstD["normalize"]= normalize
#featExtract='spectral'; featConstD["featExtrFun"]= featExtract
#n_mels = 64; featConstD["n_mels"]= n_mels; featExtract='melspectro'; featConstD["featExtrFun"]= featExtract
Nceps=20; featConstD["Nceps"]= Nceps; featExtract='cepstral'; featConstD["featExtrFun"]= featExtract
## feature extraction object
feExOb = fex.wavFeatureExtractionSplit(featConstD) # feature extraction settings
feature_str = feExOb.feature_str
#feExFun=feExOb.featExtrFun()

##### clf
metric='accuracy'
cv = 10
clfStr = 'cv{}'.format(cv)
pipe_svc = Pipeline([('clf', svm.SVC(random_state=0) )])
gamma_range = [ 0.01, 0.1, 1.0, 10.0, 100.0]
pen_range = [ 1.0, 10.0, 100.0]
clf_param_grid = [ {'clf__C': pen_range, 'clf__gamma': gamma_range, 
                    'clf__kernel': ['rbf']}]
gs_settings = dict(estimator=pipe_svc,
                  param_grid=clf_param_grid,
                  scoring=metric,
                  cv=cv,
                  n_jobs=-1)
gs = grid_search.GridSearchCV(**gs_settings)
                  
#### Classes                
callSet=['126i', '130', '127', '129', '128i', '131i', '093ii']
lt = myML.labelTransformer(callSet)

##### Out files
oDir = os.path.join('/home/florencia/whales/MLwhales/callClassification/data/experiments', parameter)
try:
    os.makedirs(oDir)
except OSError:
    pass
out_file_scores = os.path.join(oDir, "scores.txt")
out_file_votes = os.path.join(oDir, "votes.txt")

##### Data
## collection files
collFi_train = '/home/florencia/whales/data/Vocal-repertoire-catalogue-Pilot-whales-Norway/flo/wavs/wavFiles-wavAnnCollection-prototypes.txt'
collFi_test = '/home/florencia/whales/MLwhales/callClassification/data/collections/grB-balanced14collection.txt'
## collections
wavAnnColl_tr = fex.readCols(collFi_train, (0,1))
wavAnnColl_te = fex.readCols(collFi_test, (0,1))

#### Settings strings
preproStr+="-NidExperiments{}".format(n_experiments)
settingsStr = "{}-{}-{}".format(preproStr, feature_str, clfStr )

######### Functions #######
#ensembleSettings = exT.genrateData_ensembleSettings(param)

feExParamDict = {'wavAnnColl' : wavAnnColl_tr, 'lt' : lt,
                 'featExtFun' : featConstD, 
                 'labelSet' : callSet, 
                 #'wavPreprocessingT' : None,
                 'ensembleSettings' : exT.genrateData_ensembleSettings()
                 }#, 'ensembleSettings' : ensembleSettings}

###################  TASK  ####################

## print experiment settings header
with open(out_file_scores, 'w') as f:
    f.write("#{}\n#TRAIN: {}\n#TEST: {}\n#{}\n#{}\t{}\n".format(
            time.strftime("%Y.%m.%d\t\t%H:%M:%S"), 
            collFi_train, collFi_test, settingsStr, parameter, metric))

print('--------\nSETTINGS\n--------\n:', out_file_scores)#,
      #np.shape(X_test), np.shape(y_test),'\n',param_grid, '\n', feExParamDict)
      
exT.run_iter_clf_experiment(param_grid, gs_settings, feExParamDict, 
                            paramKey, updateParamInDict,           
                            wavAnnColl_te, lt,
                            updateTestSet = updateTestSet,
                            scores_file = out_file_scores, 
                            accum_file = out_file_votes)


   