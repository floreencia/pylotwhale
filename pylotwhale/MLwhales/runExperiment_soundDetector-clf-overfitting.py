# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:05:03 2015
#!/usr/bin/python
@author: florencia

Runs call classification experiments generating artificial data and trying
different parameters
"""
from __future__ import print_function
import os
import argparse
import sys
import numpy as np
import time
import pdb

#from sklearn.svc import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.MLwhales.MLtools_beta as myML
import pylotwhale.MLwhales.experimentTools as exT
from pylotwhale.MLwhales import MLEvalTools as MLvl
### load parameters
from pylotwhale.MLwhales.configs.params_whaleSoundDetection import *


###################  ASSIGNMENTS  ####################
##### OUTPUT FILES
try:
    os.makedirs(oDir)
except OSError:
    pass
out_file_scores = os.path.join(oDir, "scores.txt")

#### feature extraction object
feExOb = fex.wavFeatureExtraction(featConstD) # feature extraction settings
feature_str = feExOb.feature_str
feExFun = feExOb.featExtrFun()

#### clf settings
clfStr = 'cv{}-{}'.format(cv, metric)
settingsStr = "{}-{}".format(feature_str, clfStr)

#### write in out file
out_file = open(out_file_scores, 'a')
out_file.write("\n###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S")))
out_file.write("#" + settingsStr)
out_file.close()

#### load collection
WavAnnCollection = fex.readCols(collFi_train, colIndexes = (0,1))
print("\ncollection:", len(WavAnnCollection),
      "\nlast file:", WavAnnCollection[-1])

#### compute features
trainDat = fex.wavAnnCollection2datXy(WavAnnCollection, feExFun) #, wavPreprocesingT=wavPreprocessingFun)
## y_names train and test data
X, y_names = trainDat.filterInstances(labs) # train
lt = myML.labelTransformer(labs)
y = lt.nom2num(y_names)
labsD = lt.targetNumNomDict()
#### train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                  test_size=testFrac, 
                                                  random_state=0)

with open(out_file_scores, 'a') as out_file: # print details to status file
    out_file.write("\n#TRAIN, shape {}\n".format(np.shape(X_train)))
    out_file.write("#TEST, shape {}\n".format(np.shape(X_test))) 

    ## more info
    #out_file.write("#X {}, y {}\n".format( np.shape(X), np.shape(y)))
    out_file.write("#Target dict {}\t{}\n".format(labsD, trainDat.targetFrequencies()))


### grid search
pipe_svc = Pipeline(estimators)
param_grid = [ {'reduce_dim__n_components' : pca_range,
                'clf' : [SVC()],
                'clf__C': pen_range, 
                'clf__gamma': gamma_range, 
                'clf__kernel': ['rbf'] }]

gs = GridSearchCV(estimator=pipe_svc,
                              param_grid=param_grid,
                              scoring=metric,
                              cv=cv,
                              n_jobs=-1)

gs = gs.fit(X_train, y_train)

with open(out_file_scores, 'a') as out_file:
    out_file.write(MLvl.gridSearchresults(gs))


print(out_file_scores)

