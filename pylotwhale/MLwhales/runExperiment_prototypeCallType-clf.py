#!/usr/bin/env python

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

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import grid_search

import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.MLwhales.MLtools_beta as myML
import pylotwhale.MLwhales.experimentTools as exT
### load parameters
from pylotwhale.MLwhales.configs.params_prototypeCallType import *
from pylotwhale.MLwhales.configs.iter_params import experimentsControlParams


parser = argparse.ArgumentParser(description='Runs controled experiment.')
parser.add_argument("-cnv", "--controlVariable", type=str,
                    help="name of the experiment controled variable, "
                         "eg: NFFT, NArtificialSamples,"
                         "etc. See iter_params.py.",
                    default="Nslices")

controlVariable = parser.parse_args().controlVariable
print("Control variable: ", controlVariable)

###################  ASSIGNMENTS  ####################

#### Control parameters
controlParamO = experimentsControlParams(controlVariable)

## variables
parameter = controlParamO.parameter
controlParams = controlParamO.controlParams
paramDict = controlParamO.paramDict
updateTestSet = controlParamO.updateTestSet
preproStr = controlParamO.settingsStr
print(parameter)
#sys.exit()
#### parameter grid for the experiments
param_grid = np.repeat(controlParams, n_experiments)  # repeat each experiment n_experiments times

##### LOAD COLLECTIONS
wavAnnColl_tr = fex.readCols(collFi_train, (0, 1))
wavAnnColl_te = fex.readCols(collFi_test, (0, 1))

##### OUTPUT FILES
oDir = os.path.join(oDir, parameter)
try:
    os.makedirs(oDir)
except OSError:
    pass
out_file_scores = os.path.join(oDir, "scores.txt")
out_file_votes = os.path.join(oDir, "votes.txt")

## ensemble
Tpipe = fex.makeTransformationsPipeline(T_settings)
feExFun = Tpipe.fun
featureStr = Tpipe.string
## feature extraction object
#feExOb = fex.wavFeatureExtractionSplit(featConstD)  # feature extraction settings
#featureStr = feExOb.feature_str

#### classes mapping
lt = myML.labelTransformer(callSet)

#### Settings str
ensembleStr = "-".join(['{}{}'.format(ky, vl) for ky, vl in ensembleSettingsD.items()])
preproStr += "-NidExperiments{}".format(n_experiments)
clfStr = 'cv{}'.format(cv)
settingsStr = "{}-{}-{}-{}".format(preproStr, ensembleStr, featureStr, clfStr)

##### CLF
pipe_svc = Pipeline([('clf', svm.SVC(random_state=0))])
gamma_range = [0.01, 0.1, 1.0, 10.0, 100.0]
pen_range = [1.0, 10.0, 100.0]
clf_param_grid = [{'clf__C': pen_range, 'clf__gamma': gamma_range,
                   'clf__kernel': ['rbf']}]
gs_settings = dict(estimator=pipe_svc,
                   param_grid=clf_param_grid,
                   scoring=metric,
                   cv=cv,
                   n_jobs=-1)
gs = grid_search.GridSearchCV(**gs_settings)


######### Functions #######
#ensembleSettings = exT.genrateData_ensembleSettings(param)

feExParamsDict = {'wavAnnColl': wavAnnColl_tr, 'lt': lt,
                  'TpipeSettings': T_settings,
                  'labelSet': callSet,  # depreciated !!!
                  #'wavPreprocessingT' : None,
                  'ensembleSettings': exT.generateData_ensembleSettings(**ensembleSettingsD)
                  }  # , 'ensembleSettings' : ensembleSettings}

###################  TASK  ####################

## print experiment settings header
with open(out_file_scores, 'w') as f:
    f.write("#{}\n#TRAIN: {}\n#TEST: {}\n#{}\n#{}\t{}\n".format(
            time.strftime("%Y.%m.%d\t\t%H:%M:%S"),
            collFi_train, collFi_test, settingsStr, parameter, metric))

print('--------\nSETTINGS\n--------\n:', out_file_scores)  # ,
      #np.shape(X_test), np.shape(y_test),'\n',param_grid, '\n', feExParamDict)

exT.run_iter_clf_experiment(param_grid, parameter, paramDict, gs_settings,
                            feExParamsDict, wavAnnColl_te, lt,
                            updateTestSet=updateTestSet,
                            scores_file=out_file_scores,
                            accum_file=out_file_votes)
