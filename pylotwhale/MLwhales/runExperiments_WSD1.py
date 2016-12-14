# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 18:05:03 20156
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

import pylotwhale.MLwhales.featureExtraction as fex
from pylotwhale.MLwhales.configs.params_WSD1 import *
from pylotwhale.MLwhales.experiment_WSD1 import runExperiment
from pylotwhale.MLwhales.configs.iter_params_WSD import experimentsControlParams
from pylotwhale.MLwhales import MLEvalTools as MLvl
import pylotwhale.MLwhales.MLtools_beta as myML

parser = argparse.ArgumentParser(description='Runs controled experiment.')
parser.add_argument("-cnv", "--controlVariable", type=str,
                    help="name of the experiment controled variable, "
                         "eg: NFFT, NArtificialSamples,"
                         "etc. See iter_params.py.",
                    default="NFFT")

#parser.add_argument("-vals", '--paramVals',# nargs='+',help='values of the experiment')
args = parser.parse_args()

#### Control parameters
controlVariable = args.controlVariable
print("Control variable: ", controlVariable)
controlParamO = experimentsControlParams(controlVariable)

## variables
parameter = controlParamO.parameter
controlParams = controlParamO.controlParams
paramDict = controlParamO.paramDict
updateTestSet = controlParamO.updateTestSet
expSettingsStr = controlParamO.settingsStr

## more settings
train_coll = fex.readCols(collFi_train, colIndexes = (0,1))
test_coll = np.genfromtxt(collFi_test, dtype=object)
lt = myML.labelTransformer(clf_labs)
metricSetup = {'scorer_name': metric} #, 'classTag':lt.nom2num('c')}
optimiseCallsScorer = MLvl.get_scorer(**metricSetup) #MLvl.getCallScorer(classTag='c', lt=lt) #'accuracy'#

##### OUTPUT FILES
oDir = os.path.join(oDir, parameter)
try:
    os.makedirs(oDir)
except OSError:
    pass

Tpipe = fex.makeTransformationsPipeline(T_settings)
#out_fN = os.path.join(oDir, "scores.txt")
out_fN = os.path.join(oDir, "scores-{}.txt".format(expSettingsStr))
print(out_fN)

## clf settings
clfStr = 'cv{}-{}'.format(cv, metric)
settingsStr = "{}-{}".format(Tpipe.string, clfStr)
settingsStr += '-labsHierarchy_' + '_'.join(labsHierarchy)

## write in out file
out_file = open(out_fN, 'a')
out_file.write("# WSD1 experiment {}\n".format(expSettingsStr))
out_file.write("###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S")))
out_file.write("#" + settingsStr+'\n')
out_file.close()


for param in controlParams:
    ## update settings
    paramDict[parameter] = param
    print(param, T_settings)
    runExperiment(train_coll=train_coll, test_coll=test_coll, lt=lt,
                  T_settings=T_settings, labsHierarchy=labsHierarchy, out_fN=out_fN,
                  cv=cv, pipe_estimators=pipe_estimators, gs_grid=gs_grid,
                  scoring=optimiseCallsScorer,
                  param=param, predictionsDir=predictionsDir)