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
import time

import sys
import numpy as np
from collections import Counter

import pylotwhale.MLwhales.MLtools_beta as myML

from pylotwhale.MLwhales.configs.params_callClf import *
from pylotwhale.MLwhales.experiment_callClf import runCallClfExperiment
from pylotwhale.MLwhales.configs.iter_params_callClf import experimentsControlParams

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
iterParam = controlParamO.parameter
exp_grid_params = controlParamO.controlParams
settingsDi = controlParamO.paramDict
updateTestSet = controlParamO.updateTestSet
expSettingsStr = controlParamO.settingsStr

## more settings
wavColl = fex.readCols(filesDi['train'], (0,1))[:]
call_labels = [l[1] for l in wavColl]
lt = myML.labelTransformer(call_labels)

##### OUTPUT FILES
oDir = os.path.join(filesDi['outDir'], iterParam)

try:
    os.makedirs(oDir)
except OSError:
    pass
out_fN = os.path.join(oDir, "scores.txt")

Tpipe = fex.makeTransformationsPipeline(T_settings)

## clf settings
clfStr = 'cv{}_{}'.format(cv, metric)
settingsStr = "{}-{}".format(Tpipe.string, clfStr)

## write in out file
with open(out_fN, 'a') as out_file: # print details about the dataset into status file
    out_file.write("# call-clf experiment {}\n".format(expSettingsStr))
    out_file.write("###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S")))
    #out_file.write("#{}\n".format(lt.classes_))
    out_file.write("#" + settingsStr+'\n')
    ### dateset info
    out_file.write("# {}\n".format( filesDi['train']))
    out_file.write("# label_transformer {} '{}'\n# data {}\n".format(lt.targetNumNomDict(), 
                                                               "', '".join(lt.classes_), Counter(call_labels)))
    ### train and test sets
    #out_file.write("#TRAIN, shape {}, {}\n".format(np.shape(X_train), Counter(lt.num2nom(y_train))))
    #out_file.write("#TEST, shape {}, {}\n".format(np.shape(X_test), Counter(lt.num2nom(y_test))))


for param in exp_grid_params:
    settingsDi[iterParam] = param
    print(param, T_settings)
    runCallClfExperiment(wavColl, lt, T_settings, out_fN, testFrac=testFrac,
                         cv=cv, pipe_estimators=pipe_estimators, gs_grid=gs_grid, scoring=metric, param=param)

with open(out_fN, 'a') as out_file: # print details about the dataset into status file
    out_file.write("###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S")))
