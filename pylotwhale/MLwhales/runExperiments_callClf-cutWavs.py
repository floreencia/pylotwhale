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

import pylotwhale.MLwhales.MLtools_beta as myML

from pylotwhale.MLwhales.configs.params_CallTypeClf import *
from pylotwhale.MLwhales.experiment_callClf import runCallClfExperiment
from pylotwhale.MLwhales.configs.iter_params_callClf import experimentsControlParams

"""
import pylotwhale.MLwhales.featureExtraction as fex
from pylotwhale.MLwhales.configs.params_WSD1 import *
from pylotwhale.MLwhales.experiment_WSD1 import runExperiment
from pylotwhale.MLwhales.configs.iter_params_WSD import experimentsControlParams
from pylotwhale.MLwhales import MLEvalTools as MLvl
import pylotwhale.MLwhales.MLtools_beta as myML
"""

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
fex.readCols(filesDi['train'], (0,1))[:100]
wavColl = fex.readCols(filesDi['train'], (0,1))[:100]
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
out_file = open(out_fN, 'a')
out_file.write("# call-clf experiment {}\n".format(expSettingsStr))
out_file.write("###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S")))
out_file.write("#" + settingsStr+'\n')
out_file.close()

for param in exp_grid_params:
    settingsDi[iterParam] = param
    print(param, T_settings)
    runCallClfExperiment(wavColl, lt, T_settings, out_fN, 
                         cv, pipe_estimators, gs_grid, metric, param=param)

