# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 18:05:03 20156
#!/usr/bin/python
@author: florencia

Runs call classification experiments generating artificial data and trying
different parameters
"""

from __future__ import print_function, division
import os
#import argparse
import sys
import numpy as np
import time
from sklearn.model_selection import ParameterGrid


import pylotwhale.MLwhales.MLtools_beta as myML
import pylotwhale.MLwhales.MLEvalTools as MLvl

import pylotwhale.MLwhales.experiment_WSD1 as wsd
from pylotwhale.MLwhales.configs.params_WSD1 import *

collFi_train = '/home/flo/x1-flo/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB-tapes-desktop.txt'
#'/home/flo/x1-flo/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB-tapes-desktop.txt'
#'/home/flo/x1-flo/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB-desktop.txt'
#collFi_test = #'/home/flo/x1-flo/whales/data/mySamples/whales/tapes/NPW/B/collections/wavAnnColl_grB_fullTapes.txt'
train_coll = fex.readCols(collFi_train, colIndexes = (0,1))
test_coll = None#np.genfromtxt(collFi_test, dtype=object)
oDir = '/home/flo/x1-flo/whales/MLwhales/whaleSoundDetector/data/experiments/'

lt = myML.labelTransformer(clf_labs)


#### Feature extaction settings
T_settings=[]

## preprocessing-
prepro='maxabs_scale'
preproDict = {}
T_settings.append(('normaliseWF', (prepro, preproDict)))

## audio features
auD = {}
auD["fs"] = fs
NFFTpow = 8; auD["NFFT"] = 2**NFFTpow
overlap = 0; auD["overlap"] = overlap
#audioF='spectral'#; auD["featExtrFun"]= featExtract
n_mels = 128/4; auD["n_mels"]= n_mels; audioF='melspectro'; 
#Nceps=2**4; auD["Nceps"]= Nceps; audioF='MFCC'
T_settings.append(('Audio_features', (audioF, auD)))

## sumarisation
summDict = {'n_textWS': 20, 'normalise': True}
summType = 'walking'
T_settings.append(('summ', (summType, summDict)))

Tpipe = fex.makeTransformationsPipeline(T_settings)


##### clf settings
testFrac = 0.2
clf_labs = ['b', 'c', 'w']
labsHierarchy = ['c', 'w']

## metric
metric = 'f1c'
#metricSetup = {'scorer_name': metric, 'classTag':lt.nom2num('c')}
#optimiseCallsScorer = MLvl.get_scorer(metric) #MLvl.getCallScorer(classTag='c', lt=lt) #'accuracy'#

cv = 5

## inicialise Clf settings
paramsDi={}
pipe_estimators=[]

## CLF
from pylotwhale.MLwhales.clf_pool import svc_l as clf_settings
pipe_estimators.append(('clf',  clf_settings.fun))
paramsDi.update(clf_settings.grid_params_di)
gs_grid = [paramsDi] # clfSettings.grid_params #


#### output file
oDir = os.path.join(oDir, audioF)
try:
    os.makedirs(oDir)
except OSError:
    pass
out_fN = os.path.join(oDir, "scores.txt")

#### experiment

exp = wsd.WSD_experiment(train_coll, test_coll, lt,
                         labsHierarchy=labsHierarchy, out_file=out_fN,
                         cv=cv, clf_pipe=pipe_estimators, clf_grid=gs_grid, 
                         metric=metric)

#### experiment grid
param_grid = {'NFFT': [2**7, 2**8, 2**9, 2**10], 
              'n_mels' : np.hstack((np.arange(1, 10, 1), np.arange(10, 140, 10))),
		#'n_mels' : np.arange(1, 10, 1),
             # 'n_mels' : [2**8, 2**9, 2**10], # ceps
              #'Nceps' :  np.arange(1, 40, 2),
              'n_textWS': np.arange(1, 50, 5)}

grid = ParameterGrid(param_grid)
expSettingsStr = "{} {}".format(len(grid), str(param_grid))

#### output file
s = "\n# {}".format(clf_settings.clf_name)
s += "\n#" + Tpipe.string
#s += "\n#{}\n".format("\n#".join(["{}:{}".format([k, ": {}".format(", ".join[str(x) for x in v]) for k, v in param_grid.items()]))
s += "\n# {}\n".format("\n# ".join(["{}: {}".format(k, ", ".join([str(x) for x in v])) for k, v in param_grid.items()]))
exp.print_comments(end=s)
exp.print_experiment_header()
#sys.exit()

for p in grid:
        
    #if p['n_mels'] > p['NFFT']/4: # too many n_mels for the NFFT
    #    continue
    
    #if p['Nceps'] >= p['NFFT']/2: # too many Nceps for the NFFT
    #    continue
        
    Tpipe.steps['Audio_features'].settingsDict['NFFT'] = p['NFFT']
    Tpipe.steps['Audio_features'].settingsDict['n_mels'] = p['n_mels']
    #Tpipe.steps['Audio_features'].settingsDict['Nceps'] = p['Nceps']
    Tpipe.steps['summ'].settingsDict['n_textWS'] = p['n_textWS']
    print(p)
    
    try:
        exp.run_experiment(Tpipe, class_balance='c')
        
    except ValueError as e:
        print(e, '\nError!', p)


exp.print_in_out_file("\n#" + exp.time)
print("OUTPUT:", out_fN)
