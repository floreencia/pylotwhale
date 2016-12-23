#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:05:03 2015
@author: florencia
"""
#####
from __future__ import print_function, division
import numpy as np
import pylotwhale.MLwhales.experimentTools as exT
#### !!! the parameters file should be the same one used in the script
from pylotwhale.MLwhales.configs.params_WSD1 import summDict, auD, filtDi

### Control parameter mapping to settings object
def experimentsControlParams(iterParam):
    """
    Sets up the variables for a parameter controled experiment
    given the control parameter name (str), returns an object for
    setting up the experiment variables
    """
    controlParamsDict = {"n_textWS": n_textWSO,
                         "n_textWS_b": n_textWSO_b,
                         "overlap": overlapO,
                         "Nceps": NcepsO,
                         "NFFT": NFFTO,
                         "n_mels" : NmelsO,
                         "n_mels_b" : NmelsO_b,
                         "fmin": fminO
                         }
    assert iterParam in controlParamsDict.keys(), '{} is not a valid control parameter\nValid: {}'.format(iterParam, ', '.join(controlParamsDict.keys()))
    return controlParamsDict[iterParam]



###### Control Parameter Objects

#### n_textWS
parameter="n_textWS"
expSettings=[1, 24, 1]
controlParam = np.arange(*expSettings)
paramDict = summDict
expStr = "{}_{}".format( parameter, '_'.join([str(item) for item in expSettings]))

n_textWSO = exT.controlVariable(parameterName=parameter,
                                controlParams=controlParam,
                                updateTestSet=True,
                                paramDict=paramDict,
                                settingsStr=expStr)

#### n_textWS broad
parameter="n_textWS"
expSettings=[1, 80, 5]
controlParam = np.arange(*expSettings)
paramDict = summDict
expStr = "{}_{}".format( parameter, '_'.join([str(item) for item in expSettings]))

n_textWSO_b = exT.controlVariable(parameterName=parameter,
                                controlParams=controlParam,
                                updateTestSet=True,
                                paramDict=paramDict,
                                settingsStr=expStr)

#### overlap

parameter = 'overlap'
expSettings=[0, 1, 0.1]
paramDict=auD
controlParams = np.arange(*expSettings)
expStr = "{}_{}".format( parameter, '_'.join([str(item) for item in expSettings]))

overlapO = exT.controlVariable(parameterName=parameter,
                               controlParams=controlParams,
                               updateTestSet=True,
                               paramDict=paramDict,
                               #updateParamInDict=updateParamInDict_over,                           
                               settingsStr=expStr
                           )

##### Nceps
parameter = 'Nceps'
paramDict = auD
expSettings=[4, 30, 3]
controlParams = np.arange(*expSettings)
expStr = "{}_{}".format( parameter, '_'.join([str(item) for item in expSettings]))

NcepsO = exT.controlVariable(parameterName=parameter,
                             controlParams=controlParams,
                             updateTestSet=True,
                             paramDict=paramDict,
                             #updateParamInDict=updateParamInDict_NFFT,
                             settingsStr=expStr
                    )


##### NFFT
parameter = 'NFFT'
expSettings=[8, 11, 1]
NFFTpows = np.arange(*expSettings)  # np.linspace(a0, a, n_amps)
NFFTs = np.array([2**p for p in NFFTpows])
paramDict = auD
expStr = "{}_{}".format( parameter, '_'.join([str(item) for item in expSettings]))

NFFTO = exT.controlVariable(parameterName=parameter,
                        controlParams=NFFTs,
                        updateTestSet=True,
                        paramDict=paramDict,
                        #updateParamInDict=updateParamInDict_NFFT,
                        settingsStr=expStr
                        )

##### fmin
parameter = 'fmin'
expSettings=[0, 5000, 200]
params = np.arange(*expSettings)  # np.linspace(a0, a, n_amps) 
expStr = "{}_{}".format( parameter, '_'.join([str(item) for item in expSettings]))
paramDict = auD

fminO = exT.controlVariable(parameterName=parameter,
                             controlParams=params,
                             updateTestSet=True,
                             paramDict=paramDict,
                             #updateParamInDict=updateParamInDict_NFFT,
                             settingsStr=expStr
                         )

##### N mels
parameter = 'n_mels'
expSettings=[1, 24, 1]
Nmels = np.arange(*expSettings)  # np.linspace(a0, a, n_amps) 
expStr = "{}_{}".format( parameter, '_'.join([str(item) for item in expSettings]))
paramDict = auD

NmelsO = exT.controlVariable(parameterName=parameter,
                             controlParams=Nmels,
                             updateTestSet=True,
                             paramDict=paramDict,
                             #updateParamInDict=updateParamInDict_NFFT,
                             settingsStr=expStr
                         )

parameter = 'n_mels'
expSettings=[1, 40, 2]
Nmels = np.arange(*expSettings)  # np.linspace(a0, a, n_amps) 
expStr = "{}_{}".format( parameter, '_'.join([str(item) for item in expSettings]))
paramDict = auD

NmelsO_b = exT.controlVariable(parameterName=parameter,
                             controlParams=Nmels,
                             updateTestSet=True,
                             paramDict=paramDict,
                             #updateParamInDict=updateParamInDict_NFFT,
                             settingsStr=expStr
                         )
                         
                         
##### N mels
#parameter = 'Nceps'
#NcepsO = NmelsO
#NcepsO.NmelsO.paramKey


