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

### Control parameter mapping to settings object
def experimentsControlParams(iterParam):
    """
    Sets up the variables for a parameter controled experiment
    given the control parameter name (str), returns an object for
    setting up the experiment variables
    """
    controlParamsDict = {"Nslices": NslicesO,
                         "overlap": overlapO,
                         "NArtificialSamples": ArtificialSamplesO,
                         "noiseAmplitude": noiseAmplitudeO,
                         "Nceps": NcepsO,
                         "NFFT": NFFTO,
                         "Nmels" : NmelsO
                         }
    assert iterParam in controlParamsDict.keys(), '{} is not a valid'
    'control parameter\nValid: {}'.format(iterParam, ', '.join(controlParamsDict.keys()))
    return controlParamsDict[iterParam]


###### Iter parameters

class controlVariable():
    '''
    Data structure for an experiment's control variable
    Parameters:
    -----------
    paramater : str
    paramKey : str
        paramKey
    controlParams: numpy array
        experiment's control parameter
    updateTestSet : bool
    updateParamInDict : callable
    '''
    def __init__(self, parameterName, paramKey,
                 controlParams, updateTestSet,
                 updateParamInDict, settingsStr):
        self.parameter = parameterName
        self.paramKey = paramKey
        self.controlParams = controlParams
        self.updateTestSet = updateTestSet
        self.updateParamInDict = updateParamInDict
        self.settingsStr = settingsStr


####### Control Parameter Objects

#### Nslices
parameter="Nslices"
Nsl0 = 1
Nsl = 12
controlParam = np.arange(Nsl0, Nsl)

def updateParamInDict_Nslices(paramDict, paramKey, param):
    paramDict['featExtFun']['summariseDict'][paramKey] = param
    print('\n\nTEST\n\n', paramDict, "\n\n", paramKey)
    return paramDict

NslicesO = controlVariable(parameterName=parameter,
                           paramKey=parameter,
                           controlParams=controlParam,
                           updateTestSet=True,
                           updateParamInDict=updateParamInDict_Nslices,
                           settingsStr="{}_{}_{}".format(parameter, Nsl0, Nsl))


#### NArtificialSamples = ensembleSettings[n_artificial_samples]
parameter = "NArtificialSamples"
N0 = 4
N = 10
controlParam_NArtSamps = np.arange(N0, N)  # less than 5 doesn't work with CV=10


def updateParamInDict_NArtSamps(paramDict, paramKey, param):
    paramDict[paramKey] = exT.generateData_ensembleSettings(
                                                n_artificial_samples=param)
    return paramDict

ArtificialSamplesO = controlVariable(parameterName=parameter,
                                     paramKey="ensembleSettings",
                                     controlParams = controlParam_NArtSamps,
                                     updateTestSet=False,
                                     updateParamInDict=updateParamInDict_NArtSamps,
                                     settingsStr="{}_{}_{}".format(parameter, N0, N))


#### overlap

parameter = 'overlap'
N0 = 0
Ndelta = 0.1
N = 1

def updateParamInDict_over(paramDict, paramKey, param):
    paramDict['featExtFun'][paramKey] = param
    return paramDict

overlapO = controlVariable(parameterName=parameter,
                           paramKey=parameter,
                           controlParams=np.arange(N0, N, Ndelta),
                           updateTestSet=True,
                           updateParamInDict=updateParamInDict_over,
                           settingsStr="{}_{}".format(
                                       parameter, '{}_{}'.format(N0, N))
                           )

#### noiseAmplitude
parameter = 'noiseAmplitude'
# noise amplitude
n_amps = 10
a0 = 0.001
a = 0.2 # 0.005 for ceps
amps = np.linspace(a0, a, n_amps)  # paramter domain np.arange(5,a)#


def updateParamInDict_noise(paramDict, paramKey, param):
    paramDict[paramKey] = exT.generateData_ensembleSettings(whiteNoiseAmp=param) #
    return paramDict

noiseAmplitudeO = controlVariable(parameterName=parameter,
                                  paramKey='ensembleSettings',
                                  controlParams=amps,
                                  updateTestSet=False,
                                  updateParamInDict=updateParamInDict_noise,
                                  settingsStr="{}_{}_{}".format(parameter, a0, a)
                                  )




##### Nceps
parameter = 'Nceps'
N0 = 8
Ndelta = 4
N = 34


def updateParamInDict_Nceps(paramDict, paramKey, param):
    paramDict['featExtFun'][paramKey] = param
    return paramDict

NcepsO = controlVariable(parameterName=parameter,
                         paramKey=parameter,
                         controlParams=np.arange(N0, N, Ndelta),
                         updateTestSet=True,
                         updateParamInDict=updateParamInDict_Nceps,
                         settingsStr="{}_{}".format(parameter, '{}_{}'.format(N0,N))
                         )

##### NFFTpow
parameter = 'NFFTpow'
N0 = 8
Ndelta = 1
N = 10
NFFTs = np.arange(N0, N, Ndelta)  # np.linspace(a0, a, n_amps)


def updateParamInDict_NFFT(paramDict, paramKey, param):
    paramDict['featExtFun'][paramKey] = param
    return paramDict

NFFTO = controlVariable(parameterName=parameter,
                        paramKey=parameter,
                        controlParams=NFFTs,
                        updateTestSet=True,
                        updateParamInDict=updateParamInDict_NFFT,
                        settingsStr="{}_{}".format(parameter, '{}_{}'.format(N0,N))
                        )

##### N mels
parameter = 'n_mels'
N0 = 2
Ndelta = 1
N = 7
Nmels = np.arange(N0, N, Ndelta)  # np.linspace(a0, a, n_amps) 


def updateParamInDict_Nmels(paramDict, paramKey, param):
    paramDict['featExtFun'][paramKey] = param 
    return paramDict

NmelsO = controlVariable(parameterName=parameter,
                         paramKey=parameter,
                         controlParams=Nmels,
                         updateTestSet=True,
                         updateParamInDict=updateParamInDict_NFFT,
                         settingsStr="{}_{}".format(parameter, '{}_{}'.format(N0,N))
                         )
                         
                         
##### N mels
#parameter = 'Nceps'
#NcepsO = NmelsO
#NcepsO.NmelsO.paramKey


                        
