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
from pylotwhale.MLwhales.configs.params_callClf import auD, summDict, filesDi


### Control parameter mapping to settings object
def experimentsControlParams(iterParam):
    """
    Sets up the variables for a parameter controled experiment
    given the control parameter name (str), returns an object for
    setting up the experiment variables
    """
    controlParamsDict = {"Nslices": NslicesO,
                         "overlap": overlapO,
                         "n_artificial_samples": ArtificialSamplesO,
                         "whiteNoiseAmp": noiseAmplitudeO,
                         "Nceps": NcepsO,
                         "NFFT": NFFTO,
                         "n_mels" : NmelsO,
                         #"trainCollection": trainCollO
                         }
    assert iterParam in controlParamsDict.keys(), '{} is not a valid control parameter\nValid: {}'.format(iterParam, ', '.join(controlParamsDict.keys()))
    return controlParamsDict[iterParam]


###### Iter parameters

#### Nslices
parameter="Nslices"
Nsl0 = 1
Nsl = 12
controlParam = np.arange(Nsl0, Nsl)
paramDict = summDict

NslicesO = exT.controlVariable(parameterName=parameter,
                               controlParams=controlParam,
                               updateTestSet=True,
                               paramDict=paramDict,
                               settingsStr="{}_{}_{}".format(parameter, 
                                                             Nsl0, Nsl))



#### NArtificialSamples = ensembleSettings[n_artificial_samples]
parameter = "n_artificial_samples"
N0 = 2
N = 10
controlParam_NArtSamps = np.arange(N0, N)  # less than 5 doesn't work with CV=10
from pylotwhale.MLwhales.configs.params_prototypeCallType import ensembleSettingsD
paramDict=ensembleSettingsD

ArtificialSamplesO = exT.controlVariable(parameterName=parameter,
                                     controlParams = controlParam_NArtSamps,
                                     updateTestSet=False,
                                     paramDict=paramDict,
                                     settingsStr="{}_{}_{}".format(parameter, N0, N))


#### overlap

parameter = 'overlap'
N0 = 0
Ndelta = 0.1
N = 1
paramDict=auD

overlapO = exT.controlVariable(parameterName=parameter,
                           controlParams=np.arange(N0, N, Ndelta),
                           updateTestSet=True,
                           paramDict=paramDict,
                           #updateParamInDict=updateParamInDict_over,
                           settingsStr="{}_{}".format(
                                       parameter, '{}_{}'.format(N0, N))
                           )

#### noiseAmplitude
parameter = 'whiteNoiseAmp'
# noise amplitude
n_amps = 10
a0 = 0.001
a = 0.2 # 0.005 for ceps
amps = np.linspace(a0, a, n_amps)  # paramter domain np.arange(5,a)#
paramDict=ensembleSettingsD


noiseAmplitudeO = exT.controlVariable(parameterName=parameter,
                                  controlParams=amps,
                                  updateTestSet=False,
                                  paramDict=paramDict,
                                  #updateParamInDict=updateParamInDict_noise,
                                  settingsStr="{}_{}_{}".format(parameter, a0, a)
                                  )




##### Nceps
parameter = 'Nceps'
N0 = 8
Ndelta = 4
N = 34
paramDict = auD

NcepsO = exT.controlVariable(parameterName=parameter,
                         controlParams=np.arange(N0, N, Ndelta),
                         updateTestSet=True,                           
                         paramDict=paramDict,
                         #updateParamInDict=updateParamInDict_Nceps,
                         settingsStr="{}_{}".format(parameter, '{}_{}'.format(N0,N))
                         )

##### NFFT
parameter = 'NFFT'
N0 = 7
Ndelta = 1
N = 10
NFFTpows = np.arange(N0, N, Ndelta)  # np.linspace(a0, a, n_amps)
NFFTs = np.array([2**p for p in NFFTpows])
paramDict = auD

NFFTO = exT.controlVariable(parameterName=parameter,
                        controlParams=NFFTs,
                        updateTestSet=True,
                        paramDict=paramDict,
                        #updateParamInDict=updateParamInDict_NFFT,
                        settingsStr="{}_{}".format(parameter, '{}_{}'.format(N0,N))
                        )

##### N mels
parameter = 'n_mels'
N0 = 2
Ndelta = 1
N = 30
Nmels = np.arange(N0, N, Ndelta)  # np.linspace(a0, a, n_amps) 
paramDict = auD


NmelsO = exT.controlVariable(parameterName=parameter,
                         controlParams=Nmels,
                         updateTestSet=True,
                         paramDict=paramDict,
                         #updateParamInDict=updateParamInDict_NFFT,
                         settingsStr="{}_{}".format(parameter, '{}_{}'.format(N0,N))
                         )
                         
                         
##### trainCollO
parameter = 'wavAnnColl'
expSettings = [1,7,1]
collTemplateName = '/home/florencia/whales/data/Vocal-repertoire-catalogue-Pilot-whales-Norway/flo/annotations/callClassifier/collections/wavAnnColl_calltypes-NSAMPSSamplesTrainCollection.txt'
controlParams = [np.genfromtxt(collTemplateName.replace('NSAMPS', "%s"%i), dtype=object) for i in np.arange(*expSettings)]
paramDict = filesDi
settingsStr = '{}_{}'.format(parameter, '_'.join(['{}'.format(item) for item in expSettings]))

trainCollO = exT.controlVariable(parameterName=parameter,
                             controlParams=controlParams,
                             updateTestSet=False,
                             paramDict=paramDict,
                             settingsStr=settingsStr)

