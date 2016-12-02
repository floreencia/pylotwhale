#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 18:41:23 2016
@author: florencia
"""

from __future__ import print_function, division
import numpy as np
import os
import sys

import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.MLwhales.MLtools_beta as myML

from sklearn.utils import shuffle
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
import time


def train_clf(X, y, clf_settings):
    gs = GridSearchCV(**clf_settings)
    gs.fit(X, y)
    clf = gs.best_estimator_
    del gs
    return(clf)

def generateData_ensembleSettings(whiteNoiseAmp=0.0025, n_artificial_samples=5):
    '''defines the dictionary with the settings to generate the artificial samples
    adding white noise. See eff.generateWaveformEnsemble
    Parameters:
    -----------
        whiteNoiseAmp=0.0023
        n_artificial_samples = 10
    '''
    ensembleSettings = {"effectName": 'addWhiteNoise'} #, "param_grid" : np.ones(10)}
    ensembleSettings["generate_data_grid"] = np.ones(n_artificial_samples)*whiteNoiseAmp
    return(ensembleSettings)

def featureExtractionInstructions2Xy(wavAnnColl, lt, TpipeSettings, labelSet=None, 
                                     **kwargs):
    """All instructions for feature extraction
    Params:
    -------
        wavAnnColl : 
        lt : label transformer
        featExtFun : feature extraction function
        labelSet : list with the subset of labels to consider
        **feExParamDict : other kwargs for feature extraction
            eg. dict(wavPreprocessingT=None, ensembleSettings=ensembleSettings)
    """
    Tpipe = fex.makeTransformationsPipeline(TpipeSettings)
    feExFun = Tpipe.fun
    datO = fex.wavAnnCollection2Xy_ensemble_datXy_names(wavAnnColl, feExFun,**kwargs)
    X_train, y_train_labels = datO.filterInstances(labelSet)
    y_train = lt.nom2num(y_train_labels)
    return X_train, y_train
        
def clf_experiment(clf_settings, **feExInstructionsDict):
    '''train clf : (1) take params, the ensemble generating params, 
    (2) generate data from collection according to feExFun,
    (3) filter instances and (4) train clf
    Parameters:
    -----------
        < feExInstructionsDict : dictionary with the instrucions for feature extraction 
        eg. dict(wavAnnColl=wavAnnColl_tr, featExtFun=feExFun, labelSet,
                 wavPreprocesingT=None, ensembleSettings=ensembleSettings)
                 see fex.wavAnnCollection2Xy_ensemble()
    '''
    #print(feExInstructionsDict)
    X_train, y_train = featureExtractionInstructions2Xy(**feExInstructionsDict)
    clf = train_clf(X_train, y_train, clf_settings) # train
    return clf  #X_train, y_train # clf


class clf_experimentO():
    '''
    object to run clf experiments
    (1) train classifier and (2) administrate outcomes
    '''
    def __init__(self, clf_settings, **feExParamDict):
        self.clf = clf_experiment(clf_settings, **feExParamDict)
        
    def print_scores(self, scores_file, X_test, y_test, param=None):
        scores = self.clf.score(X_test, y_test)

        with open(scores_file, 'a') as f:
            f.write("{}\t{}\n".format(param, "\t".join("{}".format(scores).split(","))))
            
    def accumPredictions(self, XyDict, param=None, predictionsDict=None):
        
        if predictionsDict is None: 
            predictionsDict = {}
            for li in XyDict.keys():
                X, y = XyDict[li]
                predictionsDict[li]={}
                for i in range(len(y)):
                    predictionsDict[li][i, y[i].item()] = []
                #                            np.zeros((len(y), len(callSet)))))

        for li in XyDict.keys():
            X, y = XyDict[li]
            y_pred = self.clf.predict(X)
            for i in range(len(y_pred)):
                #print(y_pred[i])
                predictionsDict[li][i, y[i].item()].append( y_pred[i])

        return predictionsDict


    def print_predictions(self, accumFile, scoresDict, lt ):
        """prints the predictions for each wav ann"""

        with open(accumFile, 'w') as f:
            f.write("#{}\n".format(", ".join(["{} {}".format(
            call, lt.nom2num(call)) for call in lt.classes_])))

        with open(accumFile, 'a') as f:
            print(scoresDict.keys())
            for fiName in scoresDict:
                f.write("#{}\n".format(fiName))
                for annSec in scoresDict[fiName]:
                    i, annLabel = annSec
                    f.write("{}, {}, {}, {}\n".format(i, lt.nom2num(annLabel), annLabel,
                            ", ".join([str(pred) for pred in scoresDict[fiName][annSec]])))

        return accumFile

#def updateParam(settingsDic):

def updateParamTestSet(wavAnnColl_te, lt, featExtFun,
                       output_type='dict'):
    '''
    recomputes features of the test set
    Parameters:
    ----------
    wavAnnColl_te : collection of annotated wavs
    lt : label transformer
    featExtFun : callable
        extracts audio features for clf from wave file
    output_type : str,
        dict (for keeping track of the files) ot Xy (computing  scores)
    Returns:
    <test_set_features> : as a dict, ar as X, y nparray pair
    '''
    #paramDict['featExtFun'][paramKey] = param # update featExFun
    XyDict_test = fex.wavAnnCollection2datXyDict(wavAnnColl_te,
                                                 featExtFun=featExtFun)

    if output_type =='dict':
        return XyDict_test
    if output_type == 'Xy':
        XyO_test = fex.XyDict2XyO(XyDict_test)
        X_test, y_test_labels = XyO_test.filterInstances(lt.classes_)
        #lt = myML.labelTransformer(y_test_labels)
        y_test = lt.nom2num(y_test_labels)
        return X_test, y_test


def run_iter_clf_experiment(param_grid, paramKey, paramDict,
                            clf_settings, feExParamsDict,
                            #updateParamInDict,
                            wavAnnColl_te, lt,
                            updateTestSet=True,
                            scores_file=None,
                            accum_file=None, ):
    """
    Run a clf experiments for different parameters (param_grid)
    Parameters:
    ----------
    param_grid : array_like
        Experiment parameters.
        Often repeated according to n_experiments
    paramKey : str
        key of paramDict, used to specify the and aupdate the experiments parameter
        which can be a feature name (ceps), NFFT or the instructions for the ensemble
        generation. See updateParamInDict() and feExParamsDict.
    paramDict: dict
        dictionary where the experment param was defined
    clf_settings : dictionary
           clf settings
    feExParamsDict: feExParamDict : dictionary
        Instructions for the extraction of features and ensemble generation.
        Used user over the train set and sometimes also over the test set.
            wavAnnColl : collection of annotated wavs
            lt : label transformer
            featExtFun : feature extraction instructions callable or dicT
            labelSet : set of clf-labels
            ensembleSettings :  instructions for the generation of a sound ensemble (dict)

    updateParamInDict : callable
        Instructions for updating the experimental parameter.
        Often specified by paramsDict.
    wavAnnColl_te : list,
        test collection
    lt : label transformer
    updateTestSet : bool
        True if feture extraction changes over the experiment
        False, no need to update feature representation if the test set
    scores_file : str
        output file for saving clf scores
    accum_file : str
        output file for saving predictions

    """

    ### TEST DATA
    ## data settings
    paramDict[paramKey]=param_grid[0]
    ## 
    feExFun = fex.makeTransformationsPipeline(feExParamsDict["TpipeSettings"]).fun
    XyDict_test = fex.wavAnnCollection2datXyDict(wavAnnColl_te, feExFun)
    XyO_test = fex.XyDict2XyO(XyDict_test)
    X_test, y_test_labels = XyO_test.filterInstances(lt.classes_)
    y_test = lt.nom2num(y_test_labels)

    scoresDict = None

    for param in param_grid:
        paramDict[paramKey]=param  #paramsDict = updateParamInDict(feExParamsDict, paramKey, param)
        print(paramKey, paramDict[paramKey], paramDict, feExParamsDict["TpipeSettings"])
        clfExp = clf_experimentO(clf_settings, **feExParamsDict)
        #print("param", param, '\n\n', paramsDict['featExtFun'])

        if updateTestSet:  # True when changing feature extraction instructions
            feExFun = fex.makeTransformationsPipeline(feExParamsDict["TpipeSettings"]).fun
            XyDict_test = updateParamTestSet(wavAnnColl_te, lt,
                                             featExtFun=feExFun,
                                             output_type='dict')
            X_test, y_test = updateParamTestSet(wavAnnColl_te, lt,
                                                featExtFun=feExFun,
                                                output_type='Xy')

        if scores_file is not None:
            clfExp.print_scores(scores_file, X_test, y_test, param)

        if XyDict_test is not None:
            scoresDict = clfExp.accumPredictions(XyDict_test, param,
                                                 predictionsDict=scoresDict)

    if accum_file is not None:
        clfExp.print_predictions(accum_file, scoresDict, lt)

    return True
