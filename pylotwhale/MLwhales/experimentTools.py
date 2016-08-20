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
from sklearn import grid_search
from sklearn.pipeline import Pipeline
from sklearn import svm
import time


def train_clf(X, y, clf_settings):
    gs = grid_search.GridSearchCV(**clf_settings)
    gs.fit(X, y)
    clf = gs.best_estimator_
    del gs
    return(clf)
    
def genrateData_ensembleSettings(param):
    '''defines the dictionary with the settings to generate the artificial samples
    see eff.generateWaveformEnsemble'''
    ensembleSettings = {"effectName" : 'addWhiteNoise'}#, "param_grid" : np.ones(10)}
    ensembleSettings["generate_data_grid"] = np.ones(n_artificial_samples)*param
    return(ensembleSettings)
    
def featureExtractionInstructions2Xy(wavAnnColl, lt, featExtFun=None, labelSet=None,
                                     **feExParamDict):
    '''All instructions for feature extraction
    Params:
    -------
        wavAnnColl : 
        lt : label transformer
        featExtFun : feature extraction function
        labelSet : list with the subset of labels to consider
        **feExParamDict : other kwargs for feature extraction
            eg. dict(wavPreprocessingT=None, ensembleSettings=ensembleSettings)
    '''
    datO = fex.wavAnnCollection2Xy_ensemble(wavAnnColl, featExtFun, **feExParamDict)
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
    return clf #X_train, y_train # clf
    
    
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
        
    
def run_iter_clf_experiment(param_grid, clf_settings, feExParamDict,
                            paramKey, updateParamInDict,
                            print_score_params=(None, None, None), 
                            print_predictions_params=(None, None, None)):
    """
    Parameters:
    ----------
        param_grid : 
        clf_settings : 
        feExParamDict :
        paramKey :
        updateParamInDict :
        print_scores_params : (X, y, out_file)
        print_predictions_params : (XyDict, out_file, lt)
    """
           
    scores_file, X, y = print_score_params
    XyDict, accumFile, lt = print_predictions_params
    scoresDict = None
    
    for param in param_grid:
        print("param", param)
        
        feExParamDict = updateParamInDict(feExParamDict, paramKey, param)
        clfExp = clf_experimentO(clf_settings, **feExParamDict)
        
        if scores_file is not None:
            clfExp.print_scores(scores_file, X, y, param)
            
        if XyDict is not None:
            scoresDict = clfExp.accumPredictions(XyDict, param, 
                                                      predictionsDict=scoresDict)
   
    if XyDict is not None:
        clfExp.print_predictions( accumFile, scoresDict, lt )
        
    return True    