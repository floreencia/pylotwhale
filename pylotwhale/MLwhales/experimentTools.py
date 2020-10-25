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

import sklearn.metrics as mt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import time

import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.MLwhales.MLtools_beta as myML
from pylotwhale.MLwhales import MLEvalTools as MLvl


### General experiment tools

class experiment():
    """in silico experiment
    bounds output file with time"""
    
    def __init__(self, out_file):
        self.out_file = out_file

    @property
    def time(self):
        return time.strftime("%Y.%m.%d\t\t%H:%M:%S")

    def print_in_out_file(self, string):#, oFile=self.out_file):
        with open(self.out_file, 'a') as f:
            f.write(string)

### printing

def Tpipe_settings_and_header(Tpipe, sep=", "):
    """returns two strings with the instructions in Tpipe 
    (pipeline of Transformations)
    header_str, settings_str"""
    header = []
    values = []
    for step in Tpipe.step_sequence:
        header.append(step)
        values.append(Tpipe.steps[step].name)
        for ky in Tpipe.steps[step].settingsDict.keys():
            header.append(ky)
            values.append(Tpipe.steps[step].settingsDict[ky])

    header_str = ("{}".format(sep).join(header))
    settings_str = "{}".format(sep).join(["{}".format(item) \
                                          for item in values])
    return header_str, settings_str


### WSD

class WSD_experiment(experiment):
    """class for WSD experiments, 
    bounds out_file with experiment callable
    Parameters
    ----------
    train_coll, test_coll: list
        2 entry list --> path/to/wav.wav path/to/ann.txt
    test_frac:
    lt: labelTransformer
    labsHierarchy: list
    cv: int
    clf_pipe: 
    clf_grid: dict
    out_file: str
    """
    def __init__(self, train_coll, test_coll, test_frac,
                lt, labsHierarchy, 
                cv, clf_pipe, clf_grid, out_file, metric=None):

        self.train_coll = train_coll
        self.test_coll = test_coll
        self.test_frac = test_frac
        self.lt = lt
        self.labsHierarchy = labsHierarchy

        experiment.__init__(self, out_file=out_file)
        self.cv = cv
        self.clf_pipe = clf_pipe
        self.clf_grid = clf_grid
        self.clf_classes = lt.classes_
        self.metric = metric
        
    def print_comments(self, start='', end='\n'):
        '''time, collections, classes'''
        s = '# {}\n# Coll: {}'.format(self.time, self.train_coll)
        s += '\n# Labels H: {}'.format(self.labsHierarchy)
        self.print_in_out_file(start + s + end)

    def print_experiment_header(self, sep=', ', start='#', end='\n'):
        s = start
        s += set_WSD_experiment_header(self.clf_classes, 
                                       metric=str(self.metric), sep=sep)
        s += end
        self.print_in_out_file(s)

    def run_experiment(self, Tpipe, **kwargs):
        """runs WSD experiment
        for kwargs see run_experiment_WSD 
		e.g. class_balance = class_balance"""
        return run_experiment_WSD(Tpipe=Tpipe, 
                                  train_coll=self.train_coll, 
                                  test_coll=self.test_coll,
                                  test_frac=self.test_frac,
                                  lt=self.lt, 
                                  labsHierarchy=self.labsHierarchy,
                                  out_fN=self.out_file,
                                  cv=self.cv,
                                  clf_pipe=self.clf_pipe, 
                                  gs_grid=self.clf_grid, 
                                  metric=self.metric, **kwargs) 


def set_WSD_experiment_header(clf_class_names, metric='score', sep=', '):
    """String for the WSD experiment"""

    ## n_classes
    n_classes=[];P_classes=[];R_classes=[];f1_classes=[];sup_classes=[]
    for item in clf_class_names:
        n_classes += ["n_" + item]
        P_classes += [item + "_pre"]
        R_classes += [item + "_rec"]
        f1_classes += [item + "_f1"]
        sup_classes += [item + "_sup"]

    ## train & test sets
    scores_li = ['n_train', 'n_test']
    scores_li += [metric + '_CV_train_mean', metric + '_CV_train_2*std',
                  metric + '_CV_test_mean', metric + '_CV_test_2*std']

    header_li = n_classes + scores_li + \
                P_classes + R_classes + f1_classes + sup_classes

    return sep.join(header_li)


def run_experiment_WSD(train_coll, test_coll, test_frac,
                       lt, Tpipe, labsHierarchy, 
                       out_fN,
                       cv, clf_pipe, gs_grid, 
                       class_balance=None, metric=None,
                       predictionsDir=None):
    """Runs clf experiments
    Parameters
    ----------
        train_coll: list
        test_coll: list
        test_frac: float
            fraction of the test set, eg. 0.2
        lt: ML.labelTransformer
        T_settings: list of tuples
        labelsHierachy: list of strings
        cv: cv folds
        estimators: list
            for pipline
        gs_grid: list
                    
        out_fN: str
        returnClfs: dict, Flase => clfs are not stored
        predictionsDir: str
	class_balance: str
		name of the class to balance for
        metric: string or sklearn.metrics.scorer
    """

    feExFun = Tpipe.fun
    #### prepare DATA: collections --> X y
    ## compute features
    dataO = fex.wavAnnCollection2datXy(train_coll, feExFun, labsHierarchy)
    ## prepare X y data
    X0, y0_names = dataO.filterInstances(lt.classes_)  # filter for clf_labs
    if class_balance:
    	X0, y0_names = myML.balanceToClass(X0, y0_names, class_balance)
    X, y_names = X0, y0_names #myML.balanceToClass(X0, y0_names, 'c')  # balance classes X0, y0_names#
    y = lt.nom2num(y_names)
    #labsD = lt.targetNumNomDict()
    ## scores header
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_frac,
                                                        random_state=0)

    #### CLF
    scoring = MLvl.get_scorer(metric)
    pipe = Pipeline(clf_pipe)
    gs = GridSearchCV(estimator=pipe,
                      param_grid=gs_grid,
                      scoring=scoring,
                      cv=cv,
                      n_jobs=-1)

    gs = gs.fit(X_train, y_train)
    
    ### PRINT
    with open(out_fN, 'a') as out_file: # print details about the dataset into status file
        #out_file.write("# {} ({})\n".format( collFi_train, len(train_coll)))
        ## samples per class
        out_file.write(", ".join([str(list(y_names).count(item)) 
                                  for item in lt.classes_]))
        ## sizes of the test/train sets
        out_file.write(", {}, {}".format(len(X_train), len(X_test)))

    ## best clf scores
    with open(out_fN, 'a') as out_file:
        out_file.write('')#", {}".format(str(gs.best_params_).replace('\n', ', '), 
                         #                                     gs.best_score_))
    clf_best = gs.best_estimator_

    ## clf scores over test set
    with open(out_fN, 'a') as out_file:
        ### cv score
        cv_sc = cross_val_score(clf_best, X_test, y_test, scoring=scoring)
        out_file.write(", {:2.2f}, {:.2f}".format(100*np.mean(cv_sc),
                                                  100*2*np.std(cv_sc)))
        ### cv accuracy
        cv_acc = cross_val_score(clf_best, X_test, y_test)
        out_file.write(", {:2.2f}, {:.2f}, ".format(100*np.mean(cv_acc),
                                                    100*2*np.std(cv_acc)))

    ## print R, P an f1 for each class
    y_true, y_pred = y_test, clf_best.predict(X_test)                                                         
    MLvl.print_precision_recall_fscore_support(y_true, y_pred, out_fN)
    
    ### Tpipe -- feature extraction params
    with open(out_fN, 'a') as out_file:
        settings_str = Tpipe_settings_and_header(Tpipe)[1]
        out_file.write(", " + settings_str+'\n')


#### CALL-CLF EXPERIMENT

class callClf_experiment(experiment):
    """class for call clf experiments, from cut wav files
    bounds out_file with experiment callable
    Parameters
    ----------
    train_coll, test_coll: list
        2 entry list --> path/to/wav.wav <label>
    lt: labelTransformer
    labsHierarchy: list
    cv: int
    clf_pipe: 
    clf_grid: dict
    out_file: str
    """
    def __init__(self, train_coll, test_frac,
                lt, labsHierarchy, 
                cv, clf_pipe, clf_grid, out_file,
                test_coll=None, metric=None):

        self.train_coll = train_coll
        self.test_coll = test_coll
        self.test_frac = test_frac
        self.lt = lt
        self.labsHierarchy = labsHierarchy

        experiment.__init__(self, out_file=out_file)
        self.cv = cv
        self.clf_pipe = clf_pipe
        self.clf_grid = clf_grid
        self.clf_classes = lt.classes_
        self.metric = metric
        
    def print_comments(self, start='', end='\n'):
        '''time, collections, classes'''
        s = '# {}\n# Coll: {}'.format(self.time, self.train_coll)
        s += '\n# Labels H: {}'.format(self.labsHierarchy)
        self.print_in_out_file(start + s + end)

    def print_experiment_header(self, Tpipe, sep=',', start='#', end='\n'):
        s = start
        s += scores_header_str(metric=metric, sep=sep)
        s += sep + Tpipe_settings_and_header(Tpipe)[0]
        s += end
        self.print_in_out_file(s)

    def run_experiment(self, Tpipe, **kwargs):
        """runs WSD experiment
        for kwargs see run_experiment_WSD 
		e.g. class_balance = class_balance"""
        return callClfExperiment(Tpipe=Tpipe, 
                                  train_coll=self.train_coll, 
                                  test_coll=self.test_coll,
                                  test_frac=self.test_frac,
                                  lt=self.lt, 
                                  labsHierarchy=self.labsHierarchy,
                                  out_fN=self.out_file,
                                  cv=self.cv,
                                  clf_pipe=self.clf_pipe, 
                                  gs_grid=self.clf_grid, 
                                  metric=self.metric, **kwargs) 


def callClfExperiment(wavColl, lt, Tpipe, out_fN, test_frac,
                      cv, pipe_estimators, gs_grid,
                      filterClfClasses, scoring=None,
                      param=None):
    """Runs clf experiments
    Parameters
    ----------
        train_coll: list
            path/to/wavs    <label>
        test_coll: list
        lt: ML.labelTransformer
        T_settings: list of tuples
        labelsHierachy: list of strings
        cv: cv folds
        estimators: list
            for pipline
        gs_grid: list
        filterClfClasses: list
            can use lt.classes_
        out_fN: str
        returnClfs: dict, Flase => clfs are not stored
        predictionsDir: str
        scoring: string or sklearn.metrics.scorer
        param: float
            value of the param in experimet, for printing
    """

    feExFun = Tpipe.fun
    
    fs = Tpipe.steps['Audio_features'].settingsDict['fs']
    #### prepare DATA: collections --> X y
    ## compute features
    datO = fex.wavLCollection2datXy( wavColl, fs=fs, featExtFun=feExFun )
    X, y_names = datO.filterInstances(filterClfClasses)
    y = lt.nom2num(y_names)
    ## split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_frac, 
                                                        random_state=0)

    #### classify
    pipe = Pipeline(pipe_estimators)
    gs = GridSearchCV(estimator=pipe,
                      param_grid=gs_grid,
                      scoring=scoring,
                      cv=cv,
                      n_jobs=-1)

    gs = gs.fit(X_train, y_train)
    clf_best = gs.best_estimator_
    y_pred = clf_best.predict(X_test)

    #### print clf scores
    settings_str = Tpipe_settings_and_header(Tpipe)[1]
    with open(out_fN, 'a') as out_file:
        ### cv scores
        cv_sc = cross_val_score(clf_best, X_train, y_train, scoring=scoring)
        out_file.write("{:2.2f}, {:.2f}, ".format(100*np.mean(cv_sc),
                                                  100*2*np.std(cv_sc)))
        ### test: ACC, P, R, f1
        # average of the scores for the call classes
        P, R, f1, _ = mt.precision_recall_fscore_support(y_test, y_pred, 
                                                         average='macro') 
        acc = mt.accuracy_score(y_test, y_pred)
        out_file.write("{:.2f}, {:.2f}, {:.2f}, {:.2f}, ".format(acc*100, 
                                                                 P*100, 
                                                                 R*100, 
                                                                 f1*100))
        ### settings
        out_file.write("{}\n".format(settings_str))
        
    return clf_best.fit(X, y)













###### OLD SCRIPTS @ 13.05.17

########

def print_exeriment_header(out_fN, experiment_setup_str, configuration_str, trainFi, 
                           lt, call_labels, exp_header_str):
    """prints the experiment settings"""
    
    ## write in out file
    with open(out_fN, 'a') as out_file: # print details about the dataset into status file
        out_file.write("###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S")))
        out_file.write("# Experiment: {}\n".format(experiment_setup_str))
        # out_file.write("#{}\n".format(lt.classes_))
        out_file.write("#" + configuration_str+'\n')
        ### dateset info
        out_file.write("# {}\n".format( trainFi))
        out_file.write("# label_transformer: {}\n".format(lt.targetNumNomDict()))
        out_file.write("# classes ({}): {}\n# data {}\n".format(len(lt.classes_),
                                                               "', '".join(lt.classes_),
                                                              Counter(call_labels)))        
        ### exp header
        out_file.write("{}\n".format(exp_header_str))

    return out_fN










#########




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
    def __init__(self, parameterName, controlParams, updateTestSet, 
                 paramDict,# updateParamInDict, 
                 settingsStr):
        self.parameter = parameterName
        self.controlParams = controlParams
        self.updateTestSet = updateTestSet
        self.paramDict = paramDict
        self.settingsStr = settingsStr

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
                predictionsDict[li] = {}
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
        print(paramKey, paramDict[paramKey], paramDict)#, feExParamsDict["TpipeSettings"])
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
