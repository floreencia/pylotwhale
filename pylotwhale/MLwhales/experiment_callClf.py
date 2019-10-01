"""
Runs call classification experiments generating artificial data and trying
different parameters
"""
from __future__ import print_function
import os

# import argparse
# import sys
import numpy as np
import time

# import pdb
from collections import Counter

# from sklearn.svc import SVC
# import sklearn.base as skb
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

# from sklearn.externals import joblib
# from sklearn import cross_validation


import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.MLwhales.MLtools_beta as myML

# import pylotwhale.MLwhales.predictionTools as pT

# import pylotwhale.MLwhales.experimentTools as exT
from pylotwhale.MLwhales import MLEvalTools as MLvl

### load parameters
from pylotwhale.MLwhales.configs.params_callClf import *

###################  ASSIGNMENTS  ####################
##### OUTPUT FILES
"""
Runs call clf experiments from a collection of labeled cutted wave files


try:
    os.makedirs(oDir)
except OSError:
    pass
out_fN = os.path.join(oDir, "scores.txt")


if isinstance(predictionsDir, str):
    predictionsDir = os.path.join(oDir, "predictions")
    try:
        os.makedirs(predictionsDir)
    except OSError:
        pass


Tpipe = fex.makeTransformationsPipeline(T_settings)

## clf settings
clfStr = 'cv{}-'.format(cv)
settingsStr = "{}-{}".format(Tpipe.string, clfStr)
settingsStr += '-labsHierarchy_' + '_'.join(labsHierarchy)

## write in out file
out_file = open(out_fN, 'a')
out_file.write("#WSD1\n###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S")))
out_file.write("#" + settingsStr+'\n')
out_file.close()

## load collections
train_coll = fex.readCols(collFi_train, colIndexes=(0,1))
test_coll = np.genfromtxt(collFi_test, dtype=object)

lt = myML.labelTransformer(clf_labs)
"""

# wavColl = fex.readCols(filesDi['train'], (0,1))
# labels = [l[1] for l in wavColl]
# lt = myML.labelTransformer(labels)


def prepare_output_file(
    outDir,
    expSettingsStr,
    settingsStr,
    trainFi,
    lt,
    call_labels,
    exp_header_str="",
    fileName="scores.txt",
):
    """prints the experiment settings"""
    # oDir = os.path.join(outDir, iterParam)

    try:
        os.makedirs(outDir)
    except OSError:
        pass
    out_fN = os.path.join(outDir, fileName)

    ## write in out file
    print_exeriment_header(
        out_fN, expSettingsStr, settingsStr, trainFi, lt, call_labels, exp_header_str
    )
    return out_fN


def print_exeriment_header(
    out_fN, experiment_setup_str, configuration_str, trainFi, lt, call_labels, exp_header_str
):
    """prints the experiment settings"""

    ## write in out file
    with open(out_fN, "a") as out_file:  # print details about the dataset into status file
        out_file.write(
            "###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S"))
        )
        out_file.write("# Experiment: {}\n".format(experiment_setup_str))
        # out_file.write("#{}\n".format(lt.classes_))
        out_file.write("#" + configuration_str + "\n")
        ### dateset info
        out_file.write("# {}\n".format(trainFi))
        out_file.write("# label_transformer: {}\n".format(lt.targetNumNomDict()))
        out_file.write(
            "# classes ({}): {}\n# data {}\n".format(
                len(lt.classes_), "', '".join(lt.classes_), Counter(call_labels)
            )
        )
        ### exp header
        out_file.write("{}\n".format(exp_header_str))

    return out_fN


def scores_header_li(metric="metric"):
    return ["{}_CV".format(metric), "{}_CV_std".format(metric), "ACC", "PRE", "REC", "F1"]


def scores_header_str(header_li=None, metric="metric", sep=","):
    if header_li is None:
        header_li = scores_header_li(metric)
    exp_header_str = header_li[0]
    for item in header_li[1:]:
        exp_header_str += "{}{}".format(sep, item)

    return exp_header_str


def Tpipe_settings_and_header(Tpipe, sep=", "):
    """returns two strings with the instructions in Tpipe (pipeline of Transformations)
    header_str, settings_str"""
    header = []
    values = []
    for step in Tpipe.step_sequence:
        header.append(step)
        values.append(Tpipe.steps[step].name)
        for ky in Tpipe.steps[step].settingsDict.keys():
            header.append(ky)
            values.append(Tpipe.steps[step].settingsDict[ky])

    header_str = "{}".format(sep).join(header)
    settings_str = "{}".format(sep).join(["{}".format(item) for item in values])
    return header_str, settings_str


def callClfExperiment(
    wavColl,
    lt,
    Tpipe,
    out_fN,
    testFrac,
    cv,
    pipe_estimators,
    gs_grid,
    filterClfClasses,
    scoring=None,
    param=None,
):
    """Runs clf experiments
    Parameters
    ----------
        train_coll: list
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

    # Tpipe = fex.makeTransformationsPipeline(T_settings)
    feExFun = Tpipe.fun

    fs = Tpipe.steps["Audio_features"].settingsDict["fs"]
    #### prepare DATA: collections --> X y
    ## compute features
    datO = fex.wavLCollection2datXy(wavColl, fs=fs, featExtFun=feExFun)
    X, y_names = datO.filterInstances(filterClfClasses)
    y = lt.nom2num(y_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testFrac, random_state=0)

    #### CLF
    pipe = Pipeline(pipe_estimators)
    gs = GridSearchCV(estimator=pipe, param_grid=gs_grid, scoring=scoring, cv=cv, n_jobs=-1)

    gs = gs.fit(X_train, y_train)
    clf_best = gs.best_estimator_
    y_pred = clf_best.predict(X_test)

    ## clf scores over test set

    settings_str = Tpipe_settings_and_header(Tpipe)[1]
    with open(out_fN, "a") as out_file:
        ### cv scores
        cv_sc = cross_val_score(clf_best, X_train, y_train, scoring=scoring)
        out_file.write("{:2.2f}, {:.2f}, ".format(100 * np.mean(cv_sc), 100 * 2 * np.std(cv_sc)))
        ### test: ACC, P, R, f1
        P, R, f1, _ = mt.precision_recall_fscore_support(
            y_test, y_pred, average="macro"
        )  # average of the scores for the call classes
        acc = mt.accuracy_score(y_test, y_pred)
        out_file.write(
            "{:.2f}, {:.2f}, {:.2f}, {:.2f}, ".format(acc * 100, P * 100, R * 100, f1 * 100)
        )

        ### settings
        out_file.write("{}\n".format(settings_str))

    return clf_best.fit(X, y)


def runCallClfExperiment(
    wavColl,
    lt,
    T_settings,
    out_fN,
    testFrac,
    cv,
    pipe_estimators,
    gs_grid,
    filterClfClasses,
    scoring=None,
    param=None,
):
    """ . . . . . USE callClfExperiment instead  to be deprecated!!! . . . . .
    Runs clf experiments in the old way
    using T_settings instead of Tpipe
    Parameters
    ----------
        train_coll: list
        test_coll: list
        lt: ML.labelTransformer
        T_settings: list of tuples
        labelsHierarchy: list of strings
        cv: cv folds
        pipe_estimators: list
            for pipeline
        gs_grid: list
        filterClfClasses: list
            can use lt.classes_
        out_fN: str
        returnClfs: dict, Flase => clfs are not stored
        predictionsDir: str
        scoring: string or sklearn.metrics.scorer
        param: float
            value of the param in experiment, for printing
    """

    Tpipe = fex.makeTransformationsPipeline(T_settings)
    feExFun = Tpipe.fun
    fs = Tpipe.Audio_features.fs
    #### prepare DATA: collections --> X y
    ## compute features
    datO = fex.wavLCollection2datXy(wavColl, fs=fs, featExtFun=feExFun)
    X, y_names = datO.filterInstances(filterClfClasses)
    y = lt.nom2num(y_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testFrac, random_state=0)

    #### CLF
    pipe = Pipeline(pipe_estimators)
    gs = GridSearchCV(estimator=pipe, param_grid=gs_grid, scoring=scoring, cv=cv, n_jobs=-1)

    gs = gs.fit(X_train, y_train)
    clf_best = gs.best_estimator_
    y_pred = clf_best.predict(X_test)

    ## clf scores over test set
    with open(out_fN, "a") as out_file:
        ### cv score
        cv_sc = cross_val_score(clf_best, X_train, y_train, scoring=scoring)
        out_file.write(
            "{:}, {:2.2f}, {:.2f}, ".format(param, 100 * np.mean(cv_sc), 100 * 2 * np.std(cv_sc))
        )

        P, R, f1, _ = mt.precision_recall_fscore_support(
            y_test, y_pred, average="macro"
        )  # average of the scores for the call classes
        acc = mt.accuracy_score(y_test, y_pred)
        out_file.write(
            "{:.2f}, {:.2f}, {:.2f}, {:.2f}\n".format(acc * 100, P * 100, R * 100, f1 * 100)
        )

    return clf_best.fit(X, y)


if False:
    #### feature extraction object
    feExOb = fex.wavFeatureExtraction(featConstD)  # feature extraction settings
    feature_str = feExOb.feature_str
    feExFun = feExOb.featExtrFun()

    #### clf settings
    clfStr = "cv{}-{}".format(cv, metric)
    settingsStr = "{}-{}".format(feature_str, clfStr)

    #### write in out file
    out_file = open(out_file_scores, "a")
    out_file.write(
        "\n###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S"))
    )
    out_file.write("#" + settingsStr)
    out_file.close()

    #### load collection
    WavAnnCollection = fex.readCols(collFi_train, colIndexes=(0, 1))
    print("\ncollection:", len(WavAnnCollection), "\nlast file:", WavAnnCollection[-1])

    #### compute features
    trainDat = fex.wavAnnCollection2datXy(
        WavAnnCollection, feExFun
    )  # , wavPreprocesingT=wavPreprocessingFun)
    ## y_names train and test data
    X, y_names = trainDat.filterInstances(labs)  # train
    lt = myML.labelTransformer(labs)
    y = lt.nom2num(y_names)
    labsD = lt.targetNumNomDict()
    #### train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testFrac, random_state=0)

    with open(out_file_scores, "a") as out_file:  # print details to status file
        out_file.write("\n#TRAIN, shape {}\n".format(np.shape(X_train)))
        out_file.write("#TEST, shape {}\n".format(np.shape(X_test)))

        ## more info
        # out_file.write("#X {}, y {}\n".format( np.shape(X), np.shape(y)))
        out_file.write("#Target dict {}\t{}\n".format(labsD, trainDat.targetFrequencies()))

    ### grid search

    gs = GridSearchCV(
        estimator=pipe_estimators, param_grid=gs_grid, scoring=metric, cv=cv, n_jobs=-1
    )

    gs = gs.fit(X_train, y_train)

    with open(out_file_scores, "a") as out_file:
        out_file.write(MLvl.gridSearchresults(gs))

    print(out_file_scores)
