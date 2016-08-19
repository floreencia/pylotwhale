#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:05:03 2015
#!/usr/bin/python
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

#######################   SETTINGS   ######################################

###### Iter parameters
parameter = 'noiseAmplitude'
n_artificial_samples = 10 # number of artificial samples to generate for each amp
n_experiments = 10 # identical experiment repetitions
# noise amplitude
n_amps = 10
a0 = 0
a = 0.005
amp = np.linspace(a0, a, n_amps) # paramter domain
param_grid = np.repeat(amp, n_experiments) # reapet expriment
metric='accuracy'
preproStr="n_idExperiments{}-n_trainSamples{}_in_{}_{}".format(n_experiments,
                                                 n_artificial_samples, a0, a)

##### out files
oDir = os.path.join('/home/florencia/whales/MLwhales/callClassification/data/experiments/trashtest', parameter)
try:
    os.makedirs(oDir)
except OSError:
    pass
out_file_scores = os.path.join(oDir, "scores.txt")
out_file_votes = os.path.join(oDir, "votes.txt")


##### Feature extraction 

## preprocessing
lb = 1500; hb = 24000; order = 3 # None
wavPreprocessingFun = None#functools.partial(sT.butter_bandpass_filter, lowcut=lb, highcut=hb, order=order)
preproStr +=''#'bandfilter{}_{}'.format(lb, hb)

## features dictionary
featConstD = {}
NFFTpow = 10; featConstD["NFFTpow"] = NFFTpow
overlap = 0.5; featConstD["overlap"]= overlap
Nslices = 8; featConstD["Nslices"]= Nslices
normalize = True; featConstD["normalize"]= normalize
#featExtract='spectral'; featConstD["featExtrFun"]= featExtract
#n_mels = 64; featConstD["n_mels"]= n_mels; featExtract='melspectro'; featConstD["featExtrFun"]= featExtract
Nceps=20; featConstD["Nceps"]= Nceps; featExtract='cepstral'; featConstD["featExtrFun"]= featExtract
## feature extraction object
feExOb = fex.wavFeatureExtractionSplit(featConstD) # feature extraction settings
feature_str = feExOb.feature_str
feExFun=feExOb.featExtrFun()

##### clf
cv = 10
clfStr = 'cv{}'.format(cv)

pipe_svc = Pipeline([('clf', svm.SVC(random_state=0) )])
gamma_range = [ 0.01, 0.1, 1.0, 10.0, 100.0]
pen_range = [ 1.0, 10.0, 100.0]
clf_param_grid = [ {'clf__C': pen_range, 'clf__gamma': gamma_range, 
                    'clf__kernel': ['rbf']}]
gs_settings = dict(estimator=pipe_svc,
                  param_grid=clf_param_grid,
                  scoring=metric,
                  cv=cv,
                  n_jobs=-1)
gs = grid_search.GridSearchCV(**gs_settings)
                  
#### Classes                
callSet=['126i', '130', '127', '129', '128i', '131i', '093ii']
lt = myML.labelTransformer(callSet)

##### Data
## collection files
collFi_train = '/home/florencia/whales/data/Vocal-repertoire-catalogue-Pilot-whales-Norway/flo/wavs/wavFiles-wavAnnCollection-prototypes.txt'
collFi_test = '/home/florencia/whales/MLwhales/callClassification/data/collections/grB-balanced14collection.txt'
## collections
wavAnnColl_tr = fex.readCols(collFi_train, (0,1))
wavAnnColl_te = fex.readCols(collFi_test, (0,1))

## test data
XyDict_test=fex.wavAnnCollection2XyDict(wavAnnColl_te, feExFun)
XyO_test = fex.XyDict2XyO(XyDict_test)
X_test, y_test_labels = XyO_test.filterInstances(callSet)
lt = myML.labelTransformer(y_test_labels)
y_test = lt.nom2num(y_test_labels)

## feature extraction object / function
settingsStr = "{}-{}-{}".format(preproStr, feature_str, clfStr )

######### Functions #######

def train_clf(X, y):
    gs = grid_search.GridSearchCV(**gs_settings)
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
    
def clf_experiment(param):
    '''Train clf : (1) take params, "the ensemble generating params", 
    (2) generate data from collection according to feExFun, 
    (3) filter instances and (4) train clf'''
    # (1)
    ensembleSettings = genrateData_ensembleSettings(param)
    datO = fex.wavAnnCollection2Xy_ensemble(wavAnnColl_tr, featExtFun=feExFun, 
                                                ensembleSettings=ensembleSettings)
                                                
    X_train, y_train_labels = datO.filterInstances(callSet)
    y_train = lt.nom2num(y_train_labels)            
    clf = train_clf(X_train, y_train) # train           
    return clf
    
    
class clf_experimentO():
    def __init__(self, param):
        self.clf = clf_experiment(param)
        
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
        
            
    def print_predictions(self, accumFile, scoresDict ):
        """prints the predictions for each wav ann"""
        
        with open(accumFile, 'w') as f:
            f.write("#{}\n".format(", ".join(["{} {}".format(call, lt.nom2num(call)) for call in callSet])))
        
        with open(accumFile, 'a') as f:
            print(scoresDict.keys())
            for fiName in scoresDict:
                f.write("#{}\n".format(fiName))
                for annSec in scoresDict[fiName]:
                    i, annLabel = annSec
                    f.write("{}, {}, {}, {}\n".format(i, lt.nom2num(annLabel), annLabel,
                            ", ".join([str(pred) for pred in scoresDict[fiName][annSec]])))
        
        return accumFile
        
    
def run_iter_clf_experiment(param_grid, 
                            print_score_params=(None, None, None), 
                            print_predictions_params=(None, None)):
    """
    print_scores_params : (X, y, out_file)
    print_predictions_params : (XyDict, out_file)
    """
           
    scores_file, X, y = print_score_params
    XyDict, accumFile = print_predictions_params
    scoresDict = None
    
    for param in param_grid:
        print("param", param)
        
        clfExp = clf_experimentO(param)
        
        if scores_file is not None:
            clfExp.print_scores(scores_file, X, y, param)
            
        if XyDict is not None:
            scoresDict = clfExp.accumPredictions(XyDict, param, 
                                                      predictionsDict=scoresDict)
   
    if XyDict is not None:
        clfExp.print_predictions( accumFile, scoresDict )
                
        
    return True    


###################  TASK  ####################

## print experiment settings header
with open(out_file_scores, 'w') as f:
    f.write("#{}\n#TRAIN: {}\n#TEST: {}\n#{}\n#{}\t{}\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S"), 
            collFi_train, collFi_test, settingsStr, parameter, metric))

run_iter_clf_experiment(param_grid, print_score_params=(out_file_scores, X_test, y_test),
                        print_predictions_params=(XyDict_test, out_file_votes))


    
#### Ensemble settings









sys.exit()
for param in param_grid:
    print("param", param)
    
    #### generate data
    ensembleSettings["generate_data_grid"] = np.ones(n_artificial_samples)*param
    datO = fex.wavAnnCollection2Xy_ensemble(wavAnnColl_tr, featExtFun=feExFun, 
                                            ensembleSettings=ensembleSettings)
    X_train, y_train_labels = datO.filterInstances(callSet)    
    y_train = lt.nom2num(y_train_labels)    
    X_train, y_train = shuffle(X_train, y_train)
    
    #### train   
    clf = train_clf(X_train, y_train)
    
    #### scores
    scores = clf.score(X_test, y_test)
 
    ## print
    with open(out_file, 'a') as f:
        f.write("{}\t{}\n".format(param, "\t".join("{}".format(scores).split(","))))
    
    del gs





'''
    ## format
    try:
        len(scores)
    except TypeError:
        scores = [scores]  
        
        
'''