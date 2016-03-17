#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:05:03 2015
#!/usr/bin/python
@author: florencia
"""

from __future__ import print_function, division

import functools
import numpy as np
import time

import sys
import os

import pylotwhale.signalProcessing.signalTools_beta as sT
#import pylotwhale.utils.whaleFileProcessing as fp
import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.MLwhales.MLtools_beta as myML

#from sklearn.preprocessing import LabelEncoder
#from sklearn.utils import shuffle
from sklearn import cross_validation

from sklearn import svm
#from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
#from sklearn import preprocessing
from sklearn import grid_search

from sklearn.pipeline import Pipeline
from sklearn.decomposition import DictionaryLearning#PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


##### SETTINGS
## preprocessing
lb = 1500; hb = 24000; order = 3 # None
wavPreprocessingFun = None#functools.partial(sT.butter_bandpass_filter, lowcut=lb, highcut=hb, order=order)
preproStr = ''#'bandfilter{}_{}'.format(lb, hb)
## features dictionary
featConstD={}
NFFTpow=10; featConstD["NFFTpow"] = NFFTpow
overlap=0.6; featConstD["overlap"]= overlap
#n_mels=128; featConstD["n_mels"]= n_mels; featExtract='melspectro'; featConstD["featExtrFun"]= featExtract
textWS=0.05 ; featConstD["textWS"]= textWS
Nceps=30; featConstD["Nceps"]= Nceps; featExtract='cepstral'; featConstD["featExtrFun"]= featExtract
feExOb = fex.wavFeatureExtractionSplit(featConstD) # feature extraction settings
featExtFun = feExOb.featExtrFun() #functools.partial(sT.waveform2featMatrix, **featConstD)
print("Feature extraction settings", featConstD)
## clf
cv = 10
clfStr = 'cv{}'.format(cv)
### Files
testFrac = 0.3
oDir = '/home/florencia/whales/MLwhales/whaleSoundDetector/data/featureSelection'
collFi_test = '/home/florencia/whales/MLwhales/whaleSoundDetector/data/collection-klein.txt'
collFile = '/home/florencia/whales/MLwhales/whaleSoundDetector/data/collections/cw-all_grB_grJ.txt' #'../data/collection.txt'
fN = os.path.join(oDir, 'out.dat')
out_file = open(fN, 'a')
outModelName = True
labs = ['b', 'c', 'w']
print(fN)

## feature extraction object / function
feExOb = fex.wavFeatureExtraction(featConstD) # feature extraction settings
feature_str = feExOb.feature_str
feExFun=feExOb.featExtrFun()

settingsStr = "{}-{}-{}".format(preproStr, feature_str, clfStr )

#### FUNCTIONS

#### TASKS
out_file.write("###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S")))
out_file.write("#"+settingsStr)

## split collection
WavAnnCollection = fex.readCols(collFile, colIndexes = (0,1))
print("collection:", len(WavAnnCollection), WavAnnCollection[-1])

## extract features - train and test collections
trainDat = fex.wavAnnCollection2datXy(WavAnnCollection, feExFun, wavPreprocesingT=wavPreprocessingFun)

## y_names train and test data
X, y_names = trainDat.filterInstances(labs) # train
lt = myML.labelTransformer(labs)
y = lt.nom2num(y_names)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, 
                                                test_size=testFrac, random_state=0)

out_file.write("\n#TRAIN, shape {}\n".format(np.shape(X_train)))
out_file.write("#TEST, shape {}\n".format(np.shape(X_test))) 

## more info
labsD = lt.targetNumNomDict()
#out_file.write("#X {}, y {}\n".format( np.shape(X), np.shape(y)))
out_file.write("#Target dict {}\t{}\n".format(labsD, trainDat.targetFrequencies()))


##### CLF #######

clf_scores_li=[]

##### SVC -grid
metric='accuracy'
pipe_svc = Pipeline([('clf', svm.SVC(random_state=1) )])

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
                 {'clf__C': param_range,
                  'clf__gamma': param_range,
                  'clf__kernel': ['rbf']}]

gs = grid_search.GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=metric,
                  cv=cv,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)
out_file.write("SVC\t{}\tbest score ({}) {:.3f}\n".format(gs.best_params_, metric, gs.best_score_))
clf_svc_best = gs.best_estimator_
#y_pred = clf_svc_best.predict(X_test)
#cM = confusion_matrix(y_test, y_pred, labels=clf_svc_best.classes_)
### SCORES
## test
scO = myML.clfScoresO(clf_svc_best, X_train, y_train)
out_file.write(scO.scores2str()+'\n')
## train
scO = myML.clfScoresO(clf_svc_best, X_test, y_test)
out_file.write(scO.scores2str()+'\n')
#print()
scO.plConfusionMatrix( labs , outFig=os.path.join(oDir, 'images', 
                    settingsStr +'-sv-CM.png'))

train_sizes, train_scores, test_scores = myML.plLearningCurve(clf_svc_best, X_train, y_train, cv=cv, n_jobs=-1, y_min=0.7)

### clf measurements
clf = clf_svc_best
## learning curve
myML.plLearningCurve(clf, X_train, y_train, cv=cv, y_min=0.7, n_jobs=-1,
                     outFig=os.path.join(oDir, 'images', settingsStr +'-svc.png'))
## scores
myML.printScoresFromCollection(feExFun, clf, lt, collFi_test, out_file)
## save clf cv scores
clf_scores_li.append(('svc', myML.bestCVScoresfromGridSearch(gs)))

##########     RANDOM FOREST #############
pipe_rf = Pipeline([ #('dictLearn', DictionaryLearning(trainDat.m_instances)),
                    ('clf', RandomForestClassifier(n_estimators=100, random_state=1 ))])
pipe_rf.fit(X_train, y_train)
# tests set

#y_pred = pipe_rf.predict(X_test)
out_file.write('RF\n')
## test
scO = myML.clfScoresO(pipe_rf, X_train, y_train)
out_file.write(scO.scores2str()+'\n')
# test
scO = myML.clfScoresO(pipe_rf, X_test, y_test)
out_file.write(scO.scores2str()+'\n')
scO.plConfusionMatrix( labs , outFig=os.path.join(oDir, 'images', 
                    settingsStr +'-rf100-CM.png'))
### clf measurements
clf = pipe_rf
## pl CM

## learning curve
myML.plLearningCurve(clf, X_train, y_train, cv=cv, y_min=0.7, n_jobs=-1,
                     outFig=os.path.join(oDir, 'images', settingsStr + '-rf100.png'))## scores
myML.printScoresFromCollection(feExFun, clf, lt, collFi_test, out_file)
## save clf cv scores


########    RF- grid
ests_range = np.array([50, 100])
clf_init = RandomForestClassifier()

# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "bootstrap": [True, False],
              "n_estimators": ests_range}

gs = grid_search.GridSearchCV(estimator=clf_init, param_grid=param_dist, scoring=metric, cv=cv)
gs = gs.fit(X_train, y_train)
out_file.write("RF\t{}\tbest score ({}) {:.3f}\n".format(gs.best_params_, metric, gs.best_score_))
clf_rf_best = gs.best_estimator_


#y_pred = clf_rf_best.predict(X_test)
#cM = confusion_matrix(y_test, y_pred, labels=clf_rf_best.classes_)
## test
scO = myML.clfScoresO(clf_rf_best, X_train, y_train)
out_file.write(scO.scores2str()+'\n')
# test
scO = myML.clfScoresO(clf_rf_best, X_test, y_test)
out_file.write(scO.scores2str()+'\tTest\n')
scO.plConfusionMatrix( labs , outFig=os.path.join(oDir, 'images', 
                    settingsStr +'-rfgs-CM.png'))

### clf measurements
clf = clf_rf_best
## learning curve
myML.plLearningCurve(clf, X_train, y_train, cv=cv, y_min=0.7, n_jobs=-1,
                    outFig=os.path.join(oDir, 'images', 
                    settingsStr +'-rfgs.png'))## scores
myML.printScoresFromCollection(feExFun, clf, lt, collFi_test, out_file)
## save clf cv scores
clf_scores_li.append(('rf', myML.bestCVScoresfromGridSearch(gs)))

### print cv-scores
for clf_name, clf_pred in  clf_scores_li:
    out_file.write("{:.3f}+-{:.3f}\t".format(clf_pred[0], clf_pred[1]))

### save model
if outModelName:
    import shutil
    #classesStr = "_".join(lt.num2nom(clf.classes_))
    outModelName = os.path.join(oDir, 'models', 'rf-best'+ settingsStr)
    try: 
        shutil.rmtree(outModelName)
    except OSError:
        pass
    os.mkdir(outModelName)
    outModelName+='/model.pkl'
    joblib.dump(clf_rf_best, outModelName)
    out_file.write("\n{}\n".format(outModelName))
    print(outModelName)
    
out_file.write("###---------   {}   ---------###\n".format(time.strftime("%H:%M:%S")))
    

out_file.close()
