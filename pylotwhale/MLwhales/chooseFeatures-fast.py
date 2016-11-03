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
#from sklearn.decomposition import DictionaryLearning#PCA
from sklearn.externals import joblib


##### SETTINGS
featConstD = {}
summDict = {'summarisation': 'walking', 'n_textWS':5, 'normalise':False}
featConstD['summariseDict'] = summDict
NFFTpow = 9; featConstD["NFFTpow"] = NFFTpow
overlap = 0.5; featConstD["overlap"] = overlap
#Nslices = 4; featConstD["Nslices"] = Nslices
#normalize = True; featConstD["normalize"] = normalize
featExtract='spectral'; featConstD["featExtrFun"]= featExtract
feExOb = fex.wavFeatureExtraction(featConstD)  # feature extraction settings

print("Feature extraction settings", featConstD)
## clf
cv = 10
clfStr = 'cv{}'.format(cv)
### Files
testFrac = 0.3
oDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/WSD-clf'
 #'/home/florencia/whales/data/orchive-extract/data/'
#'/home/florencia/whales/MLwhales/whaleSoundDetector/data/featureSelection'
#collFi_test = '/home/florencia/whales/MLwhales/whaleSoundDetector/data/collection-klein.txt'
collFile = '/home/florencia/profesjonell/bioacoustics/heike/NPW/data/collections/wavAnnColl_WSD_grB_HeikesAnns.txt'
#"/home/florencia/whales/data/orchive-extract/collections/callSection-annotations-wavAnnCollection.txt"
#'/home/florencia/whales/MLwhales/whaleSoundDetector/data/collections/cw-all_grB_grJ.txt' #'../data/collection.txt'
fN = os.path.join(oDir, 'out-svc.dat')
out_file = open(fN, 'a')
outModelName = True
labs = ['b', 'c']
print(fN)

## feature extraction object / function
feExOb = fex.wavFeatureExtraction(featConstD) # feature extraction settings
feature_str = feExOb.feature_str
feExFun=feExOb.featExtrFun()

settingsStr = "{}-{}".format( feature_str, clfStr )

#### FUNCTIONS

#### TASKS
out_file.write("###---------   {}   ---------###\n".format(time.strftime("%Y.%m.%d\t\t%H:%M:%S")))
out_file.write("#"+settingsStr)

## split collection
WavAnnCollection = fex.readCols(collFile, colIndexes = (0,1))
print("collection:", len(WavAnnCollection), WavAnnCollection[-1])

## extract features - train and test collections
trainDat = fex.wavAnnCollection2datXy(WavAnnCollection, feExFun) #, wavPreprocesingT=wavPreprocessingFun)

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

gamma_range = [ 0.01, 0.1, 1.0, 10.0, 100.0]
pen_range = [ 1.0, 10.0, 100.0]

param_grid = [ {'clf__C': pen_range, 'clf__gamma': gamma_range, 'clf__kernel': ['rbf']}]

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

#train_sizes, train_scores, test_scores = myML.plLearningCurve(clf_svc_best, X_train, y_train, cv=cv, n_jobs=-1, y_min=0.7)

### clf measurements
clf = clf_svc_best
## learning curve

## scores
#myML.printScoresFromCollection(feExFun, clf, lt, collFi_test, out_file)
## save clf cv scores
clf_scores_li.append(('svc', myML.bestCVScoresfromGridSearch(gs)))

### save model
if outModelName:
    import shutil
    #classesStr = "_".join(lt.num2nom(clf.classes_))
    outModelName = os.path.join(oDir, 'models', 'sv-best'+ settingsStr)
    try: 
        shutil.rmtree(outModelName)
    except OSError:
        pass
    os.mkdir(outModelName)
    outModelName+='/model.pkl'
    joblib.dump(clf, outModelName)
    out_file.write("{}\n".format(outModelName))
    print(outModelName)
    
out_file.write("###---------   {}   ---------###\n".format(time.strftime("%H:%M:%S")))
    

out_file.close()
