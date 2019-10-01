# coding: utf-8

# # Train classfier from cut wave files and use it to predict annotated segments

# In[1]:


from __future__ import print_function
import os.path
import sys

# import re
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib
# from collections import Counter
# import functools
# import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split  # , grid_search
from sklearn import svm
from sklearn.pipeline import Pipeline

# from sklearn import grid_search
# from sklearn.utils import shuffle
import sklearn.metrics as mt


# In[2]:


import pylotwhale.signalProcessing.signalTools as sT
import pylotwhale.utils.whaleFileProcessing as fp

# import pylotwhale.utils.plotTools as pT

import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.MLwhales.MLtools_beta as myML

# import aupTextFile2mtlFile as a2m
import pylotwhale.NLP.annotations_analyser as aa
import pylotwhale.utils.fileCollections as fC
import pylotwhale.MLwhales.predictionTools as pre

# import pylotwhale.MLwhales.MLtools_beta as myML
import pylotwhale.utils.annotationTools as annT

# from pylotwhale.MLwhales.clf_pool import svc_rbf as clf_settings


# # Settings
#
#
# <!---
# sys.path.append('/home/florencia/profesjonell/bioacoustics/noriega2018sequences/data')
# import params_callClf
# from params_callClf import *
# -->

# In[21]:


pDir = os.path.dirname(os.path.abspath(__file__))


# ## Feature extraction

# In[15]:


fs = 48000  # Heike's
T_settings = []

## preprocessing
filt = "band_pass_filter"
filtDi = {"fs": fs, "lowcut": 0, "highcut": 22000, "order": 5}
# T_settings.append(('bandFilter', (filt, filtDi)))

prepro = "maxabs_scale"
preproDict = {}
T_settings.append(("normaliseWF", (prepro, preproDict)))

#### features dictionary
auD = {}
auD["fs"] = fs
NFFTpow = 9
auD["NFFT"] = 2 ** NFFTpow
overlap = 0.2
auD["overlap"] = overlap
# Nslices = 4; auD["Nslices"] = Nslices
# audioF='spectral'#; auD["featExtrFun"]= featExtract
n_mels = 32
auD["n_mels"] = n_mels
audioF = "melspectro"
# Nceps=23; auD["Nceps"]= Nceps; audioF='MFCC'
T_settings.append(("Audio_features", (audioF, auD)))

summDict = {"Nslices": 5, "normalise": True}
summType = "splitting"
T_settings.append(("summ", (summType, summDict)))

Tpipe = fex.makeTransformationsPipeline(T_settings)
feExFun = Tpipe.fun


# In[16]:


ytest = np.sin(np.arange(0, 100 * np.pi, 0.1))
Mtest = Tpipe.fun(ytest)


def test_features():
    assert (
        Tpipe.string
        == "-normaliseWF-maxabs_scale-"
        + "Audio_features-melspectro-n_mels_32-fs_48000-NFFT_512-overlap_0.2-"
        + "summ-splitting-normalise_True-Nslices_5"
    )
    assert np.shape(Mtest) == (5, 64)
    assert (Mtest[:, 32:] < 0.01).all()
    assert (Mtest[:, 4:7] > 0.65).all()


# ## Load data

# In[18]:


# path to files
train_collection = os.path.join(pDir, "data/groupB_paths2files.csv")
## load data
df = pd.read_csv(train_collection, usecols=["path_to_file", "call"])
wavColl = df.values


# ## Extract features

# In[7]:


datO = myML.dataXy_names()
datO_new = fex.wavLCollection2datXy(wavColl, featExtFun=feExFun, fs=fs)
datO.addInstances(datO_new.X, datO_new.y_names)

## label transformer
call_labels = [l[1] for l in wavColl]
lt = myML.labelTransformer(call_labels)

X = datO.X
y_names = datO.y_names
y = lt.nom2num(y_names)


# In[8]:


def test_extractFeatures():
    assert datO.shape == (1087, 320)
    assert datO.targetFrequencies()["136"] == 79


# # Classifier

# In[9]:


## settings
testFrac = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testFrac, random_state=0)


# cv = 2
# scoring = mt.make_scorer(mt.f1_score, average='macro' )
# paramsDi={}
# pipe_estimators=[]
# pipe_estimators.append(('clf',  clf_settings.fun))
# paramsDi.update(clf_settings.grid_params_di)
# gs_grid = [paramsDi]
#
# ## split data into train and test
# #X_train, y_train = shuffle(X_train, y_train)
#
# ## fit model
# pipe = Pipeline(pipe_estimators)
# gs = grid_search.GridSearchCV(estimator=pipe,
#                               param_grid=gs_grid,
#                               scoring=metric,
#                               cv=cv,
#                               n_jobs=-1)
#
# gs.fit(X_train, y_train)
# clf_svc_best = gs.best_estimator_
#
# print(gs.best_params_)
# ## Train model over whole dataset for predictions
# clf = clf_svc_best.fit(X,y)
#
#
# def test_clf():
#
#     ## predict
#     y_pred = clf_svc_best.predict(X_test)
#
#     ## evaluate
#     np.testing.assert_approx_equal(gs.best_score_, 0.72, significant=1)
#     #assert({'clf__gamma': 0.1, 'clf__C': 10.0} == gs.best_params_)
#     np.testing.assert_approx_equal(gs.score(X_test, y_test), 0.9495, significant=1)
#     np.testing.assert_approx_equal(mt.r2_score(y_test, y_pred), 0.924, significant=1)
#     np.testing.assert_approx_equal(mt.recall_score(y_test, y_pred, average='macro'),
#                                    0.908, significant=1)
#     np.testing.assert_approx_equal(mt.recall_score(y_test, y_pred, average='micro'),
#                                    0.949, significant=1)
#
#     ## full clf
#
#     Tfi, Lfi = annT.anns2TLndarrays( oFi, Tcols=(0,1), Lcols=(-1,))
#     Tp, Lp = pre.TLpredictAnnotationSections( wF, annF, clf, feExFun, lt)
#     np.testing.assert_array_equal(Lfi, Lp)
#     np.testing.assert_array_equal(Tfi, Tp)

# In[10]:


print("\nclf fit...")

clf = Pipeline([("clf", svm.SVC(kernel="rbf", C=10, gamma=0.1))])
## classifier
clf.fit(X_test, y_test)

print("\nclf fitted")


# In[11]:


def test_clf():
    ## predict
    y_pred = clf.predict(X_test)
    ## test scores
    np.testing.assert_approx_equal(mt.r2_score(y_test, y_pred), 0.994136474892, significant=2)
    np.testing.assert_approx_equal(
        mt.recall_score(y_test, y_pred, average="macro"), 0.970685, significant=2
    )
    np.testing.assert_approx_equal(
        mt.recall_score(y_test, y_pred, average="micro"), 0.98165137, significant=2
    )


# In[13]:


def test_predictions():

    # predict annotations
    wF = os.path.join(pDir, "data/WAV_0111_001-48kHz_30-60sec.wav")
    anF = os.path.join(pDir, "data/WAV_0111_001-48kHz_30-60sec.txt")
    predsF1 = os.path.join(pDir, "data/WAV_0111_001-48kHz_30-60sec-preds0.txt")
    pre.predictAnnotationSections(wF, anF, clf, feExFun, lt, outFile=predsF1)
    predsF2 = os.path.join(pDir, "data/WAV_0111_001-48kHz_30-60sec-preds0.txt")
    pre.predictAnnotationSections0(wF, anF, clf, feExFun, lt, outFile=predsF2)

    ## load annotations into T, L format
    Tfi1, Lfi1 = annT.anns2TLndarrays(predsF1, Tcols=(0, 1), Lcols=(-1,))
    Tfi2, Lfi2 = annT.anns2TLndarrays(predsF2, Tcols=(0, 1), Lcols=(-1,))

    # test both prediction types
    np.testing.assert_array_equal(Lfi1, Lfi2)
    np.testing.assert_array_equal(Tfi1, Tfi2)

    # test one more prediction mode
    Tp, Lp = pre.TLpredictAnnotationSections(wF, anF, clf, feExFun, lt)
    np.testing.assert_array_equal(Lfi1, Lp)
    np.testing.assert_array_equal(Tfi1, Tp)
