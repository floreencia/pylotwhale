
# coding: utf-8

# In[ ]:


from __future__ import print_function, division
import sys
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
## sp
import librosa as lb
## ML
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import sklearn.base as skb
import sklearn.metrics as mt

from sklearn import svm


# In[ ]:


import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.signalProcessing.signalTools as sT
import pylotwhale.MLwhales.MLtools_beta as myML
import pylotwhale.utils.annotationTools as annT

import pylotwhale.MLwhales.experiment_WSD1 as wsd
import pylotwhale.signalProcessing.audioFeatures as auf
import pylotwhale.utils.fileCollections as fcll
from pylotwhale.MLwhales.clf_pool import svc_l as clf_settings
import pylotwhale.MLwhales.predictionTools as pre

import pylotwhale.MLwhales.MLEvalTools as MLvl


# In[ ]:


import pylotwhale.MLwhales.clf_pool as clfpool


# In[ ]:


pDir = os.getcwd()


# # Train detector for segmenting whale calls
# 
# Using all data
# 
# import pdb
# 
# %debug -b /home/florencia/whales/scripts/pylotwhale/pylotwhale/MLwhales/featureExtraction.py:29 fex.wavAnnCollection2datXy(collLi, feExFun, labsHierarchy)
# 
# ## Dataset

# In[ ]:


collection = os.path.join(pDir, 'data/collection_NPWcallSegments.txt')
#'/home/florencia/whales/scripts/test_pylotwhale/data/collection_NPWcallSegments.txt'

collList = np.loadtxt(collection, dtype=object, usecols=(0,1), ndmin=2)
#collList
wavF = collList[0,0]

waveform, sr = lb.core.load(wavF, sr=None)
tf = 1.*len(waveform)/sr


# In[ ]:


def test_input():
    assert os.path.basename(wavF) == 'WAV_0111_001-48kHz.wav'
    assert len(waveform) == 16640880
    assert sr == 48000
    np.testing.assert_almost_equal(tf, 346.685, decimal=2)


# ## Feature extraction settings
# 
# Create pipeline for the feature extraction settings
# 
# **y** settings

# In[ ]:


clf_labs = ['c', 'b']
lt = myML.labelTransformer(clf_labs)


# **X** settings

# In[ ]:


T_settings =[]

fs = 48000
auD = {}
auD["fs"] = fs
NFFTpow = 8; auD["NFFT"] = 2**NFFTpow
overlap = 0; auD["overlap"] = overlap
#audioF='spectral'#; auD["featExtrFun"]= featExtract
n_mels = 128/4; auD["n_mels"]= n_mels; audioF='melspectro'; 
#Nceps=2**4; auD["Nceps"]= Nceps; audioF='MFCC'
T_settings.append(('Audio_features', (audioF, auD)))

## sumarisation
summDict = {'n_textWS': 20, 'normalise': True}
summType = 'walking'
T_settings.append(('summ', (summType, summDict)))

Tpipe = fex.makeTransformationsPipeline(T_settings)

##### clf settings
testFrac = 0.2
clf_labs = ['b', 'c', 'w']
labsHierarchy = ['c', 'w']

feExFun = Tpipe.fun


# In[ ]:


dataO = fex.wavAnnCollection2datXy(collList, feExFun, labsHierarchy)


# In[ ]:


X0, y0_names = dataO.filterInstances(lt.classes_)  # filter for clf_labs
X, y_names = X0, y0_names #myML.balanceToClass(X0, y0_names, 'c')  # balance classes X0, y0_names#
y = lt.nom2num(y_names)
labsD = lt.targetNumNomDict()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


def test_featureExtraction():
    assert(dataO.targetFrequencies() == {'b': 2870, 'c': 361, 'n': 3, 'nl': 4, 'w': 12})
    
print('\nFEATURE EXTRACTION')


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=testFrac,
                                                        random_state=0)


# # py.test gets stuck with gridsearch,no need to test it anyway
# 
# 
# pipe_estimators = []
# 
# ## CLF
# pipe_estimators.append(('clf',  clf_settings.fun))
# 
# gs_grid = [clf_settings.grid_params_di]
#  
# scoring = mt.make_scorer(mt.f1_score, pos_label=lt.nom2num('c') )
# 
# cv = 5
# 
# print('\npre gs')
# 
# 
# pipe = Pipeline(pipe_estimators)
# gs = GridSearchCV(estimator=pipe,
#                   param_grid=gs_grid,
#                   scoring=scoring,
#                   cv=cv,
#                   n_jobs=-1)
# 
# gs = gs.fit(X_train, y_train)
# clf = gs.best_estimator_
# 
# '''
# def test_gs():
#     # grid search
#     #assert gs.best_params_ == {'clf__C': 100, 'clf__kernel': 'linear'}
#     #np.testing.assert_approx_equal(gs.best_score_ , 0.833331355189)
#     
# '''

# In[ ]:


print('\nclf fit...')

clf = Pipeline([('clf', svm.SVC(kernel='linear', C=100))])

## classifier
clf.fit(X_test, y_test)

print('\nclf fitted')
clf


# In[ ]:


def test_clf():
    ## clf
    np.testing.assert_approx_equal(mt.f1_score(clf.predict(X_test), y_test) , 0.95999999999999985)


# # Predict

# In[ ]:


y_pred = clf.predict(X)
T_pred, L_pred = annT.clf_y_predictions2TLsections(y_pred, tf=tf, sections=np.array([lt.nom2num(['c'])]))
T_true, L_true = annT.clf_y_predictions2TLsections(y, tf=tf, sections=np.array([lt.nom2num(['c'])]))


# In[ ]:


def test_predictions():
    ## test clf predictions
    np.testing.assert_array_almost_equal( mt.confusion_matrix(y, y_pred), 
                                         np.array([[2837,   33], [  55,  306]]) )
    np.testing.assert_almost_equal(mt.recall_score(y, y_pred), 0.8476454293628809)
    np.testing.assert_almost_equal(mt.precision_score(y, y_pred), 0.90265486725663713)
    assert len(T_pred) == 72
    
def test_sections():
    ## test predicted sections
    np.testing.assert_array_almost_equal(T_pred[0], np.array([ 0.53649799,  1.93139276]))
    np.testing.assert_array_almost_equal(T_pred[-1], np.array([ 279.08625348,  279.19355308]))
    ## test ground truth sections
    np.testing.assert_array_almost_equal(T_pred[0], np.array([ 0.53649799,  1.93139276]))
    np.testing.assert_array_almost_equal(T_pred[-1], np.array([ 279.08625348,  279.19355308]))

