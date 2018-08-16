
# coding: utf-8

# ### Tests loading audio from df with file name and annotation section

# In[6]:


from __future__ import print_function, division
import sys
import os
from collections import Counter
import numpy as np
import pandas as pd

import librosa as lb


# In[10]:


import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.signalProcessing.signalTools as sT
import pylotwhale.MLwhales.MLtools_beta as myML


# # Train detector for bats A and B
# 
# instance length 0.05s
# 
# Using all data
# 
# ## Dataset

# In[7]:


flacDir = '/home/florencia/profesjonell/bioacoustics/kdarras/data/flac/'
recl = 'file'
oFigDir = '/home/florencia/profesjonell/bioacoustics/kdarras/data/images/'
oDir = '/home/florencia/profesjonell/bioacoustics/kdarras/data/'

clf_id_str = 'batA-batB'

# params
sr = 192000
fft_ws = 512
sum_ws = 10


# In[8]:


pDir = os.path.dirname(os.path.abspath(__file__))
inF = os.path.join(pDir, 'data/annotations_301117.csv')
#inF = '/home/florencia/profesjonell/bioacoustics/kdarras/data/annotations/annotations_301117.csv'
df0 = pd.read_csv(inF)


# ## Feature extraction settings
# 
# Create pipeline for the feature extraction settings
# 
# **y** settings

# In[26]:


classes = ['noise'] + clf_id_str.split('-')
lt = myML.labelTransformer(classes)

df = df0[df0['type'].isin(classes)]


# **X** settings

# In[12]:


T_settings=[]

#### preprocessing
## band pass filter
filt = 'band_pass_filter'
filtDi = {"fs": sr, "lowcut": 10000, "highcut": 100000, "order": 5}
#T_settings.append(('bandFilter', (filt, filtDi)))
## normalisation
prepro = 'maxabs_scale'
preproDict = {}
#T_settings.append(('normaliseWF', (prepro, preproDict)))

#### audio features
auD = {}
auD["fs"] = sr
auD["NFFT"] = fft_ws
overlap = 0; auD["overlap"] = overlap
n_mels = 4; auD["n_mels"] = n_mels;
fmin = 9000; auD["fmin"] = fmin;
audioF = 'melspectro' #'MFCC'#
T_settings.append(('Audio_features', (audioF, auD)))

#### summ features
summDict = {'n_textWS': sum_ws, 'normalise': True}
summType = 'walking'
T_settings.append(('summ', (summType, summDict)))

print(T_settings)
Tpipe = fex.makeTransformationsPipeline(T_settings)
print(Tpipe.string)

feExFun = Tpipe.fun


# #### Extract features and create data object

# In[13]:


datO = myML.dataXy_names()

classes = ['noise'] + clf_id_str.split('-')

for idx, s in df[:3].iterrows():
    if s['type'] in classes: #

        t0 = s['tmin']
        tf = s['tmax']
        ## load waveform
        y, sr = lb.core.load(os.path.join(flacDir, s[recl] + '.flac'), offset=t0,
                              sr=None, duration=tf-t0)
        #y = auf.getWavSec(y0, sr, t0, tf)
        ## extract features
        try:
            M0 = feExFun(y)
        except AssertionError:
            print(idx, s['type'], "skipping")
            continue
        labs = np.repeat(s['type'], len(M0))
        datO.addInstances(M0, labs)


# In[31]:


def test_load_df_data():
    np.testing.assert_array_equal(lt.classes_ , np.array(['batA', 'batB', 'noise']))
    assert(lt.targetNumNomDict() == {0: 'batA', 1: 'batB', 2: 'noise'})
    assert(T_settings == 
          [('Audio_features',
          ('melspectro',
           {'NFFT': 512, 'fmin': 9000, 'fs': 192000, 'n_mels': 4, 'overlap': 0})),
             ('summ', ('walking', {'n_textWS': 10, 'normalise': True}))])
    assert( datO.targetFrequencies() == {'noise': 68, 'batB': 4})

