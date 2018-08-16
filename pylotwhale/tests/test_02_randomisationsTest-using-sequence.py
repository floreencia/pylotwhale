
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import nltk
import seaborn as sns
import pandas as pd
import scipy as sp
from scipy.stats import entropy
from collections import Counter, defaultdict
import networkx as nx


# In[2]:


import pylotwhale.utils.whaleFileProcessing as fp
import pylotwhale.utils.fileCollections as fcll
import pylotwhale.utils.annotationTools as annT
import pylotwhale.NLP.annotations_analyser as aa
import pylotwhale.NLP.ngramO_beta as ngr

import pylotwhale.utils.dataTools as daT
import pylotwhale.utils.plotTools as pT

import pylotwhale.NLP.myStatistics_beta as mysts
import pylotwhale.NLP.tempoTools as tT
import pylotwhale.utils.netTools as nT


# # Load df

# In[3]:


df_file = '/home/florencia/profesjonell/bioacoustics/noriega2018sequences/data/groupB_annotations_df.csv'
# load
df = pd.read_csv(df_file)#; df= df0
# tape separation
tapedf = daT.dictOfGroupedDataFrames(df)

def test_data():
    # N_calls
    assert(len(df) == 425)
    # N_call types
    assert(len(set(df['call'])) == 22)
    # tapes set
    assert(set(df['tape'].values) ==  set([113, 114, 115, 111]))
    assert(set(tapedf.keys()) == set([113, 114, 115, 111]))


# # Bigrams  and randomisations test
# 
# Define the **sequences**

# In[4]:


## parameters and settings
Dt = 0.3; Dtint = (None, Dt)
## define the sequences
sequencesList0 = aa.dfDict2listOfSeqs(tapedf, Dt=Dtint, l='call', time_param='ici')
## filter out isolated calls (sequences of size one)
sequencesList = [l for l in sequencesList0 if len(l)>1]

seqO = mysts.sequenceBigrams(sequencesList)

def test_seqO():
    assert(len(seqO.seqOfSeqs) == 71)
    assert(seqO.callCounts['129'] == 89)


# **Bigrams**, counts and probabilities

# In[5]:


## arange order of the calls
minCalls = 5
condsLi = seqO.conditionsLi(minCalls)
sampsLi = seqO.samplesLi(minCalls)
print("conds", condsLi, '\nsamps', sampsLi)

## filter out bigrams observed less than 5 times
minBigrams=3
mask = seqO.df_cfd.loc[condsLi, sampsLi] < minBigrams #sys.float_info.min

def test_conds_samps():
    assert(np.shape(mask) == (len(condsLi), len(sampsLi)))


# ### Randomise 
# 
# randomise
# 
# <!---
# #### less than minCalls
# mask_p_value = p_values > pc
# mask_minCalls = M <= minCalls
# mask = np.logical_or(mask_p_value, mask_minCalls)# mask_minCalls #mask_p_value# 
# 
# f,ax = plt.subplots(1,3, figsize=(18, 4))
# sns.heatmap(mask_minCalls, ax=ax[0])
# sns.heatmap(mask_p_value, ax=ax[1])
# sns.heatmap(mask, ax=ax[2])
# 
# !--->

# In[21]:


Nsh = 200
np.random.seed(0)

p_values, sh_dists = mysts.randtest4bigrmas_inSequences(seqO.seqOfSeqs, Nsh,
                                                        condsLi=condsLi, sampsLi=sampsLi)


# In[22]:


def test_p_values():
    np.testing.assert_approx_equal(114.93, np.sum(p_values.values), significant=3)
    np.testing.assert_approx_equal(p_values.loc['129', '129'], 0)
    np.testing.assert_approx_equal(p_values.loc['130', '130'], 0)
    assert(p_values.loc['130', '129'] > 0.2)
    


# ### Network

# In[23]:


pc = 0.01
bigrams_pc = daT.get_indexColFromDataFrame(p_values, lambda x: x > pc)

kkD_counts = seqO.df_cfd
minBigrams = 3
bigrams_minBigrams = daT.get_indexColFromDataFrame(seqO.df_cfd, lambda x: x < minBigrams)

rmBigrams = bigrams_pc | bigrams_minBigrams

netO = nT.dict2network(seqO.cfd)


# In[24]:


def test_network():
    assert(len(bigrams_pc) == 133)
    assert(len(bigrams_minBigrams) == 374)
    assert(len(rmBigrams) == 386)
    assert(netO.net_dict['079'] == {'126i': 1, '129': 1})

