
# coding: utf-8

# # Explore sequences of calls
# 
# ### Quantifying animal vocal sequences
# 

# In[5]:


from __future__ import print_function, division

import sys
import os

import numpy as np

import pandas as pd

import scipy as sp
from collections import Counter, defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
from matplotlib import ticker

from sklearn.manifold import MDS
from scipy.stats import entropy
from sklearn.model_selection import ParameterGrid


# In[6]:


import pylotwhale.utils.whaleFileProcessing as fp
import pylotwhale.utils.fileCollections as fcll
import pylotwhale.utils.plotTools as pT
import pylotwhale.utils.dataTools as daT

import pylotwhale.utils.annotationTools as annT
import pylotwhale.NLP.annotations_analyser as aa
import pylotwhale.NLP.myStatistics_beta as myst
import pylotwhale.NLP.tempoTools as tT
import pylotwhale.MLwhales.MLtools_beta as myML

import nltk
import pylotwhale.NLP.ngramO_beta as ngr

import pylotwhale.signalProcessing.audioFeatures as auF

reload(daT)


# # Plotting settings

# Settings for the analysis

# In[7]:


call_label = 'cluster_four'
time_param = 'ici'
cs = {'sex': 'm', 'predator': 'leo'}

id_str = '{}-{}-{}'.format(call_label, time_param, "-".join(['{}_{}'.format(k, y) for k, y in cs.items()]))
print(id_str)


# # Load dataframe of annotations

# In[8]:


cfile = '/home/florencia/profesjonell/bioacoustics/Kurt/green/data/vervet_tc_annotations_df-1cs.txt'

df0 = pd.read_csv(cfile)

df = df0


# ## Segment tapes when missing calls
# 
# 
# Initialise segment column

# In[9]:


df['segment'] = np.nan


# Define segments looking at the $\Delta$ tag 'Dtag'

# In[10]:


tapes = list(set(df['tape']))
for t in tapes:
    thisdf = df[df['tape'] == t]
    j = 0
    relabel_tape = [] 
    for i, col in thisdf.iterrows():
        if col['Dtag'] == 1:  # continuous
            df.loc[i, 'segment'] = '{}_{}'.format(t, j)
            continue
        elif(col['Dtag'] < 0 and 
           df.loc[i - 1, 'tape'] == col['tape']):  # last element in the tape
            df.loc[i, 'segment'] = '{}_{}'.format(t, j)
            break
        elif col['Dtag'] > 1:  # elements missing -->  new segment
            j+=1
            df.loc[i, 'segment'] = '{}_{}'.format(t, j)
            continue
        else:
            print("else! ", i, col['Dtag'] )
            break
    


# ### Separate data frames by segment and drop nans
# The nans, come from all the non labelled items in the previous step (segment assignation) and they correspond to missing calls and new tapes.

# In[11]:


tape_df0 = daT.dictOfGroupedDataFrames(df, groupingKey='segment')
# filter df with only one element and segments nan segment
tape_df = {k: v for k, v in tape_df0.items() if (len(v) > 1 and k != np.nan) }


# In[12]:


def test_segment_tape_df0():
    assert(len(tape_df0) == 232)
    assert(len(tape_df) == 175)
    
test_segment_tape_df0()


# ## Distrubution of N-grams as a function of $\tau$

# In[25]:


t0 = 0
tf = 1
n_time_steps = 300
Dtvec = np.linspace(t0, tf, n_time_steps)

# count number of Ngrans of size N
ngramDist_Dt = aa.Ngrams_distributionDt_ndarray(tape_df, Dtvec,
                                                seqLabel='cluster_four',
                                                time_param='ici')

# count the number of calls in Ngrams of size N,  i.e. multiplies the Ngram distribution by N (the size of the Ngram)
calls_in_ngramDist_Dt = aa.NgramsDist2callsInNgramsDist(ngramDist_Dt)


# In[30]:


def test_ngramDist():
    assert(np.shape(ngramDist_Dt) == (300, 55))

test_ngramDist()


# Plot the distribution of Ngrams

# In[31]:


Dt_chunks=0.1
Dtint_chunks=(None, Dt_chunks)

## useful numbers
noteFreqs = Counter(df[call_label].values) #notes = daT.returnSortingKeys(noteFreqs)
notes = list(set(df[call_label].values))
## define the sequences
sequences = []
for t in tape_df.keys(): # for each tape
    this_df = tape_df[t]
    sequences += aa.df2listOfSeqs( this_df, Dt=Dtint_chunks, l=call_label, time_param=time_param) # define the sequeces
## sequence statistics
ngram_seqs = defaultdict(list)
for s in sequences:
    ngram_seqs[len(s)].append(tuple(s))    
## count notes in each sequence size
note_chunks_arr = np.zeros((len(noteFreqs), np.max(ngram_seqs.keys())+1))
for j in ngram_seqs.keys():
    s = ngram_seqs[j]
    for i,n in enumerate(notes):
        note_chunks_arr[i,j] = sum(x.count(n) for x in s)
        
seqSizes = np.arange(1, np.shape(note_chunks_arr)[1] + 1)
## sequence stats
Ngrams_dist = np.array([len(ngram_seqs[k]) for k in ngram_seqs.keys()])
calls_in_Ngrams = np.array([(k+1)*n_k for k, n_k in enumerate(Ngrams_dist)])
N = sum(calls_in_Ngrams)
N_grouped = sum(calls_in_Ngrams[1:])
print(id_str,
    "\nnumber grouped notes: {}/{}".format( N_grouped, N),
      "\nNgrams distribution ({}): {}".format(Dtint_chunks, Ngrams_dist), 
      "\nPercentage of notes in Ngrams (1,{}): {}".format(len(Ngrams_dist), 100*calls_in_Ngrams/N),
     "\n\tin sequences ( 3 - 7 ): {:.1f}%".format(sum(100*calls_in_Ngrams[2:7]/N_grouped)))


# In[37]:


def test_ngram_stats():
    assert(N_grouped == 5594)
    assert(N == 7680)
    #assert(Ngrams_dist == [122, 143, 30, 12, 3] )
    np.testing.assert_almost_equal(sum(100*calls_in_Ngrams[2:7]/N_grouped), 40.4, decimal=2)


# # Bigrams

# ## Define sequences iteratively for combinations of parameters (sex, predator) 
# 
# ### Settings

# In[42]:


## bigrams settings
Dt = 0.5
Dtint = (None, Dt)
#timeLabel = 'ict_end_start'
#callLabel = 'note'
minCalls = 0
minNumBigrams = 0

## randomisations test
Nsh = 1000
pc = 0.05

## labels for the calls
calls = [1.0, 2.0, 3.0, 4.0]
        #if item[1] > minCalls]
sampsLi = calls[:] + ['_end'] #None #[ 'A', 'B', 'C', 'E', '_ini','_end']
condsLi = calls[:] + ['_ini'] 
# remove sequences with cs = -1
rmNodes = list(set(df[call_label]) - set(calls) ) + [-1.] # nodes to remove from network
#
minBigrams = 3
### define sequence object 
# sequences
sequencesList0 = aa.dfDict2listOfSeqs(tape_df, Dt=Dtint, l=call_label, time_param=time_param)
# filter out isolated calls (sequences of size one)
sequencesList = [l for l in sequencesList0 if len(l)>1]
# object
seqO = myst.sequenceBigrams(sequencesList)


# In[52]:


def test_seqO():
    assert((len(seqO.seqOfSeqs)==2027))
    np.testing.assert_array_almost_equal(seqO.callCounts[2.0], 345)

