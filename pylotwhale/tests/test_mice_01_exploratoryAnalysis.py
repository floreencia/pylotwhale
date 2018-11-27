
# coding: utf-8

# In[7]:


from __future__ import print_function, division
import sys
import os

import numpy as np
import pandas as pd
from collections import Counter, defaultdict


# In[8]:


import pylotwhale.utils.dataTools as daT
import pylotwhale.NLP.myStatistics_beta as mysts


# # Load csv with sequences

# In[9]:


cfile = '/home/florencia/profesjonell/bioacoustics/Kurt/mice/data/sequenceFiles_df.txt'

df0 = pd.read_csv(cfile)


# In[13]:


name_df = daT.dictOfGroupedDataFrames(df0, groupingKey='name')
print('TEST', len(name_df))

def test_data():
    assert(len(df0) == 25282)
    assert(len(name_df) == 67)
    assert({1,2,3} == set(df0['genecode']))


# In[14]:


len(name_df)


# # Call from al mice

# define sequences

# In[18]:


name_df = daT.dictOfGroupedDataFrames(df0, groupingKey='name')

# gather the sequences of all mice in a strain
seqsL = []
for n, thisdf in name_df.items():
    x = thisdf['call'].values
    seqsL.append(x)


# In[19]:


Seqs = mysts.sequenceBigrams(seqsL)

def test_bigrams():
    assert(len(Seqs.seqOfSeqs) == 67)
    assert(len(Seqs.bigrams) == 25415)
    assert(Seqs.callCounts['10'] == 4993)


# List of soreted calls, conditions and samples

# # Subset the dataset to a particular strain
# 

# Subseting settings

# In[21]:


gc = 3
cs = {'genecode': gc}


# take a geneticcode subset 

# In[31]:


df_temp = []
lastdf = df0

for k, v in cs.items():
    print(k, v)
    df_temp.append(lastdf[lastdf[k] == v])
    lastdf = df_temp[-1]

df_gc = df_temp[-1].reset_index(drop=True)  #[~df0.isnull().any(axis=1)]

def test_subset():
    assert(df_gc['genecode'].values[0] == gc)
    assert(len(df_gc) == 8446)


# In[29]:


# split data by mouse
name_df_gc = daT.dictOfGroupedDataFrames(df_gc, groupingKey='name')

# gather the sequences of all mice in a strain
seqsL_gc = []
for n, thisdf in name_df_gc.items():
    x = thisdf['call'].values
    seqsL_gc.append(x)
    
Seqs_gc = mysts.sequenceBigrams(seqsL_gc)

def test_bigrams_gc():
    assert(len(Seqs_gc.seqOfSeqs) == 23)
    assert(len(Seqs_gc.bigrams) == 8491)
    assert(Seqs_gc.callCounts['10'] == 1434)

