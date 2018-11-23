
# coding: utf-8

# ## Alarm calls from Vervet monkeys

# In[3]:


from __future__ import print_function

import sys, os

from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels as sm


# In[4]:


import pylotwhale.NLP.myStatistics_beta as myst


# ## Load data
# 
# Indices from session 3 were correctred so that they match with those of session 2
# 
# zcompiled_session3full.csv --> zcompiled_session3full-bird_indices_like_in_s2.csv
# 
# <!---
# df3 = pd.read_csv(data3, names = ['t0', 'dt', 'b0'])
# b23mapping = lambda x: 3-x
# df3['b'] = df3['b0'].apply(b23mapping)
# df3.head()
# 
# oF = '/home/florencia/profesjonell/bioacoustics/noriega2017guided/data/zcompiled_session3full-bird_indices_like_in_s2.csv'
# df3[['t0','dt','b']].to_csv(oF, index=False)
# 
# -->

# In[5]:


annsDir = '/home/florencia/profesjonell/bioacoustics/Kurt/green/data/annotations/'


# In[6]:


annfiles = [item for item in os.listdir(annsDir) if item.endswith('.txt')]

files_df =  pd.DataFrame(annfiles, columns=['file_name'])

files_df['predator'] = [ fi.split('_')[0] for fi in files_df['file_name']]
files_df['monkey'] = [ fi.split('_')[-1][:-4] for fi in files_df['file_name']]

files_df['annSections'] = [ len(open(os.path.join(annsDir, annFi), 'r').readlines())
                             for annFi in files_df['file_name']]

files_df['tape'] = [ annFi[:-4] for annFi in files_df['file_name']]

files_df['sex'] = [ annFi[0] for annFi in files_df['monkey']]


# In[12]:


def test_data():
    assert(len(files_df) == 45)
    assert(set(files_df.columns) == set(['monkey', 'file_name', 'predator', 'sex', 'tape', 'annSections']))

test_data


# # Create the annotations dataframe

# In[13]:


# number of calls
call_df = pd.DataFrame(columns=['call', 't0', 'tf', 'ici0', 'ioi', 'cl', 'file_name', 'tag', 'Dtag', 'ici']) #'tape' , 'bird', 'area',

annCalls = []
t0 = []
tf = []
cl = []
ici = []
ioi = []

rest_df=[]

for i in range(len(files_df)):
    annFi = files_df['file_name'].ix[i]
    A = np.loadtxt(os.path.join(annsDir, annFi), dtype='S', ndmin=2)
    _t0 = A[:,0].astype('float')
    _tf = A[:,1].astype('float')
    t0.extend( _t0 )
    tf.extend( _tf )
    cl.extend( _tf - _t0 )
    
    ioi.extend(_t0[1:] - _t0[:-1])
    ioi.extend([np.NaN])
    
    ici.extend(_t0[1:] - _tf[:-1])
    ici.extend([np.NaN])
    
    annCalls.extend( A[:,2] )
    rest_df.extend([files_df['file_name'].ix[i]]*len(A)) 

call_df['t0'] = t0
call_df['tf'] = tf
call_df['cl'] = cl
call_df['ici0'] = ici
call_df['ioi'] = ioi
call_df['call'] = annCalls
call_df['file_name'] = rest_df

### pupulate table with ann file info
df = pd.merge(call_df, files_df[['file_name', 'tape' , 'monkey', 'predator', 'sex']], 
              how = 'left', on = ['file_name'])

print(len(call_df), len(df))
#df[df['file_name'] == 'CA-117.txt']


# In[14]:


def test_merged_df():
    assert(len(call_df) == 7718)
    assert(len(df) == 7718)
   
test_merged_df()

