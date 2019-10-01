# coding: utf-8

# ## Alarm calls from Vervet monkeys

# In[1]:


from __future__ import print_function

import sys, os

from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ## Load time segments dataframe

# In[10]:


cfile = "/home/florencia/profesjonell/bioacoustics/Kurt/green/data/vervet_annotations_df.txt"

df_time = pd.read_csv(cfile)
# df_time.head(20)
this_df = df_time[df_time["tape"] == "leo_f1"]


def test_data():
    assert len(df_time) == 7718
    assert len(this_df) == 47


test_data()


# ## Load call type data_frame (cluster_results)

# In[12]:


callTypeFile = (
    "/home/florencia/profesjonell/bioacoustics/Kurt/green/data/result_cluster_analysis.csv"
)

df_call0 = pd.read_csv(callTypeFile)
df_call = df_call0[
    ["name", "sequence_nr", "sex", "context", "calldur", "cluster_two", "cluster_four"]
]


# ### Add `tape` column to call_dataframe
# parse tape name from name

# In[13]:


tape_name = []

for i, row in df_call.iterrows():
    # pr, sx, ind = call_str.split('_')[:3]
    pr = row["context"]
    sx = row["sex"]
    ind = row["name"].split("_")[2]
    tape_name.append("{}_{}{}".format(pr.lower(), sx.lower(), int(ind)))

# uncomment to add column
df_call["tape"] = tape_name

df_call.head()


# ### Add `tag` column to call_dataframe
#

# In[14]:


tags = []
for c in df_call["name"]:
    x = c.split(".lma")[0]
    tags.append((x.split("_")[3]).lstrip("0"))

df_call["tag"] = tags

df_call.head()


# ## create data fame full annotations
#
# by merging the later two

# In[15]:


new_df = pd.merge(df_time, df_call, how="outer", on=["tape", "sex", "tag"])


# Annotations for tapes: 'sn_f6' and 'sn_m3' beacuse these monkesy only produced two vocalisations

# In[19]:


def test_mergedDF():
    x = Counter(new_df[pd.isnull(new_df["call"])]["tape"])
    assert len(x) == 2
    assert sum(x.values()) == 4
    assert set(["sn_f6", "sn_m3"])
    assert len(df_time) == 7718
    assert len(df_call) == 3591
    assert len(new_df) == 7872
