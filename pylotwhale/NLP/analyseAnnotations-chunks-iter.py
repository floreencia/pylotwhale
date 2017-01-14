#!/usr/bin/python
from __future__ import print_function, division
import numpy as np
import matplotlib
import os

from collections import Counter
import pandas as pd

from analyseAnnotations_chunks import chunkplots

oFigDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/images/chunks'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/images/chunks'
#
cfile = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/groupB_tapes_ict.csv'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/groupB_0111_001_ict.csv'

matplotlib.rcParams.update({'font.size': 14})

### SETTINGS
subsetLabel ='tape'
timeLabel = 'ict_end_start'
callLabel = 'call'#'note'
t0 = 0
tf = 2
n_time_steps = 300
Dtvec = np.linspace(t0, tf, n_time_steps)
ixtimesteps = np.arange(len(Dtvec))[:200:20] # select the time intervals

df0 = pd.read_csv(cfile)

labelList = [item[0] for item in sorted(Counter(df0[subsetLabel]).items(),
                                 key = lambda x : x[1], reverse=True)]#[:10]

### create dirs
try: os.mkdir(os.path.join(oFigDir))
except OSError: pass

for l in labelList: 
    df = df0[df0[subsetLabel] == l].reset_index(drop=True)

    chunkplots(df, l, 
               timeLabel=timeLabel, callLabel=callLabel,
               Dtvec=Dtvec, ixtimesteps=ixtimesteps)