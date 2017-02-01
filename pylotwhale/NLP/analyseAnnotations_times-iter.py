#!/usr/bin/python
from __future__ import print_function, division
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import argparse
import os
import sys

from collections import Counter
import pandas as pd

from analyseAnnotations_times import timing_plots

matplotlib.rcParams.update({'font.size': 18})

### SETTINGS
subsetLabel = 'tape'
callLabel = 'call'  # 'note'
t_interval = 10  # calling rate
timeLabel = 'ict_end_start'

oFigDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/images/times/tapes/trash'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/images/times'
cfile = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/groupB_tapes_ict.csv'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/groupB_0111_001_ict.csv'
#

statusFile = os.path.join(oFigDir, 'status.txt')
df0 = pd.read_csv(cfile)
#call_label = 'note'
#Dt = 0.8
labelList = [item[0] for item in sorted(Counter(df0[subsetLabel]).items(),
                                 key = lambda x : x[1], reverse=True)]#[:10]
#minCalls = 5
#minNumBigrams = 5


### create dirs
#for dirName in ["times"]:
try: os.mkdir(os.path.join(oFigDir))
except OSError: pass


for l in labelList:
    df = df0[df0[subsetLabel] == l].reset_index(drop=True)
    timing_plots(df, l, timeLabel, callLabel, t_interval, oFigDir)
