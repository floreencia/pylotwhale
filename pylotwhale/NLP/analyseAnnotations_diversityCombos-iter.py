#!/usr/bin/python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pandas as pd

from collections import Counter
from analyseAnnotations_diversityCombos import bigramPlots

### SETTINGS
oFigDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/images/test'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/data/not_curated/images'
#'/home/florencia/whales/vocalSequences/whales/NPW-groupB/images'
#'/home/florencia/profesjonell/bioacoustics/christian/data/images/birds'
cfile = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/groupB_tapes_ict.csv'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/groupB_0111_001_ict.csv'
#
#'/home/florencia/whales/pilot-whales/NPW/sequences/data/WAV_0111_001-48kHz-predictions_svc-prototypesNoiseFiltered.csv'
#'/home/florencia/whales/pilot-whales/NPW/sequences/data/-f111-1_.csv'
bN = os.path.splitext(os.path.basename(cfile))[0]

#'/home/florencia/whales/pilot-whales/NPW/sequences/data/WAV_0111_001-48kHz-predictions_svc-prototypesNoiseFiltered.csv'
df0 = pd.read_csv(cfile)#, sep='\t')#, names=header)
Dt = 0.3
Dtint = (None, Dt)
timeLabel = 'ict_end_start'
callLabel='call'
subsetLabel = 'tape'
labelList = [item[0] for item in sorted(Counter(df0[subsetLabel]).items(),
                                 key = lambda x : x[1], reverse=True)]#[:10] #[bN]
minCalls = 0
minNumBigrams = 0

Nsh = 100

### create dirs
try: 
    os.mkdir(os.path.join(oFigDir))
except OSError: 
    pass

for dirName in ['bigrams', "networks", "calls", "times"]:
    try:
        os.mkdir(os.path.join(oFigDir, dirName))
    except OSError: pass


for l in labelList:
    df = df0[df0[subsetLabel] == l].reset_index(drop=True)
    bigramPlots(df, l, callLabel, timeLabel, Dtint, oFigDir, minCalls=0, 
                minCounts_ictHist=10, Nbins_ictHist=50)
    