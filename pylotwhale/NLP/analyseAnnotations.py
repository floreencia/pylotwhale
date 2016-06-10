#!/usr/bin/python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

from collections import Counter
import pandas as pd
import nltk

import pylotwhale.utils.whaleFileProcessing as fp
import pylotwhale.utils.fileCollections as fcll
import pylotwhale.utils.plotTools as pT
import pylotwhale.utils.dataTools as daT
import pylotwhale.utils.netTools as nT

import pylotwhale.NLP.annotations_analyser as aa
import pylotwhale.NLP.ngramO_beta as ngr


### SETTINGS
oFigDir = '/home/florencia/whales/pilot-whales/NPW/sequences/images/'
#'/home/florencia/profesjonell/bioacoustics/christian/data/images/birds'
cfile = '/home/florencia/whales/pilot-whales/NPW/sequences/data/WAV_0111_001-48kHz-predictions_svc-prototypesNoiseFiltered.csv'
#'/home/florencia/whales/pilot-whales/NPW/sequences/data/-f111-1_.csv'
bN = os.path.splitext(os.path.basename(cfile))[0]

#'/home/florencia/whales/pilot-whales/NPW/sequences/data/WAV_0111_001-48kHz-predictions_svc-prototypesNoiseFiltered.csv'
df0 = pd.read_csv(cfile)#, sep='\t')#, names=header)
Dt = 0.25
Dtint = (None, Dt)
subsetLabel = 'ann'
labelList = [bN]#[item[0] for item in sorted(Counter(df0[subsetLabel]).items(),
             #                    key = lambda x : x[1], reverse=True)]#[:10]
minCalls = 0
minNumBigrams = 0

### create dirs
for dirName in ['bigrams', "networks", "calls", "times"]:
    try: os.mkdir(os.path.join(oFigDir, dirName))
    except OSError: pass


for l in labelList:
    df = df0#[df0[subsetLabel] == l].reset_index(drop=True)
    
    ## define the calls plotting order
    calls = [item[0] for item in 
        sorted(Counter(df.call).items(), key = lambda x : x[1], reverse=True) 
        if item[1] > minCalls]
    samplesLi = calls[:] + ['_end'] #None #[ 'A', 'B', 'C', 'E', '_ini','_end']
    condsLi = calls[:] + ['_ini'] 
    rmNodes = list(set(df.call) - set(calls) ) # nodes to remove from network
    
    ## separate data by tape
    tapedf = { str(l) : df}
    
    ## PLOT call frequencies
    oFile = os.path.join(oFigDir, "calls", 
                         "{}_{}-callFreqs.png".format(subsetLabel, l))
    df.call.value_counts().plot(kind='bar')
    plt.savefig(oFile); plt.clf()

    ## define the sequences and cut bigrams
    sequences = aa.seqsLi2iniEndSeq( aa.dfDict2listOfSeqs(tapedf, Dt=Dtint, l='call'))
    my_bigrams = nltk.bigrams(sequences)
    ## count bigrams
    cfd = ngr.bigrams2Dict(my_bigrams)    
    
    ## PLOT counts
    M, samps, conds = ngr.bigramsDict2countsMatrix( cfd, condsLi, samplesLi)
    oFile = os.path.join(oFigDir, "bigrams", 
                         "{}_{}-bigramCounts_minCll{}-dt{}.png".format(subsetLabel, l, minCalls, Dt))
    pT.plImshowLabels( M, samps, conds, cbarLim=(1,None), cbarAxSize=5, 
                  xLabel='$c_i$', yLabel='$c_{i-1}$', 
                  plTitle = "{}, $\Delta t$={} s".format(l, Dt), outFig=oFile )
    plt.clf()
                  
    ### conditional probabilities  
    cpd = ngr.condFreqDictC2condProbDict(cfd)
    Mp, samps, conds =  ngr.condProbDict2matrix(cpd, condsLi, samplesLi) 
    ## PLOT cond probs
    oFile = os.path.join(oFigDir, "bigrams",
                         "{}_{}-bigramProbs_minCll{}-dt{}.png".format(subsetLabel, l, minCalls, Dt))
    pT.plImshowLabels( Mp, samps, conds, cbarLim=(0.001, None), cbarAxSize=5, 
                  xLabel='$c_i$', yLabel='$c_{i-1}$', 
                  plTitle = "{}, $\Delta t$={} s".format(l, Dt), outFig=oFile )
    plt.clf()

            
    ### DRAW networks
    oFile = os.path.join(oFigDir, "networks", 
                         "{}_{}-net_minCll{}-dt{}".format(subsetLabel, l, minCalls, Dt))
    nT.drawNetFrom2DimDict(cfd, oFile+'.dot', oFile+'.png', cpd, 2, rmNodes=rmNodes)
    
    ### PLOT time violins
    oFile = os.path.join(oFigDir, "times", 
                         "{}_{}-timeViolin-minBigs{}".format(subsetLabel, l, minNumBigrams))
    try:
        ngr.pl_ic_bigram_times(df, my_bigrams, oFig=oFile, minNumBigrams=minNumBigrams) 
    except IndexError:
        pass
    plt.clf()
