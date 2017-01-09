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
import pylotwhale.NLP.sequencesO_beta as seqT
import pylotwhale.NLP.myStatistics_beta as mysts

import pylotwhale.NLP.ngramO_beta as ngr


### SETTINGS
oFigDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/images'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/data/not_curated/images'
#'/home/florencia/whales/vocalSequences/whales/NPW-groupB/images'
#'/home/florencia/profesjonell/bioacoustics/christian/data/images/birds'
cfile = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/groupB_0111_001_ict.csv'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/data/groupB_tapes_ict.csv'
#'/home/florencia/whales/pilot-whales/NPW/sequences/data/WAV_0111_001-48kHz-predictions_svc-prototypesNoiseFiltered.csv'
#'/home/florencia/whales/pilot-whales/NPW/sequences/data/-f111-1_.csv'
bN = os.path.splitext(os.path.basename(cfile))[0]

#'/home/florencia/whales/pilot-whales/NPW/sequences/data/WAV_0111_001-48kHz-predictions_svc-prototypesNoiseFiltered.csv'
df0 = pd.read_csv(cfile)#, sep='\t')#, names=header)
Dt = 100
Dtint = (None, Dt)
subsetLabel = 'tape'
labelList = [item[0] for item in sorted(Counter(df0[subsetLabel]).items(),
                                 key = lambda x : x[1], reverse=True)]#[:10] #[bN]
minCalls = 0
minCalls_display = 5

minNumBigrams = 0
minCalls_H0test = 5 # limit to apply diff proportoins test
pcValue = 0.9
Nsh = 50 # number of shufflings


### create dirs
try: 
    os.mkdir(os.path.join(oFigDir))
except OSError: 
    pass


for l in labelList:
    df = df0[df0[subsetLabel] == l].reset_index(drop=True)
    
    ## define the calls plotting order
    calls = [item[0] for item in 
        sorted(Counter(df.call).items(), key = lambda x : x[1], reverse=True) ]
    # if item[1] > minCalls]
    samplesLi = calls[:] + ['_end'] #None #[ 'A', 'B', 'C', 'E', '_ini','_end']
    condsLi = calls[:] + ['_ini'] 
    rmNodes = list(set(df.call) - set(calls) ) # nodes to remove from network
    
    ## separate data by tape
    tapedf = { str(l) : df}
    
     ## define the sequences and cut bigrams
    sequences = aa.seqsLi2iniEndSeq( aa.dfDict2listOfSeqs(tapedf, Dt=Dtint, l='call',
                                                          time_param='ict_end_start'))
    my_bigrams = nltk.bigrams(sequences)

    ## count bigrams
    cfd = ngr.bigrams2Dict(my_bigrams)
    M, samps, conds = ngr.bigramsDict2countsMatrix(cfd, condsLi, samplesLi)

    ## shuffle calls
    cfd_nsh = nltk.ConditionalFreqDist()
    for i in range(Nsh):  ## shuffle iter
        for t in tapedf.keys(): # for each tape
            thisdf = tapedf[t]
            sh_df = seqT.shuffleSeries(thisdf, shuffleCol='call') # shuffle the calls
            sequences = aa.seqsLi2iniEndSeq( aa.df2listOfSeqs(sh_df, Dt=Dtint, l='call',
                                                              time_param='ict_end_start'))
            my_bigrams = nltk.bigrams(sequences)
            cfd_nsh += ngr.bigrams2Dict(my_bigrams) # count bigrams
            
    Msh, samps, conds = ngr.bigramsDict2countsMatrix(cfd_nsh, condsLi, samplesLi) # matrix

    ### H0: obs = random (diff of proportions test)
    XYtest = mysts.elementwiseDiffPropTestXY( M, Msh, minCalls_H0test, pcValue=pcValue)
    ## visualise
    Mtest, yconds, xsamps = daT.matrixSubsample(XYtest, conds, samps, condsLi, samplesLi)
    oFile = os.path.join(oFigDir, 
                         "{}_{}-H0EWdiffProportions_minCll{}-dt{}-pc{}.png".format(
                         subsetLabel, l, minCalls_display, Dt, pcValue))
    pT.plImshowLabels( Mtest, xsamps, yconds, cbarAxSize=5, cbarLim=(-1,1),
                  xLabel='$c_{i+1}$', yLabel='$c_{i}$', 
                  plTitle = " $\Delta t$={} s, p_c  = {}".format( Dt, pcValue), 
                    outFig=oFile, badClr='white' )
    plt.clf()
    
    ### compare with RANDOMISED calls
