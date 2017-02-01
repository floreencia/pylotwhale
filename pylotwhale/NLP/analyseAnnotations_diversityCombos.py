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
import pylotwhale.NLP.tempoTools as tT



### SETTINGS
oFigDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/images/test/'
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
Dt = 0.3
Dtint = (None, Dt)
timeLabel = 'ict_end_start'
callLabel='call'
#subsetLabel = 'tape'
#labelList = [item[0] for item in sorted(Counter(df0[subsetLabel]).items(),
#                                 key = lambda x : x[1], reverse=True)]#[:10] #[bN]
minCalls = 0
minNumBigrams = 0

### create dirs
try:
    os.mkdir(os.path.join(oFigDir))
except OSError:
    pass


for dirName in ['bigrams', "networks", "calls", "times"]:
    try:
        os.mkdir(os.path.join(oFigDir, dirName))
    except OSError:
        pass


def bigramPlots(df, l, callLabel, timeLabel, Dtint, oFigDir, minCalls=0, 
                minCounts_ictHist=10, Nbins_ictHist=50):
    #df = df0[df0[subsetLabel] == l].reset_index(drop=True)
    
    ## define the calls plotting order
    calls = [item[0] for item in 
        sorted(Counter(df[callLabel]).items(), key = lambda x : x[1], reverse=True) ]
        #if item[1] > minCalls]
    samplesLi = calls[:] + ['_end'] #None #[ 'A', 'B', 'C', 'E', '_ini','_end']
    condsLi = calls[:] + ['_ini'] 
    rmNodes = list(set(df.call) - set(calls) ) # nodes to remove from network
    
    ## separate data by tape
    df_dict = { str(l) : df}
    
    #### PLOT call frequencies
    oFile = os.path.join(oFigDir, "calls", 
                         "{}-callFreqs.png".format(l))
    df[callLabel].value_counts().plot(kind='bar', title='%s'%l)
    plt.savefig(oFile); plt.clf()

    ## define the sequences and cut bigrams
    sequences = aa.seqsLi2iniEndSeq( aa.dfDict2listOfSeqs(df_dict, Dt=Dtint, l=callLabel, 
                                                          time_param=timeLabel))
    my_bigrams = list(nltk.bigrams(sequences))

    ## count bigrams
    cfd = ngr.bigrams2Dict(my_bigrams)    
    
    #### PLOT bigram counts
    M, samps, conds = ngr.bigramsDict2countsMatrix( cfd, condsLi, samplesLi)
    oFile = os.path.join(oFigDir, "bigrams", 
                         "{}-bigramCounts_minCll{}-dt{}.png".format(l, minCalls, Dt))
    pT.plImshowLabels( M, samps, conds, cbarLim=(1,None), cbarAxSize=5, 
                      xLabel='$c_i$', yLabel='$c_{i-1}$', 
                      plTitle = "{}, $\Delta t$={} s".format(l, Dt), outFig=oFile )
    plt.clf()
                  
    ### conditional probabilities  
    cpd = ngr.condFreqDictC2condProbDict(cfd)
    Mp, samps, conds =  ngr.condProbDict2matrix(cpd, condsLi, samplesLi) 
    #### PLOT bigram cond probs
    oFile = os.path.join(oFigDir, "bigrams",
                         "{}-bigramProbs_minCll{}-dt{}.png".format(l, minCalls, Dt))
    fig, ax = pT.plImshowLabels( Mp, samps, conds, cbarLim=(0.001, None), cbarAxSize=5, 
                  xLabel='$c_i$', yLabel='$c_{i-1}$', 
                  plTitle = "{}, $\Delta t$={} s".format(l, Dt))
                  
    fig, ax = pT.display_numbers(fig, ax, M, 12, condition=lambda x:x>0)
    fig.savefig(oFile)
    plt.clf()

    #### DRAW networks
    oFile = os.path.join(oFigDir, "networks", 
                         "{}-net_minCll{}-dt{}".format(l, minCalls, Dt))
    nT.drawNetFrom2DimDict(cfd, oFile+'.dot', oFile+'.png', cpd, 2, rmNodes=rmNodes)


    #### network properties
    G = nT.cfd2nxDiGraph(cfd)
    oFile = os.path.join(oFigDir, "networks",
                         "{}-betweenness_centrality_minCll{}-dt{}.png".format(l, minCalls, Dt))
    nT.pl_betweenness_centrality(G, oFig=oFile)
    oFile = os.path.join(oFigDir, "networks", 
                         "{}-degree_centrality_minCll{}-dt{}.png".format(l, minCalls, Dt))
    nT.pl_degree_centrality(G, oFig=oFile)

    """
    #### PLOT coloured ict
    ## histogram range
    ixc_min = 0.01
    ixc_max = 1
    rg = (ixc_min, ixc_max)

    sequences = aa.seqsLi2iniEndSeq(aa.df2listOfSeqs(df, l=callLabel,
                                                     Dt=(ixc_min, ixc_max),
                                                     time_param=timeLabel))
    topBigrams0 = daT.returnSortingKeys(Counter(my_bigrams))
                                       # minCounts=minCounts_ictHist)
    topBigrams = daT.removeElementWith(topBigrams0, l_ignore=['_ini', '_end'])

    ## ict by bigram
    tapedf = daT.dictOfGroupedDataFrames(df)  # dictionary of ict: ict_XY
    ict_di = ngr.dfDict2dictOfBigramIcTimes(tapedf, topBigrams, label=callLabel,
                                            ict_label=timeLabel)

    ## bigrams you want to colour, all is to much => select a subset
    #minBigr = 4
    bigrs = ngr.selectBigramsAround_dt(ict_di, (ixc_min, ixc_max), 4)
    print("\nTEST", topBigrams0, bigrs, ict_di)
    oFile = os.path.join(oFigDir, "times",
                         "{}-ictHist_minCll{}.png".format(l, minCalls))

    tT.pl_ictHist_coloured(df[timeLabel], ict_di, bigrs=bigrs, Nbins=Nbins_ictHist,
                           rg=rg, oFig=oFile)
    """

    """
    ### PLOT time violins
    oFile = os.path.join(oFigDir, "times", 
                         "{}-timeViolin-minBigs{}".format(l, minNumBigrams))
    try:
        tT.pl_ic_bigram_times(df, my_bigrams, oFig=oFile, minNumBigrams=minNumBigrams) 
    except IndexError:
        pass
    plt.clf()
    """
    ### compare with RANDOMISED calls


if __name__ == '__main__':
  bigramPlots(df0, 'all', callLabel=callLabel, timeLabel=timeLabel, 
               Dtint=Dtint, oFigDir=oFigDir, minCalls=minCalls)
               
