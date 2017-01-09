#!/usr/bin/python
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

from collections import Counter
import pandas as pd
#import nltk

#import pylotwhale.utils.whaleFileProcessing as fp
#import pylotwhale.utils.fileCollections as fcll
import pylotwhale.utils.plotTools as pT
import pylotwhale.utils.dataTools as daT
#import pylotwhale.utils.netTools as nT

import pylotwhale.NLP.annotations_analyser as aa
#import pylotwhale.NLP.ngramO_beta as ngr
#import pylotwhale.NLP.tempoTools as tT

oFigDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/images/chunks'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/images/chunks'
#

cfile = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/groupB_tapes_ict.csv'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/groupB_0111_001_ict.csv'

### SETTINGS
subsetLabel ='tape'
timeLabel = 'ict_end_start'
callLabel = 'call'#'note'
t0 = 0
tf = 3
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
    tapesdf = daT.dictOfGroupedDataFrames(df)


    ####### ngrams distribution
    ngramDist_Dt = aa.Ngrams_distributionDt_ndarray( tapesdf, Dtvec,
                                                    seqLabel=callLabel,
                                                    time_param=timeLabel)

    #### plot : DISTRIBUTION AND INV CUMSUM
    fig, ax = plt.subplots(1,2, figsize=(16,5))

    color=iter(plt.cm.hsv(np.linspace(0.2,1, len(ixtimesteps))))

    for i in ixtimesteps:
        c=next(color)
        ngramDist_Dt_norm = ngramDist_Dt[i, :]/np.sum(ngramDist_Dt[i, :])
        cusu = np.cumsum(ngramDist_Dt_norm[::-1])[::-1]
        ax[0].plot(cusu, label ="{:3.2} s".format(Dtvec[i]), c=c, lw=4, alpha=0.6)
        ax[1].plot(ngramDist_Dt_norm, label ="{:3.2} s".format(Dtvec[i]), c=c, lw=4, alpha=0.6 )

    ax[0].set_title('inv cum')
    ax[1].set_title('# seqs of size k')
    for cax in ax:
        cax.set_xlim(0,10)
        #cax.set_ylim(0,20)
        cax.legend() 

        cax.set_ylabel('p(k)')
        cax.set_xlabel('k')
        
    oFig = os.path.join(oFigDir, '{}-ngramDist_invCumSumNgrams-Dt{:d}_{:d}ms.png'
                            .format(l, int(1000*Dtvec[ixtimesteps[0]]), int(1000*Dtvec[ixtimesteps[-1]])))
    fig.savefig(oFig)
    print(oFig)
    
    ### calls in ngrams of size k
    calls_in_ngramDist_Dt = aa.NgramsDist2callsInNgramsDist(ngramDist_Dt)
    Ncalls = np.sum(calls_in_ngramDist_Dt[0])
    
    # plot 
    fig, ax = plt.subplots(1,2, figsize=(16,5))
    ixtimesteps = np.arange(len(Dtvec))[:200:20] # select the time intervals
    
    color=iter(plt.cm.hsv(np.linspace(0.2,1, len(ixtimesteps))))
    
    for i in ixtimesteps:
        c=next(color)
        P_calls_ngram = calls_in_ngramDist_Dt[i, :]/Ncalls
        cusu = np.cumsum(P_calls_ngram[::-1])[::-1]
        ax[0].plot(cusu, label ="{:3.2} s".format(Dtvec[i]), c=c, lw=4, alpha=0.6)
        ax[1].plot(P_calls_ngram, label ="{:3.2} s".format(Dtvec[i]), c=c, lw=4, alpha=0.6 )
    
    ax[0].set_title('inv cum')
    ax[1].set_title('# calls in a sequence of size k')
    for cax in ax:   
        cax.set_xlim(0,10)    
        #cax.set_ylim(0,20)
        cax.legend()    
        
        cax.set_ylabel("p(k')")
        cax.set_xlabel("k")
        
    oFig = os.path.join(oFigDir, '{}-calls_in_ngramDist_invCumSumNgrams-Dt{:d}_{:d}ms.png'
                            .format(l, int(1000*Dtvec[ixtimesteps[0]]), int(1000*Dtvec[ixtimesteps[-1]])))
    fig.savefig(oFig)
    print(oFig)    

    ## density plots
    Tf = Dtvec[-1]; T0 = Dtvec[0]
    nT, nN = np.shape(ngramDist_Dt)
    n_ticks = 8
    yT0 = np.linspace(0, nT, n_ticks)
    yt = (yT0, ['{:2.2}'.format(item) for item in np.linspace(T0, Tf, len(yT0))])

    ## ngrams dist
    vmax = int(np.max(ngramDist_Dt[:, 2:]))
    fig, ax = pT.fancyClrBarPl(ngramDist_Dt, vmax, 1, maxN=12,
                               clrBarGaps=20, cmap=plt.cm.viridis_r, yTicks=yt, yL='$\Delta t$' )

    oFig = os.path.join(oFigDir, '{}-numNgrams_vs_Dt{}-{}.png'.format(l, T0, Tf) )                               
    fig.savefig(oFig)

    ## calls
    vmax = int(np.max(calls_in_ngramDist_Dt[:, 2:]))
    fig, ax = pT.fancyClrBarPl(calls_in_ngramDist_Dt, vmax, 1, maxN=12,
                               clrBarGaps=20, cmap=plt.cm.viridis_r, yTicks=yt, yL='$\Delta t$' )
    
    oFig = os.path.join(oFigDir, '{}-callinNgrams_vs_Dt{}-{}.png'.format(l, T0, Tf) )                               
    fig.savefig(oFig)
    


