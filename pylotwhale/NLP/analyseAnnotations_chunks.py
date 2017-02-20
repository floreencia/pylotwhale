#!/usr/bin/python
from __future__ import print_function, division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from scipy.integrate import simps
from collections import Counter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.colors as colors

import pylotwhale.utils.plotTools as pT
import pylotwhale.utils.dataTools as daT

import pylotwhale.NLP.annotations_analyser as aa


oFigDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/images/chunks'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/images/chunks'
#
cfile = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/groupB_tapes_ict.csv'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/groupB_0111_001_ict.csv'

matplotlib.rcParams.update({'font.size': 14})

### SETTINGS
subsetLabel ='tape'
timeLabel = 'ict_end_start'
callLabel = 'call'  #'note'
t0 = 0
tf = 2
n_time_steps = 300
Dtvec = np.linspace(t0, tf, n_time_steps)
ixtimesteps = np.arange(len(Dtvec))[:200:20]  # select the time intervals

df0 = pd.read_csv(cfile)

labelList = [item[0] for item in sorted(Counter(df0[subsetLabel]).items(),
                                 key = lambda x : x[1], reverse=True)]#[:10]

### create dirs
try: os.mkdir(os.path.join(oFigDir))
except OSError: pass

def chunkplots(df, l, oFigDir,
               timeLabel=timeLabel, callLabel=callLabel,
               Dtvec=Dtvec, ixtimesteps=ixtimesteps):  #  ["all"]:#

    tapesdf = daT.dictOfGroupedDataFrames(df)

    ####### ngrams distribution
    ngramDist_Dt = aa.Ngrams_distributionDt_ndarray(tapesdf, Dtvec,
                                                    seqLabel=callLabel,
                                                    time_param=timeLabel)

    #### plot : n grmas DISTRIBUTION AND INV CUMSUM
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))

    color = iter(plt.cm.hsv(np.linspace(0.3, 1, len(ixtimesteps))))

    for i in ixtimesteps:
        c = next(color)
        ngramDist_Dt_norm = ngramDist_Dt[i, :]/np.sum(ngramDist_Dt[i, :])
        cusu = np.cumsum(ngramDist_Dt_norm[::-1])[::-1]
        ax1.plot(np.arange(1, len(cusu)+1), cusu,
                 label="{:3.2} s".format(Dtvec[i]), c=c, lw=4, alpha=0.6)
        ax2.plot(np.arange(1, len(ngramDist_Dt[i, :]) + 1), ngramDist_Dt[i, :],
                 label="{:3.2} s".format(Dtvec[i]), c=c, lw=4, alpha=0.6)

    ax1.set_title('{}'.format(l))
    ax1.set_ylabel(r"$\pi$ (k')")
    ax2.set_ylabel(r"#sequences (k)")
    #ax2.set_title('# seqs of size k')
    for cax in [ax1, ax2]:
        cax.set_xlim(0, 10)
        #cax.set_ylim(0,20)
        cax.legend()
        cax.set_xlabel('k')

    oFig = os.path.join(oFigDir,
                        '{}-ngramDist_invCumSumNgrams.png'.format(l))#, int(1000*Dtvec[ixtimesteps[0]]), int(1000*Dtvec[ixtimesteps[-1]])))
    fig1.savefig(oFig)
    oFig = os.path.join(oFigDir,
                        '{}-ngramDistCounts.png'.format(l)) #, int(1000*Dtvec[ixtimesteps[0]]), int(1000*Dtvec[ixtimesteps[-1]])))
    fig2.savefig(oFig)

    print(oFig)
    
    ### calls in ngrams of size k
    calls_in_ngramDist_Dt = aa.NgramsDist2callsInNgramsDist(ngramDist_Dt)
    Ncalls = np.sum(calls_in_ngramDist_Dt[0])
    
    # plot 
    fig1, ax1 = plt.subplots(1,1, figsize=(8,5))
    fig2, ax2 = plt.subplots(1,1, figsize=(8,5))    
    ixtimesteps = np.arange(len(Dtvec))[:200:20] # select the time intervals
    
    color=iter(plt.cm.hsv(np.linspace(0.2,1, len(ixtimesteps))))
    
    for i in ixtimesteps:
        c=next(color)
        P_calls_ngram = calls_in_ngramDist_Dt[i, :]/Ncalls
        cusu = np.cumsum(P_calls_ngram[::-1])[::-1]
        ax1.plot(np.arange(1, len(cusu)+1), cusu, 
                 label ="{:3.2} s".format(Dtvec[i]), c=c, lw=4, alpha=0.6)
        ax2.plot(np.arange(1, len(P_calls_ngram)+1), P_calls_ngram, 
                 label ="{:3.2} s".format(Dtvec[i]), c=c, lw=4, alpha=0.6 )
    
    #ax1.set_title('inv cum')
    ax1.set_ylabel("p(k')")
    ax2.set_ylabel("p(k)")
    for cax in [ax1, ax2]:   
        cax.set_xlim(0,10)   
        cax.set_title('{}'.format(l))
        #cax.set_ylim(0,20)
        cax.legend()            
        cax.set_xlabel("k")
        
    oFig = os.path.join(oFigDir, 
                        '{}-calls_in_ngramDist_invCumSumNgrams.png'.format(l))#, int(1000*Dtvec[ixtimesteps[0]]), int(1000*Dtvec[ixtimesteps[-1]])))
    fig1.savefig(oFig)    
    
    oFig = os.path.join(oFigDir, 
                        '{}-calls_in_ngramDist.png'.format(l))#, int(1000*Dtvec[ixtimesteps[0]]), int(1000*Dtvec[ixtimesteps[-1]])))
    fig2.savefig(oFig)
    print(oFig)    

    ## density plots chunks vd tau
    Tf = Dtvec[-1]; T0 = Dtvec[0]
    nT, nN = np.shape(ngramDist_Dt)
    n_ticks = 8
    yT0 = np.linspace(0, nT, n_ticks)
    maxN=10
    #xt = np.arange(1, maxN+1)
    yt = (yT0, ['{:2.2}'.format(item) for item in np.linspace(T0, Tf, len(yT0))])

    ## ngrams dist
    vmax = int(np.max(ngramDist_Dt[:, 1:]))
    M = ngramDist_Dt
    fig, ax = pT.fancyClrBarPl(np.hstack((np.zeros((len(M),1)), M)), vmax, 1, 
                               maxN=maxN, #figsize=(5,5),
                               clrBarGaps=20, cmap=plt.cm.viridis_r, 
                               yTicks=yt, yL=r'$\tau \, [s]$', xL = 'k' )

    oFig = os.path.join(oFigDir, '{}-numNgrams_vs_Dt{}-{}.png'.format(l, T0, Tf) )                               
    fig.savefig(oFig)

    ## calls
    vmax = int(np.max(calls_in_ngramDist_Dt[:, 1:]))
    M = calls_in_ngramDist_Dt
    fig, ax = pT.fancyClrBarPl(np.hstack((np.zeros((len(M),1)), M)), vmax, 1, 
                               maxN=maxN, #figsize=(5,5),
                               clrBarGaps=20, cmap=plt.cm.viridis_r, 
                               yTicks=yt, yL=r'$\tau  \, [s] $', xL = 'k' )

    oFig = os.path.join(oFigDir, '{}-callinNgrams_vs_Dt{}-{}.png'.format(l, T0, Tf) )                               
    fig.savefig(oFig)

    #### cosine similarity of the chunk structure at selected taus
    vmin = -1e-14
    vmax = 1.0
    Dtvec_ix = np.arange(len(Dtvec))[::20]
    X = calls_in_ngramDist_Dt[Dtvec_ix]
    y = ["{:.2f}".format(item) for item in Dtvec[Dtvec_ix] ]
    dist = 1 - cosine_similarity(X)
    oFig = os.path.join(oFigDir, '{}-cosSim-chunk-structure_Dt.png'.format(l))#, y[0], y[-1]))
    pT.plImshowLabels(dist, y, y, clrMap = plt.cm.viridis, figsize=(5,5),
                      norm=colors.Normalize(vmin=vmin, vmax=vmax, clip = False),
                      plTitle='{}'.format(l),  outFig=oFig)

    #### cosine similarity of the cum sum of the chunk structure at selected taus
    Dtvec_ix = np.arange(len(Dtvec))[::20]
    cusu = np.cumsum(X[:,::-1]/Ncalls, axis=1)[:,::-1]
    y = ["{:.2f}".format(item) for item in Dtvec[Dtvec_ix] ]
    dist = 1 - cosine_similarity(cusu)
    oFig = os.path.join(oFigDir, '{}-cosSim-cusu-chunk-structure_Dt.png'.format(l))#, y[0], y[-1]))
    pT.plImshowLabels(dist, y, y, clrMap = plt.cm.viridis, figsize=(5,5),
                      norm=colors.Normalize(vmin=vmin, vmax=vmax, clip = False),
                      plTitle='{}'.format(l),  outFig=oFig)

    ### integral: P(k')
    n = 5
    A = []
    Asu = []
    tvec_ix = np.arange(len(Dtvec))[:] #ixtimesteps: #
    for i in tvec_ix: 
        h = np.cumsum(calls_in_ngramDist_Dt[i, ::-1]/Ncalls)[::-1]
        #a = simps(cusu[i])
        #print("{:.2f} {:.2f}".format(Dtvec[i], a))
        Asu.append(np.sum(h[:n]))
        A.append(simps(h[:n]))
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    oFig = os.path.join(oFigDir, '{}-areaSum_Dt_n{}=Dt{:.1f}-{:.1f}.png'.format(l, n, Dtvec[tvec_ix[0]], Dtvec[tvec_ix[-1]]))#, y[0], y[-1]))
    ax1.plot(Dtvec[tvec_ix], Asu, 'bo')
    ax1.set_xlabel(r'$ \tau \, [s]$')
    ax1.set_ylabel(r'$ A ( \tau )$')
    fig1.savefig(oFig, bbox_inches='tight')

    oFig = os.path.join(oFigDir, '{}-areaInt_Dt_n{}.png'.format(l, n))#, y[0], y[-1]))    
    ax2.plot(Dtvec[tvec_ix], A, 'bo')
    ax2.set_xlabel(r'$ \tau \, [s]$')
    ax2.set_ylabel(r'$A ( \tau $)')
    fig2.savefig(oFig, bbox_inches='tight')



if __name__ == '__main__':
  chunkplots(df0, 'all')