 #!/usr/bin/python
from __future__ import print_function, division
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
#import argparse
import os
#import sys

from collections import Counter
import pandas as pd
import nltk

#import pylotwhale.utils.whaleFileProcessing as fp
#import pylotwhale.utils.fileCollections as fcll
#import pylotwhale.utils.plotTools as pT
import pylotwhale.utils.dataTools as daT
#import pylotwhale.utils.netTools as nT

import pylotwhale.NLP.annotations_analyser as aa
import pylotwhale.NLP.ngramO_beta as ngr
import pylotwhale.NLP.tempoTools as tT

matplotlib.rcParams.update({'font.size': 22})

### SETTINGS
#subsetLabel ='tape'
callLabel = 'call'#'note'
t_interval = 10 #calling rate
timeLabel = 'ict_end_start'

oFigDir = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/images/times/tapes'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/images/times'
cfile = '/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/not_curated/groupB/groupB_tapes_ict.csv'
#'/home/florencia/profesjonell/bioacoustics/heike/NPW/vocalSequences/NPW/data/curated/groupB_0111_001_ict.csv'
#

#statusFile = os.path.join(oFigDir, 'status.txt')
df0 = pd.read_csv(cfile)
#call_label = 'note'
#Dt = 0.8
#minCalls = 5
#minNumBigrams = 5
"""
with open(statusFile, 'a') as out_file:
    out_file.write('#{}\n#{}'.format(cfile, subsetLabel))
"""
### create dirs
#for dirName in ["times"]:
try: os.mkdir(os.path.join(oFigDir))
except OSError: pass



def timing_plots(df, l, timeLabel, callLabel, t_interval, oFigDir, NbinsFrac=0.2):

    #df = df0[df0[subsetLabel] == l].reset_index(drop=True)
    ict = df[timeLabel].values  # df0.ict.values
    ict = ict[~np.isnan(ict)]
    N_tapeSamples = len(ict)

    #### plot ict histogram
    ## log scale
    Nbins = int(NbinsFrac*N_tapeSamples)
    pltitle = None#"tape: {}".format(l)  # in ({}, {}) s".format(rg[0], rg[1]))
    oFig = os.path.join(oFigDir, 'log10ict-{}hist-{}.png'.format(Nbins, l))
    tT.y_histogram(np.log10(ict), range=None, Nbins=Nbins,
                   xl=r'$\log _{10}( \tau _{ICI})$', max_xticks = 4,
                   oFig=oFig, plTitle=pltitle)
    plt.close()
    ## in range
    oFig = os.path.join(oFigDir, 'ict-{}hist-{}.png'.format(Nbins, l))
    rg = (np.nanmin(ict), ict_max)
    tT.y_histogram(ict, range=rg, Nbins=Nbins, xl=r'$\tau _{ICI}, [s]$', oFig=oFig, plTitle=pltitle)
    plt.close()

    #### fraction of calls with ict smaller than x
    ict_sorted = np.sort(ict)
    x = ict_sorted[:-1]
    y = np.arange(len(x))/len(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, 'bo')
    ax.set_xlabel("t', [s]")
    ax.set_ylabel('fraction of calls')
    ax.set_xlim((x[0], x[int(0.75*len(x))]))
    ### ticks
    max_xticks = 5
    xloc = plt.MaxNLocator(max_xticks)
    ax.xaxis.set_major_locator(xloc)
    #ax.set_title(pltitle)
    oFig = os.path.join(oFigDir, 'ictSorted_ict0-ict3of4-{}.png'.format(l))
    fig.savefig(oFig, bbox_inches='tight')
    plt.close()


    #### plot call length histogram
    cl = df.cl.values
    cl_max = np.max(cl)
    oFig = os.path.join(oFigDir, 'cl-{}hist-{}.png'.format(Nbins, l))
    tT.y_histogram(cl, range=(0, cl_max), Nbins=Nbins, oFig=oFig,
                   xl='call length, [s]', plTitle=pltitle)
    plt.close()

    #### calling rate
    oFig = os.path.join(oFigDir, 'callingRate-Dt{}-{}.png'.format(t_interval, l))
    tT.pl_calling_rate(df, t_interval, oFig=oFig, plTitle=pltitle)
    plt.close()

    #### print stats of the current ict dist
    """
    with open(statusFile, 'a') as out_file:
        out_file.write("\n{}\n{}".format(l, df[timeLabel].describe()))
        out_file.write("\nFraction of calls with an ict<=0.2c {}\n".format(len(x[x<0.2])/len(x)))
    """
    ##### call-coloured histograms
    ## sequences
    ta=0
    Dt=20
    sequences = aa.seqsLi2iniEndSeq(aa.df2listOfSeqs(df, l=callLabel, Dt=(ta, Dt), 
                                                     time_param=timeLabel))
    ## bigrams
    my_bigrams = list(nltk.bigrams(sequences))
    topBigrams0 = daT.returnSortingKeys(Counter(my_bigrams), minCounts=1)
    topBigrams = daT.removeElementWith(topBigrams0, l_ignore=['_ini', '_end'])

    ## plot log(ict) with coloured bins for bigrams


if False:
    if N_tapeSamples > 20:
        ## coloured ict histograms
        #rg=(np.log10(0.1), np.log10(10))
        
        tapedf = daT.dictOfGroupedDataFrames(df)
        ### dictionary of ict: ict_XY
        ict_di = ngr.dfDict2dictOfBigramIcTimes(tapedf, topBigrams,
                                                ict_label=timeLabel) 
        
        ## plot ict histogram
        #log_ict_di = {XY: np.log10(ict_di[XY]) for XY in ict_di.keys()}
        fig = plt.figure()
        ax = fig.add_subplot(111)
        minBigr = 4
        ixc_min = 0.01
        ixc_max = 0.3
        rg = (ixc_min, ixc_max)
        bigrs = ngr.selectBigramsAround_dt(ict_di, (ixc_min, ixc_max), minBigr)

        ax.hist(df[timeLabel], range=rg, 
                 label='other', alpha=0.4, color='gray', bins=Nbins)#, cumulative=True)
        
        cmap = plt.cm.gist_rainbow(np.linspace(0,1,len(bigrs))) #gist_car
        ax.hist([ict_di[''.join(item)] for item in bigrs[:]], stacked=True, range=rg, 
                label=bigrs, rwidth='stepfilled', bins=Nbins, color=cmap)#, cumulative=True)
        ax.set_xlabel('ict')
        plt.legend()
        oFig = os.path.join(oFigDir, 'ict-XYhist-{}.png'.format(l)) # ict-hist-all-rg{}.png'.format('None'))#

        fig.savefig(oFig)
        plt.close()

        
        ## plot log10(ict) histogram
        log_ict_di = {XY: np.log10(ict_di[XY]) for XY in ict_di.keys()}
        fig = plt.figure()
        ax = fig.add_subplot(111)
        minBigr = 4
        ixc_min = 0.01
        ixc_max = 20
        rg = (np.log10(ixc_min), np.log10(ixc_max))
        bigrs = ngr.selectBigramsAround_dt(ict_di, (ixc_min, ixc_max), minBigr)

        ax.hist(np.log10(df.ict_end_start), range=rg,
                 label='other', alpha=0.4, color='gray', bins=Nbins)#, cumulative=True)
        
        cmap = plt.cm.gist_rainbow(np.linspace(0,1,len(bigrs))) #gist_car
        ax.hist([log_ict_di[''.join(item)] for item in bigrs[:]], stacked=True, range=rg, 
                             label=bigrs, 
                 rwidth='stepfilled', bins=Nbins, color=cmap)#, cumulative=True)
        ax.set_xlabel('$\log_{10}(ict)$')        
        plt.legend()
        oFig = os.path.join(oFigDir, 'log_ict-XYhist-{}.png'.format(l)) # ict-hist-all-rg{}.png'.format('None'))#

        fig.savefig(oFig)
        plt.close()



if __name__ == '__main__':
  timing_plots(df0, 'all', timeLabel=timeLabel, callLabel=callLabel, 
               t_interval=t_interval, oFigDir=oFigDir)

                  
