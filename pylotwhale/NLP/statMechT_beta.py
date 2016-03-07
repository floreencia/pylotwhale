#!/usr/mprg/bin/python

import numpy as np
#import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import os
import re
import pandas as pd
#import random
#import ast
import itertools as it
#import sequencesO_beta as seqs


"""
    Module for the computation of probabilities
    statisticalmachanically inspired
    florencia @ 19.08.14

    Starting from a data frame and leading to the bigram counts.
    -- data frame --> data frames by recording (groupByRec + sortedRecDatFr)
    -- data frame --> time iterval distribution (plTimeIntervals)
    -- data frame --> sequences ()
    -- sequences --> bigram counts
        
    see:
    /NLP/NPWvocalRepertoire/ising/

"""

###########################################################
#####                    plotting                     #####
###########################################################

'''
def plTape(tapeFile, outF = '', shwcBar = False, xScale = 50, yScale = 2):

    """plots the time seris of a tape (Bbeta_statMEch.ipynb)
    < tapeFile - csv file
    < outdir, when this is not specified the imae will be stored in the same dir as th input file
    """
    
    noExt = tapeFile.split('.csv')[0]
    #outF = os.path.abspath(os.path.expanduser(noExt)) if not outDir else os.path.join(outDir, os.path.split(noExt)[-1])
    #outF = outF + 'imShw.png'

    winSz = tapeFile.split('WS')[-1][0] if re.search('WS', tapeFile) else 0
    
    df = pd.read_csv(tapeFile)
    calls_T = df.columns.values[2:]
    datM = np.asarray(df[calls_T].T)
    Nc, Nt = np.shape(datM)
    fig, ax =  plt.subplots(figsize = (int(Nt/xScale),int(Nc/yScale)))
    im = ax.imshow( datM, aspect = 'auto', interpolation = 'nearest', cmap = pl.cm.gist_stern_r, extent = [0, Nt*int(winSz), 0, Nc] )
    ax.set_yticks(np.arange(len(calls_T)) + 0.5)
    ax.set_yticklabels(calls_T[::-1])
    ax.set_xlabel('time [s]')
    if shwcBar: cbar = fig.colorbar(im, ticks = np.arange(np.max(datM)+1)) #, shrink = 1.0/asp)

    print outF, winSz

    if outF: fig.savefig(outF, bbox_inches='tight')
'''

    
def fancyClrTicksPl(X, vmax, vmin, clrMapN='jet', clrBarGaps=15,
                  tickLabsCbar='', outplN='', plTitle='', xL='time [s]', ytickL='',
                  figureScale=(), extent=[], xlim = ''):
    
    '''
    draws a beautiful color plot
    tickLabs=("<%d"%vmin, int(vmax), '>%d/%d'%(Ncalls, frac))
    '''

    fig, ax = plt.subplots()

    #colors setting
    cmap = plt.cm.get_cmap('jet', clrBarGaps)    # discrete colors
    cmap.set_under((1, 1, 1)) #min -> white

    #plot
    cax=ax.imshow(X, aspect ='auto', interpolation='nearest', 
               norm = colors.Normalize(vmin=vmin, vmax=vmax, clip = False),
               cmap=cmap)
    
               
    if len(extent) ==4: cax.set_extent(extent)
        
    #labels
    ax.set_yticks(np.arange(len(ytickL)) + 0.5)
    ax.set_yticklabels(ytickL[::-1])
    ax.set_xlabel(xL)
    if xlim: ax.set_xlim(xlim)
    
    if plTitle: ax.set_title(plTitle)

    #clrbar
    cbar = fig.colorbar(cax, extend='min')  #"neither", "both", "min", "max"
    cbar.set_clim((vmin,vmax))
    cbar.set_ticks((vmin, int(vmax/2), vmax))
    if not tickLabsCbar: tickLabsCbar = (vmin, int(vmax/2), vmax)
    cbar.set_ticklabels(tickLabsCbar)
    
    #figScale
    if len(figureScale)==2: fig.set_size_inches(figureScale)        
    
    if outplN: fig.savefig(outplN, bbox_inches='tight')


def csvPlTape(tapeFile, outDir= '', xScale = 50, yScale = 2):

    """plots the time seris of a tape (Bbeta_statMEch.ipynb)
    < tapeFile - csv file
    < outdir, when this is not specified the image will be stored in the same
    dir as th input file
    """
    
    # file handling
    noExt = tapeFile.split('.csv')[0]
    baseN = os.path.basename(noExt)    
    outF = os.path.abspath(os.path.expanduser(noExt)) if not outDir else os.path.join(outDir, os.path.split(noExt)[-1])
    outF = outF + '-imShw.png'
    
    # determine window size
    winSz = re.search(r'WS([0-9]*)', tapeFile).group(1) # tapeFile.split('WS')[-1][0] if re.search('WS', tapeFile) else 0
    
    # read time series data frame
    df = pd.read_csv(tapeFile)
    calls_T = df.columns.values[2:]
    datM = np.asarray(df[calls_T].T)
    maxNumOfCalls = np.max(datM)
    if maxNumOfCalls == 1 : maxNumOfCalls +=1
    Nc, Nt = np.shape(datM)
    extent = [0, Nt*int(winSz), 0, Nc]
    
    ### plot settings
    figScale = (int(Nt/xScale),int(Nc/yScale))
    fancyClrTicksPl(datM, vmax=maxNumOfCalls, vmin=1, clrBarGaps = maxNumOfCalls,
                        plTitle='%s'%baseN, xL='time [s]', ytickL=calls_T,
                        outplN=outF, figureScale=figScale, extent=extent)
        
    print outF, winSz

    #if outF: fig.savefig(outF, bbox_inches='tight')
###########################################################
#####                 data processing                 #####
###########################################################


def IO_timeSeriesF(time, callsRaw, outF, winSz = 5):
    """
    This funtion takes the a list with the calls and time stamps
    and generates a binary csv file with the activation of the calls in a time window.
    """
    i2c = list(set(callsRaw))
    c2i = {i2c[ix]: ix for ix in range(len(i2c))}
    
    t0 = 0
    time = list(time)
    f = open(outF, 'w')    
    f.write("t0,tf,%s\n"%",".join(i2c)) # header
    
    for dt in range(time[0]-winSz/2, time[-1]+2*winSz, winSz): # window the tape
        
        li = [(time[ix] ,callsRaw[ix]) for ix in range(len(time)) if time[ix] <= dt and  time[ix] > t0] # choose calls in the interval
    
        IOvec = [0]*len(i2c)
        for (bla, item) in li: #
            IOvec[c2i[item]] +=1
    
        f.write("%d,%d,%s\n"%(t0, dt, ",".join(str(item) for item in IOvec ))) # time
        t0 = dt

    f.close()   

###########################################################
#####               observed correlations             #####
###########################################################

def Probs_obs_fromDataFrame(dataFrame):
    """
    Probabilities of finding N calls in a time window
    < dataFrame, contains columns with the observed calls
    > returns a numpy array P, 
        where P[n] is the probability of finding n correlations in a time window
    """
    
    calls_T = dataFrame.columns.values[2:] # count the number of calls in the window
    Ncalls_time = dataFrame[calls_T].sum(axis = 1).values # number of calls in a time window
    norm_obs = len(Ncalls_time)*1.0
    largest_corr = int(np.max(Ncalls_time))
    PnCalls_obs = np.bincount( Ncalls_time )/norm_obs
    
    return PnCalls_obs


def Probs_obs(dataM):
    """
    Probabilities of finding N calls in a time window
    < dataFrame, contains columns with the observed calls
    > returns a numpy array P, 
        where P[n] is the probability of finding n correlations in a time window
    """
    
    #calls_T = dataFrame.columns.values[2:] # count the number of calls in the window
    Ncalls_time = dataM.sum(axis = 1) # number of calls in a time window
    norm_obs = len(Ncalls_time)*1.0
    largest_corr = int(np.max(Ncalls_time))
    PnCalls_obs = np.bincount( Ncalls_time )/norm_obs
    print np.sum(PnCalls_obs)
    return PnCalls_obs

###########################################################
#####       Independent particle probabilities        #####
###########################################################

def partition(number): # pirateada de stackovreflow
     answer = set()
     answer.add((number, ))
     #print answer
     for x in range(1, number):
         for y in partition(number - x):
             answer.add(tuple(sorted((x, ) + y)))
     return answer

def buildProbMatrix( dataM ):
    N_ts, N_c = np.shape(dataM)
    P = np.zeros((np.max(dataM)+1, N_c))
    for i in np.arange(N_c):
        cts = np.bincount( dataM[:, i] )
        for j in np.arange(len(cts)):
            P[j,i] = 1.0*cts[j]/N_ts
    return P

def genCombinations(n, N_calls, maxCorr):
    """ determines all the configurations of a given energy n
    n = energy
    N_calls =  number of calls (particles)
    maxCorr = largest energy of a particle, this E constrains the rows of P, = len(P) -1
    > C combiat
ions array
    """    
    parts = list(partition(n)) # get the partitions
    parts_f0 = [i for i in parts if len(i) <= N_calls] # filter out, more particles
    parts_f = [i for i in parts_f0 if np.max(i) <= maxCorr] # filter out, supper high energies
    C=[]
    # C = np.asarray(list(set(it.permutations( parts_f[0] )))) # inicialize C
    for item in parts_f:
        part = np.asarray(item)
        if len(part) < N_calls: part = np.hstack( ( part, np.zeros(N_calls - len(part), dtype=int ) ) )  # add zeros
        
        combos = np.asarray(list(set(it.permutations( part )))) # combinations for partition
        if len(C) < 1: 
            C = combos
        else:
            C = np.vstack((C,combos))
    return C
    
def comboP(P, ix):
    """ Evaluates the probability of one combination given by the indexes ix """
    assert( len(ix) == len(P[0]) )
    p = 1.0
    for colI in np.arange(len(ix)):
        p *= P[:, colI][ix[colI]]
    return p


def P_n( P, n ):
    """ computes the probability of getting a state with energy n, gieven the probabilities P """
    
    maxC, N_calls = np.shape(P)
    combosI = genCombinations(n, N_calls, maxC -1 )

    s = 0
    for ix in combosI: # the P of evaluate all the combinations with energy n
        pr = comboP(P, ix)
        # print pr
        s += comboP(P,ix)
    return s

def P_n_wD( P, n, maxEpp ):
    """ computes the probability of getting a state with energy n, gieven the probabilities P """
    
    maxEpp, N_calls = np.shape(P)
    combosI = get_partitions(n, N_calls, maxEpp )

    s = 0
    for ix in combosI: # the P of evaluate all the combinations with energy n
        pr = comboP(P, ix)
        # print pr
        s += comboP(P,ix)
    return s




#######################################################################
###################                   deb                ##############
#######################################################################


def partitionD(li, npart, done, maxEpp):
    #print done 
    if npart == 1: # ending condition of the recursive function
        return [li]
    else:
        toret = [li[:done] + frag + li[done+1:] for frag in split(li[done], maxEpp, npart)] 
        #print toret
        """
        toret -- take all the splits, join them with li, skiping the splited number, and saves each split as an ele
                ment of the list
        
        [partition( gg, npart - 1, done + 1 ) for gg in toret] -- calls partition util npart == 1, result is here!
        ### rest is just for flattering
        for sublist in []
        [for item in sublist] == [item for sublist in l for item in sublist] 
        
        """
        ans =  [item for sublist in [partitionD( gg, npart - 1, done + 1, maxEpp ) for gg in toret] for item in sublist]
        return ans #filter( lambda x : max( x ) < maxEpp, ans ) 
        

    
def split(N, maxEpp, npart):
    """ splits a number into all it's 2 nmber partitions """
    
    if N>0:
        return [[i,N-i] for i in range(N+1) if i <= maxEpp ]
    else:
        return [[0,0]]

def get_partitions(N,npart, maxEpp=100):
    """
    This function returns the integer composition of N in Z^+ inro npart nonnegative parts
    N = energy
    npart = number of particles
    """
    print N, npart
    assert( 0 <= N < 20)
    assert( 0 <= npart < 20)
    x = partitionD([N], npart, 0, maxEpp)
    return [i for i in x if np.max(i) < maxEpp]
