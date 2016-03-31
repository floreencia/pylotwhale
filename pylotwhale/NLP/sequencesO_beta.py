#!/usr/mprg/bin/python

import numpy as np
import pylab as pl
#import sys
import os
import pandas as pd
import random
import ast
import matplotlib.pyplot as plt
import matplotlib.colors as colors


"""
    Module for for the definition of sequences
    florencia @ 06.05.14

    Starting from a data frame and leading to the bigram counts.
    -- data frame --> data frames by recording (groupByRec + sortedRecDatFr)
    -- data frame --> time iterval distribution (plTimeIntervals)
    -- data frame --> sequences ()
    -- sequences --> bigram counts

"""


###########################################################
#####                     plotting                    #####
###########################################################


def plTimeIntervals(datB, plTitle='', outFig='', shuffl=0,
maxNTicks=False, TLims=(0, 19), frac=0, xLabel='ict [s]', yLabel='N'):
    """
    time interval distributions
    takes:
    < datB, data frame we whant to look into the time stamps
    < groupN, this is only given for the naming of the output plot
    < outFig, can be: outDir+"timeDistHist_%s.eps"%groupN
    * frac, display a line at the value of T below which frac*100%
            of the call pars are semarated, e.e, frac =0.75

    and returnsplots an histogram in the image folder

    """
    rec_datFrames, recs = sortedRecDatFr(datB, shuff=shuffl)  # recording data frames

    # time interval histogram
    interDist = {rec: rec_datFrames[rec].intervals.values for rec in rec_datFrames}  # intervals dictionary by recording

    # interval ditribution
    allT = list(interDist.values())  # all interval values
    allT = np.asarray([item for subli in allT for item in subli])  # flattern the intervals
    allT.reshape((len(allT), 1))  # reshape for dictionary
    allT = allT[~np.isnan(allT)]  #filter nans out
    print( np.shape(allT), allT.min(), allT.max())

     # histogram
    fig, ax = plt.subplots(figsize=(6,3))
    cax = ax.hist(allT[~np.isnan(allT)], bins=allT.max(), color =([0, 0.49, 0.47]))
    ax.set_xlim(TLims)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    if maxNTicks: plt.locator_params(nbins=maxNTicks)

    if frac:
        Cumx = np.cumsum(cax[0])
        Tot = sum(cax[0])
        tbins = cax[1]
        ax.axvline(x=sum(Cumx <= Tot * frac), linewidth=4, color=(1, 0.1, 0.0))#[(0, 0.7, 0.9)])
        print("line at", sum(Cumx <= Tot * frac))

    if plTitle: ax.set_title(plTitle)

    if outFig: fig.savefig( outFig, bbox_inches='tight')
    print("outFig:", outFig)

    #return fig, ax


def plTape(t, yRaw, plName='', scaleF = 20, lw = 0.05, title=''):
    """
    plots the calls vs. time
    """
    assert(len(t) == len(yRaw))

    freq_call = sorted([(yRaw.count(ucall), ucall) for ucall in
             np.unique(yRaw)], reverse=True, key=lambda x: x[0])  # sort calls
    i2c_tape = [thisCall[1] for thisCall in freq_call]
    c2i_tape = {i2c_tape[ix]: ix for ix in range(len(i2c_tape))}  # c2i
    #print np.unique(yRaw), c2i_tape, i
    #sys.exit()
    y = [c2i_tape[item] for item in yRaw]
    #print y[:5], i2c_tape, c2i_tape
    # plot

    #if not tapeN: tapeN = "%s%s%s"%(i2c_tape[0],len(i2c_tape), i2c_tape[-1])
    #figN = outDir+"tape_%s.pdf"%tapeN
    print(((t[-1] - t[0])/scaleF, np.min([np.max([1, len(freq_call) / 2]), 3])))
    fig = pl.figure(figsize=((t[-1] - t[0]) / scaleF, np.min([np.max([1, len(freq_call) / 2]), 3])))
    ax = fig.add_subplot(111)
    pl.plot(t, y, marker='|', lw=lw, markeredgewidth=1.5)
    ax.set_ylim(-0.5, len(c2i_tape))  # +0.1)
    ax.set_xlim(t[0] - 5, t[-1] + 5)
    ax.set_yticks(np.arange(len(c2i_tape)))
    ax.set_yticklabels(i2c_tape, fontsize=8)
    ax.set_xlabel('time [s]')
    ax.set_title(title)
    if plName: pl.savefig(plName, bbox_inches='tight')


def scattTape(t, yRaw, quality, plName='', scaleF=20, title=''):
    """
    Scatter plot of the calls in a tape, sorted acendinfgly with the
    frequence of the callS. the coulors represent the quality of the recording.
    yRaw must be a list
    """
    assert(len(t) == len(yRaw))
    q2i = {'A': 'k', 'B': 'b', 'C': 'g', 'D': 'r'}
    qual = [q2i[i] if i in q2i.keys() else 'gray' for i in quality]  # el: for different quality (buzz)

    freq_call = sorted([(yRaw.count(ucall), ucall) for ucall in np.unique(yRaw)], reverse=True, key=lambda x: x[0])  # inicialize
    i2c_tape = [thisCall[1] for thisCall in freq_call]
    c2i_tape = {i2c_tape[ix]: ix for ix in range(len(i2c_tape))}  # c2i

    print i2c_tape, len(qual), len(t)
    y = [c2i_tape[item] for item in yRaw]

    #plot
    print(t[-1]-t[0])/scaleF, np.min([np.max([1, len( freq_call )/2]), 3])
    fig, ax = plt.subplots(figsize=((t[-1]-t[0])/scaleF, np.min([np.max([1, len( freq_call )/2]), 3])))

    ax.scatter(t, y, marker='|', c=qual)
    ax.set_xlim(t[0] - 5, t[-1] + 5)
    ax.set_ylim(-0.5, len(c2i_tape))  # +0.1)
    ax.set_yticks(np.arange(len(c2i_tape)))  # ticks positions
    ax.set_yticklabels(i2c_tape, fontsize=8)  # ticks labels
    ax.set_xlabel('time [s]')
    ax.set_title(title)
    if plName: fig.savefig(plName, bbox_inches='tight')


def ngramsHist(df0, tau, histSize=500):
    '''
    This funcitons created an histogram of bigram sizes
    '''
    liS = df2listOfSeqs(df0, timeT = tau)
    ngrams = np.zeros(histSize)

    for i in range(len(liS)):
        #print i, liS[i]
        ngrams[len(liS[i])] +=1
    
    return ngrams


def plNgrams(df0, tau, outFN='', xLim=None, yScale='log', 
             xLabel='n', yLabel='#n-grams'):
    """
    This function plots the histograms of ngrams of the data frame
    """
    ngrams = ngramsHist(df0, tau)
        
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar( np.arange(len(ngrams[:xLim])), ngrams[:xLim])
    plt.yscale(yScale, nonposy='clip')
    ax.set_ylabel(yLabel)
    ax.set_xlabel(xLabel)
    if outFN: fig.savefig(outFN, bbox_inches = 'tight')
    print( outFN )
    return ngrams
    
    
def fancyClrBarPl(X, vmax, vmin, maxN=10, clrMapN='jet', clrBarGaps=15,
                  tickLabsDict='', outplN='', plTitle='', xL='N', yL=r'$\tau$',
                  figureScale=(), extendCbar='both', extent=None):
    
    '''
    draws a beautiful color plot
    tickLabsDict     dictionary where the keys are the label of the cba ticks
                    and the vallues a re te postions
    Parameters:
    ------------                    
        X : 2d numpy array
        vmax : max value to plot
        vmin : cutoff min value
        maxN : maximum number of columns
        extent : scalars (left, right, bottom, top)
        
    '''

    fig, ax = plt.subplots()

    #colors setting
    cmap = plt.cm.get_cmap('jet', clrBarGaps)    # discrete colors
    cmap.set_under((0.9, 0.9, 0.8)) #min
    cmap.set_over((1, 0.6, 0.6)) #max
    #cmap.set_nan((1, 0.6, 0.6)) #nan

    #plot
    cax=ax.imshow(X[:,:maxN], aspect ='auto', interpolation='nearest', 
               norm = colors.Normalize(vmin=vmin, vmax=vmax, clip = False),
               cmap=cmap, extent=extent)
    #labels
    ax.set_xlabel(xL)
    ax.set_ylabel(yL)
    if plTitle: ax.set_title(plTitle)

    #clrbar
    cbar = fig.colorbar(cax, extend=extendCbar) #min, max, both
    cbar.set_clim((vmin, vmax)) # normalize cbar colors
    if not tickLabsDict: 
        tickLabsDict = {vmin: vmin, int(vmax/2):int(vmax/2), vmax:vmax} # tick labels
    cbar.set_ticks(tickLabsDict.values())        
    cbar.set_ticklabels(tickLabsDict.keys())
    
    #figScale
    if len(figureScale)==2: fig.set_size_inches(figureScale)        
    
    if outplN: fig.savefig(outplN, bbox_inches='tight')


###########################################################
#####              data base construction           #####
###########################################################

def timeS2secs(datFr, timeStName='time_stamp'):
    timeS = datFr[timeStName]
    timeSs = [60*60*int(tS.split('_')[0])+60*int(tS.split('_')[1])+int(tS.split('_')[-1]) for tS in timeS]
    return timeSs
    
    
def constrainDF(df0, constrainType, constrainLi, baseName=''):
    '''
    Apply one constrain to a data base
    > df0 : constrained database
    > constrainType : a traing indicating the constrain
    ----------      
    < df0 : in data frame
    < contrainType : name of the column where constrai will be applyed
    < constrainLi : list of contrains ['B', 'C']
    < baseName : a string to which we'll indicate the added contrain
    '''
    if not isinstance(constrainLi, list) or isinstance(constrainLi, tuple):
        constrainLi = [constrainLi]
    
    baseName += '_%s'%constrainType
    boolarr = np.zeros(len(df0), dtype=bool)
    
    for const in constrainLi:
        boolarr = np.logical_or(boolarr, df0[constrainType] == const)
        baseName += '%s'%const
        
    df = df0[boolarr]
    
    return df, baseName
    
def constrainDataFrame(df0, constDict, baseN = '', reindex=False):
    '''
    this functions constrains a data frame accordingly with the 
    given constrain dictionary
    Apply one constrain to a data base
    Params:
    -------
    < df0 : in data frame
    < constDict : constains dictionary 'col_name' : [values]
    < baseN : string to which we'll append the constrain string
    < reindex : if True, the indexes fo the returned array will 
                renewed for newly fresh natural numbers
    ------->
    > df : constrained dataframe
    > baseN : a traing indicating the constrain
    '''
    
    df = df0.copy()
    for constType in constDict.keys():
        print(constType)
        df, baseN = constrainDF(df, constType, constDict[constType], baseN)
    
    if reindex: df = df.set_index(np.arange(len(df)))
    return df, baseN
    
    
    
def df2timeStamps(df, txtFN,
                  t0_col='timeSs', tf_col=None,
                  label_col='call' ):
    '''
    creates an annotations file from the timestamps in the dataframe
    < df  : data frame
    < txtFN : out fileName
        'whales/data/mySamples/whales/tapes/NPW/annotations/'
    < t0_col : name the column with the starting time
    < tf_col : name the column with the ending time 
                in None, this is one second after the starting col
    < label_col : name of the column with the label
    '''
    
    ## file names handling
    FN0, exFN = os.path.splitext(txtFN)
    if not exFN: exFN = 'txt'
    
    ## sort dataframes and separate them by tape
    dfD, recs = sortedRecDatFr(df)

    for r in dfD.keys(): ## iterate over the tape-dataframes
        annotFN = FN0 + '-%s.'%r + exFN ## out file name
        thisdf = dfD[r]
        ## columns
        t0s = thisdf[ t0_col ] # c1
        if tf_col == None: # c2
            tfs = t0s + 1
        else:
            tfs = thisdf[ tf_col ]  
        cs = thisdf[ label_col ] # c3
        anndf = pd.concat([t0s, tfs, cs], axis=1) # conncat series
        anndf.to_csv(annotFN, sep='\t', header=False, index=False)
        

###########################################################
#####          litsting, sorting and counting         #####
###########################################################

def groupByRec(dataFr, categ='recording'):
    """
    groups data by recording
    returns:
    > 1 dictionary: recordings --> indexes of the recordings in same rec
    > 2 an array of with the recordig names
    """
    recSetsD = dataFr.groupby(categ).groups #dictionary with the recordings
    recs = recSetsD.keys()
    #print "#recordings", len(recSetsD)
    return recSetsD, recs


def sortedRecDatFr(dataFr, shuff=0, categ='recording', sortL='timeSs'):
    """
    Params:
    -----
    < dataFr : data frames for each recording sorted temporally
    < shuff : if shuff != 0 the time series are shuffled
    < categ : category into which we will separate the data
    < sortL : sorting label, the time stamp by default
    ---->
    > sortedRecsDF : a dictionary of the data frames sorted sorted ascendingly
    > recs : an array of with the recording names
    """
    if(shuff):
        f = shuffleSeries
    else:
        f = lambda x: x

    recGr, recs = groupByRec(dataFr, categ = categ)
    sortedRecsDF = {}

    for thisRec in recGr:
        recDatFr = dataFr.ix[ recGr[thisRec] ]  # select series
        recDatFr = f(recDatFr)  # shuffle or not
        recDatFr = recDatFr.sort(sortL)
        recDatFr['intervals'] = (recDatFr[sortL]-recDatFr[sortL].shift())  #compute intervals
        sortedRecsDF[thisRec] = recDatFr

    #print "#recordings", len(sortedRecsDF)
    return sortedRecsDF, recs # data frames and labels


def shuffleSeries(dataFr, shuffleCol='timeSs'):
    """
    shuffles a series (shuffleCol) from a data frame:
    > data frame
    > name a of the column ti shuffle
    """
    x = dataFr[shuffleCol].values.copy()  # select series
    random.shuffle(x)  #shuffle
    shuffledRecs = dataFr.copy()  # new series
    shuffledRecs[shuffleCol] = x # set shuffled series

    return shuffledRecs # data frames and labels


def bigramCounts( li, adj0 = {}, call2index0 = {}, index2call0 = []):

    """
    sequences to dictionary of bigram counts
    list a list and returns:
    < li, sequence of tolkens, in for of a list
    > adj0, the adjacency 2D-dictionary of consecutive elements
    > the call to index dicitonary
    > the index to call list

    If an adjacency matrix is already given then we add the counts to it
    """

    ### CHECK INPUT
    assert( len(call2index0) == len(index2call0))# and len(adj0) >= len(call2index0) -1 )
    # dictionaries must have the same lenght and adj at least the same lenght - 1

    ### INICIALIZATIONS
    ## copy the values o avoid operating over a global mutable variable
    adj = adj0.copy()  # adjacenct dicitonary 
    call2index = call2index0.copy()  # call to index dict
    index2call = index2call0[:]  # index to call dict
    
    ##
    call_0 = '__INI' # previous call ini
    call_1 = '__END'  # end call
    sequence = li[:] # copy the list
    sequence.append(call_1) # add end tolken to li

    if len(call2index) == 0: # if empty dict
        print "inicializing"
        call2index[call_0] = 0
        call2index[call_1] = 1
        index2call.insert(0, call_0)  # inicialize it w/ __INI
        index2call.insert(1, call_1)  # inicialize it w/ __END

    n = len(index2call)  # init with the no. of calls in the dict.

    ### ITERATE OVER TOLKENS IN THE SEQUECE
    for call in sequence:
        if call not in call2index.keys():  # call in not in dict
            call2index[call] = n  # add call to dictionary
            index2call.insert(n, call)
            n += 1

        # adjacency dictionary
        if (call2index[call_0], call2index[call]) in adj.keys(): # plus one
            adj[call2index[call_0], call2index[call]] += 1
        else:
            adj[call2index[call_0], call2index[call]] = 1
        call_0 = call # reeset previous call

    return adj, call2index, index2call

def df2listOfSeqs( datB0, timeT=5, feature='call', shuffl=0):
    """
    ~ listOfSeqs, but doesn't filters the dataframe by group
    takes a data frame and returns a list of sequences
    < datB0, dataframe with the calls (feature) and the timestamps
    < timeT, maximum time interval within consecutive calls
    < shuffle != 0, shuffles the data
    > list of lists, where the lists are que sequences.
    """
    seqsLi = []
  
    rec_datFrames, recs = sortedRecDatFr(datB0, shuff=shuffl)  # dataframes by recording sorted temporally
    seqsLi += [ seqsFromRec(rec_datFrames[rec], timeT, feature).sequences for rec in rec_datFrames ]  # list of sequences by rec
    thisGrSeqs = [item for sublist in seqsLi for item in sublist]  # flatter the list of sequecne so taht they are no longer separated by recording
    return thisGrSeqs

def listOfSeqs( datB0, groupN=[], timeT=5, feature='call', shuffl=0):
    """
    list of sequences
    takes a data frame and returns a list of sequences
    ! assumes that the recording names are not repeated. CHECK whether this holds
    < datB0, dataframe with the calls (feature) and the timestamps
    < groupN = [], gives the option to create the list of sequences for
        various groups, without mixing the sequences accross groups
    > list of lists, where the lists are que sequences.
    """
    seqsLi = []
    for gr in groupN:
        '''
        defines the sequences
        '''
        datB = datB0.loc[datB0.group == gr]  # select group data
        rec_datFrames, recs = sortedRecDatFr(datB, shuff=shuffl)  # dataframes by recording sorted temporally
        seqsLi += [ seqsFromRec(rec_datFrames[rec], timeT, feature).sequences for rec in rec_datFrames ]  # list of sequences by rec

    thisGrSeqs = [item for sublist in seqsLi for item in sublist]  # flatter the list of sequecne so taht they are no longer separated by recording
    return thisGrSeqs


def listOfSeqs2BigramCounts(li, M = None, c2i = None, i2c = None):
    """
    transforms a list(li) of sequences into bigram counts
    returns:
    beta - because inicializes the dicionaries from the calling of the function
    giving the chance countinue adding bigrams to existing counts
    > bigrma counts dictionary (M)
    > call to index dictionary (c2i)
    > index to call array (i2c)
    """
    if M is None: M = {}
    if c2i is None: c2i = {}
    if i2c is None: i2c = []

    for thisLi in li:
        M, c2i, i2c = bigramCounts(thisLi, M, c2i, i2c)#M2

    return (M, c2i, i2c)


###########################################################
#####             sequence statistics           #####
###########################################################


def getTimes(datB):
    rec_datFrames, reccs = sortedRecDatFr(datB) # reccording data frames
    # time interval histogram
    interDist = {rec: rec_datFrames[rec].intervals.values for rec in rec_datFrames} # intervals dictionary by reccording

    # interval ditribution
    allT = interDist.values() # all interval values
    allT = np.asarray([item for subli in allT for item in subli]) # flattern the intervals
    allT.reshape((len(allT), 1)) # reshape for dictionary
    allT = allT[~np.isnan(allT)] #filter nans out
        
    return allT
    

def tauNgramsMatrix(df, tau0=0, tauf=20, histSize=None):
    '''
    ngrams as a function of tau
    > matrix with the number of ngrams
    rows -- time
    columns - ngram
    '''
    if not histSize: histSize = len(df) + 1 # set hist size
        
    X = ngramsHist(df, tau0, histSize=histSize ) 
    Ncalls = np.sum(X*np.arange(len(X)))
    assert( len(df) == Ncalls )
    
    for tau in np.arange(tau0 + 1, tauf):
        x= ngramsHist(df, tau, histSize=histSize)
        X = np.vstack((X,x.T))
    return(X, Ncalls)

###########################################################
#####             bigram analysis functions           #####
###########################################################

def normalize_2grams_dict(biGrmDict):
    """
    this funciton gets a bigram dictionary with the bigram counts
    and returns the normalized probablies

    Jul, 2014 (A)
    """

    assert( isinstance( biGrmDict, dict) )  # check input
    counts2grm = {i_x[0]: 0 for i_x in biGrmDict.keys()} # inicialize normalized dictionary w/ zeros

    # count the bigrams
    for A in counts2grm.keys(): # iterate the initial keys 'A'
        #print A
        for elem in biGrmDict.keys(): # iterate over the tuple key
            #print elem
            if elem[0] == A: counts2grm[A] += biGrmDict[elem]

    # normalize counts
    norm1_BiGrmDict = {}
    for elem in biGrmDict.keys():
        norm1_BiGrmDict[elem] = 1.0 * biGrmDict[elem] / counts2grm[elem[0]] if counts2grm[elem[0]] else np.nan

    return norm1_BiGrmDict

def select2grmsDict_lrgr_NotIni( datFr_li, lgr=5, removeSet=[0,1]):
    """
    selects the relevant labels from a bigram dictionary. Labels:
    - from the bigrams with more than lgr occurrences
    - not taking into account the __INI label
    - removeSet = [c2i['__INI'], c2i['__END']], list of indexes to leave out
    - the first line of the data frame contains the number of bigrams.
    > list of tuples, with the keys of the elements larger than lgr

    Jul, 2014 (B)
    """
    myProbs = datFr_li[ datFr_li > lgr ]  # larger than threshold
    
    litu = [ast.literal_eval(item) for item in myProbs.keys().values]  # string -> tuple
    no0 = litu[:]    
    for remItem in removeSet:
        no0 = [item for item in no0 if remItem not in item]  # filter out __INI, __END
    return no0

def plDistributionAndRealVal( dist, realVal = False, mu = False, h = False, nBins = 10, outFN='', plLabel='', plTitle='', maxNTicks = False):
    """
    Plots a the distribution (~histogram) of the distribution 'dist' and a line where the real value is

    Jul, 2014 (B)
    """
    #histogram
    Nvec, binVec = np.histogram( dist, bins = nBins )  # shuffled data histogram
    bincenters = 0.5*( binVec[1:] + binVec[:-1] )
    #pl.plot(bincenters, Nvec, 'b-')
    x = bincenters
    y = Nvec
    #print x, y
    pl.fill_between( x, y, alpha = 0.5)
    pl.plot(x,y, '.')
    if plLabel: pl.xlabel("%s"%plLabel)
    if plTitle: pl.title("%s"%plTitle)
    if maxNTicks: plt.locator_params(nbins=maxNTicks)


    # lines
    if realVal: pl.axvline( x = realVal, color='r', ls ='-', lw = 3.5)  # real value
    if mu: pl.axvline( x = mu, color='b', ls ='-', lw = 2)  # mean
    if h:
        pl.axvline( x = mu-h, color='b', ls ='--', lw = 2)  # left conf int
        pl.axvline( x = mu+h, color='b', ls ='--', lw = 2)  # right conf int

    if(outFN):
        print outFN
        pl.savefig(outFN, bbox_inches='tight')

def plDistributionAndRealVal_bar( dist, realVal = False, mu = False, h = False, nBins = 10, outFN='', plLabel='', plTitle=''):
    """
    Plots a the distribution (~histogram) of the distribution 'dist' and a line where the real value is

    Jul, 2014 (B)
    """
    #histogram
    pl.hist( dist, bins = nBins )  # shuffled data histogram
    #bincenters = 0.5*( binVec[1:] + binVec[:-1] )
    #pl.plot(bincenters, Nvec, 'b-')
    #x = bincenters
    #y = Nvec
    #print x, y
    #pl.fill_between( x, y, alpha = 0.5)
    #pl.plot(x,y, '.')
    if plLabel:
        pl.xlabel("%s"%plLabel)
    if plTitle:
        pl.title("%s"%plTitle)

    # lines
    if realVal: pl.axvline( x = realVal, color='r', ls ='-', lw = 3.5)  # real value
    if mu: pl.axvline( x = mu, color='b', ls ='-', lw = 2)  # mean
    if h:
        pl.axvline( x = mu-h, color='b', ls ='--', lw = 2)  # left conf int
        pl.axvline( x = mu+h, color='b', ls ='--', lw = 2)  # right conf int

    if(outFN):
        print outFN
        pl.savefig(outFN, bbox_inches='tight')


def adjBining( dists, Nbin0=5, minBinCont = 3, maxNbins=100 ):
    """
    finds the number of bins such that no bin gets less than minBinCont
    * dists, an array with the bigram probabilities of the shufflings

    Jul, 2014 (B)
    Nbin0 - stating number of bins
    minBinCont  - minimum number of elements per bin
    maxNbin - maximum number of bins
    """

    #Adjusting the bining
    n = Nbin0
    while True:
        n+=1
        Nvec = np.histogram( dists, bins = n )[0]  # shuffled data histogram
        if Nvec.min() < minBinCont or n > maxNbins:
            print n, Nvec.min()
            break


    return n-1


#### integrated functions

def printShuffledDistribution(datB, Nshuffl, baseN, minNumCalls=4,\
                              timeT=4, feature='call', normalize=0):

    ''' 
    This function takes a data frame with the calls and time stamps and writes
    down a csv with the bigram counts and the Nshshuflins bigrams observations
    ----        
    > csvShufflingDataFrame
    > .dat file with the index to call dictionary
    '''

    if( len(datB) <= minNumCalls):
        print('Only %d calls (min %d calls)'%(len(datB), minNumCalls))
        return 0
    
    
    ### Non shuffled data
    liS = df2listOfSeqs(datB, timeT=timeT, feature=feature, shuffl=0)  # define seqs
    NBiG0, c2i, i2c = listOfSeqs2BigramCounts(liS)  # count bigrams
    df = pd.DataFrame(NBiG0, index=range(1))  # inicialize dataframe w/ the counts
    NBiG_normalized = normalize_2grams_dict(NBiG0)  # normalize bigrams
    newDf = pd.DataFrame(NBiG_normalized, index=np.arange(1))  # dict -> dataFrame
    df = pd.concat([df, newDf], ignore_index=True)  # concat dataFrames
    
    ### shuffled data calculations
    baseN += '_T%d_NORM%d_N%d'%(timeT, normalize, Nshuffl)
    if normalize: 
        normfun = lambda x : normalize_2grams_dict(x) 
    else:
        normfun = lambda x : x
        
    for i in range(Nshuffl):  # SHUFFLE, Nshuffl=1 was inicialization => substract 1
        liS = df2listOfSeqs(datB, timeT=timeT, feature=feature, shuffl=1)  # this group slection just double checks

        X = {j: 0 for j in NBiG0.keys()}  # inicialize adj dictionary

        print(i)
        NBiG, c2i, i2c = listOfSeqs2BigramCounts(liS, X, c2i, i2c)  # bigrams
        
        NBiG_normalized = normfun(NBiG)  # normalize
            
        newDf = pd.DataFrame(NBiG_normalized, index=np.arange(1))
        df = pd.concat([df, newDf], ignore_index=True)  # save run statistics


    ### print distributions
    outcsv = baseN +'.csv'
    print("out:", outcsv, len(df))
    df.to_csv(outcsv, index=False)

    ### print dictionary
    outdict = baseN + '.dat'   
    print( "out:", outdict, len(df))
    fi = open(outdict, 'w')
    fi.writelines(["%s\n" % item for item in i2c])
    fi.close()




#################################################################
# OBJECTS
#################################################################

class seqsFromRec:
    """
    this class creates the sequences form a data frame with:
        * a single recording
        * elements sorted temporally (out: sortedRecDatFr)
    """

    def __init__( self, data_frame, timeT = 5, feature ='call' ):

        self.dataFr = data_frame #one recording data frame,
        self.timeT = timeT
        self.feature = feature
        self.sequences = self.rec2seqs()

    def rec2seqs(self):
        """
        defines the sequences for each day-tape label
        """
        indx = self.dataFr.index
        #print "#calls", len(indx)
        seqs = []
        thisSeq = [ self.dataFr[self.feature].ix[indx[0]] ]  # the fist call in the rec

        for i in indx[1:]:
            if self.dataFr.ix[i].intervals <= self.timeT: # continue sequence
                thisSeq.append(self.dataFr[self.feature].ix[i])
            else: # new sequence
                seqs.append(thisSeq)
                thisSeq = [ self.dataFr[self.feature].ix[i]]

        seqs.append(thisSeq)

        return seqs


