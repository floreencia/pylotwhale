#!/usr/mprg/bin/python

import numpy as np
#import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
import sys
#import time 
import itertools as it
from collections import Counter

#import os
import ast
import pandas as pd
import nltk
from nltk.probability import ConditionalProbDist, MLEProbDist
import pylotwhale.utils.dataTools as daT

#import sequencesO_beta as seqs
sys.path.append('/home/florencia/whales/scripts/')
import matrixTools as mt 

"""
    Module for doing statistics over ngrams
    florencia @ 16.05.14

"""

#################################################################################
##############################    FUNCTIONS    ##################################
#################################################################################


##############################    PLOTTING    ##################################

def plBigramM(A, i2c, groupN='', figDir='', shuffl_label=0, 
              Bname='bigram', ext='eps', plTitle='', clrMap = 'winter_r', 
              labSize='18', cbarLim=(1,None), cbarOrientation='vertical', 
              cbarTicks=False, cbarTickLabels=False, cbar=True, outFig='',
              figsize=None):
    """
    plots a bigram matrix
    A : bigram matrix    
    
    OPTIONAL PARAMETERS
    * figDir, if you whant to save the image, give the directory
    * cbarLim, tuple that with the limits of the cbar i.e. (0,1) for
        probabilities. False sets the default limits, the min and the max of the matrix.
    * cbarTick, location of the ticks in the cbar (list) i.e. [0, 0.5, 1]
    * cbarTickLabs, tick labes of the cbar (list)

    > shuffl_label -- doesen't shuffles the data (since yhis method pnly plots) but includes the shuffle lable in the name of the file
    > xM and yM, if False plot all
    """
    if(shuffl_label):
        Bname = Bname+'_shuffled'
    
    '''
    if not xM: xM = len(i2c) # masked labels
    if not yM: yM = len(i2c) #masked labels
    '''

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.get_cmap(clrMap, 11)    # 11 discrete colors
    cmap.set_under('white') #min
    cmap.set_bad('white') #nan

    cax = plt.imshow(A, interpolation = 'nearest', cmap = cmap, 
                    extent = [0,len(i2c),0,len(i2c)], origin='bottom')
    
    #ax.set_xlim((0,xM))
    #ax.set_ylim(0,yM)
    xM = len(i2c) # masked labels
    yM = len(i2c) #masked labels
    ax.set_yticks( np.arange(len( i2c[:yM+1]) ) + 0.5 ) #flo -> +0.1)
    ax.set_yticklabels( i2c[:yM+1] ) 
    ax.set_xticks( np.arange(len( i2c[:xM+1]) ) + 0.5 ) 
    ax.set_xticklabels( i2c[:xM+1], rotation=90 ) 
    if plTitle: ax.set_title(plTitle)
    #COLOR BAR
    if isinstance(cbarLim, tuple): cax.set_clim(cbarLim) # cmap lims

    if cbar: 
        cbar = fig.colorbar(cax, extend='min')
        if isinstance(cbarLim, tuple): cbar.set_clim(cbarLim) # cbar limits
        if isinstance(cbarTicks, list): cbar.set_ticks(cbarTicks)
        if isinstance(cbarTickLabels, list): cbar.set_ticklabels(cbarTickLabels)
    '''
    if figDir: 
        figN = os.path.join(figDir, Bname+'%s.'%groupN + ext)
        pl.savefig(figN, bbox_inches = 'tight')
        print len(i2c), "\nout:%s"%figN
    '''
        
    if outFig: 
        fig.savefig(outFig, bbox_inches = 'tight')
        print len(i2c), "\nout:%s"%outFig


def barPltsSv(y, labs, figN='', figSz=(10, 3), yL='# bigrams', 
              plTit='', plLegend=0, maxNyTicks=0, yTickStep=50 ):
    """
    plots stocked histograms
    i.e. the number of bigrams repetiotions/differences
    y can be an array of the bar values of the plot
    * plLegend list with legends of the plots, i.e ['different call', 'repetition'] 
    * maxNyTicks - maximum number of y ticks. 0- default (matplotlib decides)
        not working
    """

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']   
    fig = plt.figure(figsize=figSz)
    ax = fig.add_subplot(111)
    y0 = np.zeros(len(y[0]))
    p = []
    maxY = 0 # for ajusting the y scale
    for i in range(len(y)):
        assert(len(y0) == len(y[i])) #all the arr must have the same dimension
        p.append(ax.bar( range( len(y[i]) ), y[i], color = colors[i], bottom = y0))
        y0 = y[i]
        maxY += y0[0]
        
    ax.set_xticks( np.arange(len( y0 ) ) + 0.5 )
    ax.set_xticklabels( labs, rotation='vertical' ) 
    ax.set_ylabel(yL)
    if plTit: ax.set_title(plTit)#, size = 16)
    
    print np.nanmax(y), y0
    if yTickStep: ax.set_yticks(np.arange(0, int(maxY), yTickStep))
    
    if isinstance(plLegend, list):
        if len(plLegend) == len(p): # set labels
            ax.legend( tuple(p), tuple(plLegend) )
            print "LABELS:", plLegend

    if figN: 
        fig.savefig(figN, bbox_inches='tight')
        print figN


### stats plots


def pl_ic_bigram_times(df0, my_bigrams, ignoreKeys='default', label='call', oFig=None, 
                       violinInner='box', yrange='default', ylabel='time (s)',
                       minNumBigrams=5):
    '''violin plot of the ict of a my_bigrams
    Parameters:
    -----------
        df0 : pandas dataframe wirth ict column
        mu_bigrams : sequence to search for
        ignoteKeys : 'default' removes  ['_ini', '_end']
        label : type of sequence
        oFig : output figure
        violinInner : viloin lor parameter
        yrange : 'default' (0, mu*2)
    '''
        
    if ignoreKeys == 'default': ignoreKeys = ['_ini', '_end']
        
    topBigrams = daT.removeFromList(daT.returnSortingKeys(Counter(my_bigrams)), ignoreKeys)
    bigrTimes=[]
    bigrNames=[]

    for seq in topBigrams:
        df = daT.returnSequenceDf(df0, seq, label=label)
        #print(len(df))
        ict = df.ict.values
        if len(ict) > minNumBigrams:
            #bigrTimes[tuple(seq)] = ict[ ~ np.isnan(ict)]
            bigrTimes.append(ict[ ~ np.isnan(ict)]) 
            bigrNames.append(seq)
        
    kys = ["{}{}".format(a,b) for a,b in bigrNames ]
    #sns.violinplot( bigrTimes, names=kys, inner=violinInner)
    sns.boxplot( bigrTimes, names=kys)
            
    if yrange == 'default':
        meanVls = [np.mean(item) for item in bigrTimes if len(item) > 1]
        yrange = (0, np.mean(meanVls))
    plt.ylim(yrange)
    plt.ylabel(ylabel)
    plt.savefig(oFig)  


##### NLTK - pandas - related ngram code  #####

def bigrams2Dict(bigrams_tu):
    '''
    converts a 2D-tuples list into a conditionalFreqDist (~2D-dictionary)
    eg. [(a,b) ... ] --> Di[a][b] = #(a,b)
    :bigrams_tu: bigrams as a list of tuples
    '''
    cfd = nltk.ConditionalFreqDist(bigrams_tu)
    return cfd
 
def bigramsdf2bigramsMatrix(df, conditionsList=None, samplesList=None):
    '''returns the bigram matrix of the conditionsList and samplesList, with:
        conditions as the rows
        and the saples as columns
    Parameters:
    -----------
        df : conditional data frame (output of twoDimDict2DataFrame)
        conditionsList : list/np.array of conditions to read (None reads all)
        samplesList : list/np.array of samples to read (None reads all)
    Return:
    -------
        M : matrix representations of the conditions
        samps : labels of the columns of the matrix
        conds : labels of the rows of the matrix
    NOTICE that the matrix is transposed with respect to the df
    '''
    if conditionsList is None: conditionsList = df.columns
    if samplesList is None: samplesList = df.index
        
    bigrsDF = df[conditionsList].loc[samplesList]
    samps = bigrsDF.index
    conds = bigrsDF.columns
    M = bigrsDF.as_matrix().T # transpose to have consitions as rows
    return M, samps, conds
    
def bigramsDict2countsMatrix(bigramsDict, conditionsList=None, sampleList=None):
    '''2dim bigram counts dict --> bigrams matrix
    > M, samps, conds'''
    df = twoDimDict2DataFrame(bigramsDict)
    return bigramsdf2bigramsMatrix(df, conditionsList, sampleList)
    
def bigrams2countsMatrix(bigrams_tu, conditionsList=None, sampleList=None):
    '''bigrams --> bigrams matrix'''
    return bigramsDict2countsMatrix((bigrams2Dict(bigrams_tu)))
    

### + CONDITIONAL PROBABILITIES
    
def condFreqDictC2condProbDict(condFreqDict, conditions=None, samples=None):
    '''estimates the conditional probabilities (dict) form cond freq dist'''
    
    if conditions is None or samples is None:
        conditions = condFreqDict.keys()
        samples = condFreqDict.keys()
        
    cpd = nltk.ConditionalProbDist(condFreqDict, MLEProbDist)
    P = {}
    for cond in conditions:
        P[cond] = {}
        for samp in samples:
            P[cond][samp] = cpd[cond].prob(samp)
    return P    

def condProbDict2matrix(cpd, conditions, samples):
    '''
    return the matrix of consitional probabilities
    M, x_tick_labels, y_tick_labels
    '''
    return bigramsdf2bigramsMatrix(twoDimDict2DataFrame(cpd), 
                                   conditionsList=conditions, samplesList=samples)#, condsLi, samplesLi)
    
def condFreqDict2condProbMatrix(cfd, conditions, samples): 
    '''
    return the matrix of conditional probabilities
    > M, x_tick_labels, y_tick_labels
    '''
    cpd = condFreqDictC2condProbDict(cfd)
    return condProbDict2matrix(cpd, conditions, samples)


### + GENERAL

def twoDimDict2DataFrame(kykyDict):
    '''two key dict D[condition][sample] = x --> pandas dataframe
    P (sample = row |condition = column),
    columns are the condition, rows are sampes'''
    return pd.DataFrame(kykyDict).fillna(0)


#############################    LISTS AND ARRAYS    ##################################
    
def bigrams2matrix(A, n):
    """
    translates the bigram counts (tuple dictionary* ) into a bigram matrix 
    < A, tuple dictionary, *output of sequencesO_beta.bigramCounts
    < n, number of calls. len(i2c), len(c2i)
    > 2D matrix

    """
    #n = len(c2i)
    M = np.zeros((n,n))

    for c1 in A.keys():
        #print c1, c1[0], A[c1]
        M[c1[0], c1[1]] = A[c1]
    
    return M

def matrix2dict(M):
    """
    translates a matrix into a 2D tuple dictionary, inverse of bigrams2matrix
    < M, 2D numpy array
    > D, 2D tuple dictionary

    """
    nR, nL = np.shape(M)
    D = {}

    for i in np.arange(nR):
        for j in np.arange(nL):
            D[i, j] = M[i, j]
    
    return D


""" USE bigrams2matrix
def tup2mat(D, lab2ix):
    '''
    translates a tuple (2D) dictionary into a bigram matrix
    < tuple dict
    < labels dictionary, labs --> index of the matrix. This is important to tack back the elements of the matrix the labels
def tup2mat(D, lab2ix):
    '''

    n = len(lab2ix)
    M = np.zeros((n,n))
    #print M
    for c1 in D.keys():
         print c1[0], c1[1], type(c1[0]), lab2ix[c1[0]]#, lab2ix[c1[1]], "\nD:", D[c1]
         M[lab2ix[c1[0]], lab2ix[c1[1]]] = D[c1]
    return M
"""

###### BIGRAM MATRICES MAKEUP >>>>>>>>>>>>>>>>

def normalizeMatrix(M):
    '''
    normalizes the rows of a matrix
    < M, matrix
    > N, row normalized of M
    '''
    return np.divide(M.T, M.sum(axis=1)).T
    
def getElementsLargerThan(array1D, N_tresh, sort='yes'):
    """
    Index selector
    returns the indexes of the items with a value larger than N_tresh
    < array 1D is the array that we will use as a condition to sort the indexes
    * the elements can be sorted decedengly by:
      - the elements can be sorted descendingly with the values of array1D, or not
    """
    if(sort == 'yes'):
        """sorting elemets descendingly"""
        return np.array([i[0] for i in sorted(enumerate((array1D)), key = lambda x: x[1], reverse = True ) if i[1] > N_tresh])
        # enumerate, enumerates the elements in the list
        # then we sort them according to the second element, the value of the array
        # and filter out elemenst with a value smaller than N_tresh
    else:
        return np.array([i for i in range(len(array1D)) if array1D[i] > N_tresh]) #iterate over the elements of the array
    

def elements22Dindexes( items_idx ):
        """
        items to 2 dimentional index array
        converts the items indexes into 2dim indexes
        > i, row indexes
        > j, column indexes
        > Ndim, dimension of the new matrix
        """
        M_elements = list(it.product(*[items_idx, items_idx])) #returns a list of all the combinations of the given arrays
        i = np.array([item[0] for item in M_elements])
        j = np.array([item[1] for item in M_elements])
        Ndim = len(set(i))
        return (i, j, Ndim)
    
def reducedMatrix(A2D, items_idx):
    """
    returns the reduced matrix:
    < A2D -- matrix to reduce
    < items_idx -- items to keep, ordered (out of 'getElementsLargerThan()')
    """
    (i, j, Ndim) = elements22Dindexes(items_idx)
    #print("TEST", i, j, Ndim)
    return A2D[i, j].reshape(Ndim, Ndim)
   
def reduceDict(anti_di, items_idx):
    """
    returns the reduced
    di - dictionary with the call labels
    anti_di
    """
    red_di = {anti_di[item] : item for item in items_idx}
    return red_di


def call2mkUpIx(mkUp_arr, c2i, call):
    """
    == get makeup indexes from call ==
    When sellecting the calls to work with (make-upping), i.e. the most frequent
    whe work with the make up indexes that evaluate the i2c list in a way that
    the bigrams are sorted in the desired way (see getElementsLArgerThan()).
    However, sometimes we whant to mak up the indexes according to the 
    name of the call, i.e. eliminate the index that corresonds to tha pseudo-call
    __INI or __END. In this case we need to map the "call" back to the make-up
    index. and this is what this function.
    < mkUp_arr, 1D *numpy array* with the makeup-indexes
    < c2i, call to index *dictionary*
    < call, call *string*
    > $0 make up index, 
    > $1 the index that leads to the make-upped in the mkUp_arr
    """  
    if( np.sum(mkUp_arr == mkUp_arr[mkUp_arr == c2i[call]] )):
        "if the call is in the mkUP-indexes"        
        mkUpIx = mkUp_arr[mkUp_arr == c2i[call]][0]        
        return mkUpIx, np.where(mkUp_arr == mkUpIx)[0][0]
    else:
        print( "%s does't exists"%call)
        return None
        
def rmCallFromMkUpArr(mkUp_arr, c2i, call ):
    """
    remove call index form make-up array
    """
    ix = call2mkUpIx(mkUp_arr, c2i, call)
    if ix:
        return np.delete(mkUp_arr, ix[1])
    else:
        print("%s is not in the array\nIdentical array returned!"%call)
        return mkUp_arr


#########   hypothesis test    >>>>>>>>>>>>>>>>


def binomialProportions(i, j, M):
    """
    Binomial distribution (p1, p2) of the (i,j)-bigrams given the bigrams matrix.
    Where p1 is the numeber of times the bigram (i, j) was obseved and 
    p2 is the number to times a call differnt form j, followed i.
    p1 = #(i, j)
    p2 = #(i, not j)
    > (p1, p2)    
    """    
    return M[i, j], np.nansum(M[i, :]) - M[i, j]
    
def df_li2mtx(df, N, i):
    """
    take the i-th line of the a data frame with colums (n, m) (i.e. the 
    shuffled probabilities dataFrame) and transform it into a matrix
    < df, data frame with (n, m) column names (ex. shuffledProbDist_NPWVR-seqsData...)
    < N, number of call types. len(i2c) = len(c2i)
    < i, line of the data frame we want to take
    > bigrams matrix
    """
    df_li_str = dict(df.ix[i]) # 2-grams counts dict
    df_li_tu = {ast.literal_eval(ky): df_li_str[ky] for ky in df_li_str.keys()} # keys str --> tu
    return bigrams2matrix( df_li_tu, N ) # 2-grams matrix

     
########################################################
##################    BIGRAMS    #######################
########################################################


class bigramProbabilities():
    """
    reads the the matrix with the word frequencies and returns the bigram probabilities
    """

    def __init__(self, frequencyMatrix ):

        self.freqMatrix = frequencyMatrix
        self.bigramMatrix, self.Nbigrams = self.__lineNormalized( self.freqMatrix )
        self.repetitionP  = self.__repetition( self.bigramMatrix )#probability to get the same call
        self.differentP  = self.__different( self.bigramMatrix ) #probability to get a diff call
        self.mostProbableC = self.__mostProbable( self.bigramMatrix )
        self.MPrepetitionCallSet = self.__MPCrepetitionSet()
        self.MPdiffCallSet = self.__MPCdiffCallSet()
        self.numberOfBigrams = self.countBigrams(self.freqMatrix)
        #self.bigramsOccurrences = self.bigrams_occurrences(self.freqMatrix)
                

    def __lineNormalized(self, A):
        norm = np.nansum(A, axis = 1)
        return ( 1.0*A.T/norm ).T, norm

    def __colNormalized(self, A):
        norm = np.nansum(A, axis = 0)
        return ( 1.0*A/norm ), norm

    def __repetition(self, A):
        Nwords = len(A)
        rep = [A[i,i] for i in range(Nwords)]
        return np.asarray(rep)
 
    def __different(self, A):
        Nwords = len(A)
        diffc = [1-A[i,i] for i in range(Nwords)]
        return np.asarray(diffc)

    def __filterZeros(self, v):
        vz = []
        vdz = []
        for i in range(len(v)):
            if v[i] > 0:
                vdz.extend([i])
            else:
                vz.extend([i])
        return(vdz, vz)
               
    def noZeros(self, v):
       """
       returns the indexes of the vector v whose value is grater than zero
       """
       return self.__filterZeros(v)[0]

    def Zeros(self, v):
       """
       returns the indexes of the vector v whose value is lower than zero
       """
       return self.__filterZeros(v)[1]

    def __mostProbable(self, A):
        return np.argmax(A, axis = 1 )

    def __probableCalls(self, A):
        """
        for each call, returns the  indexes of the following calls\
        sorted from the most probable to the lest
        """
        a = [np.argsort(i)[::-1] for i in A] 
        return np.asarray(a)

    def getProbableCalls(self):
        """
        returns a the a vector with the indexes of the most prabable calls after a given call
        sorted form the most probable to the least
        """
        return self.__probableCalls(self.bigramMatrix)

    def subSetMoreThanN(self, n):
        """
        this method returns the indexes of the calls with more than n occurrences
        """       
        subbiG_i=[]
        for i in np.arange(len(self.bigramMatrix)):
            if self.Nbigrams[i] > n: # filter low frequency events
                subbiG_i.append(i)
        return subbiG_i

    def __MPCrepetitionSet(self):
        """
        retuns the set of calls whose MPC is the same as the previous call
        """
        subSet = [i for i in np.arange(len(self.mostProbableC)) if i == self.mostProbableC[i] ]
        return subSet

    def __MPCdiffCallSet(self):
        """
        retuns the set of calls whose MPC is the same as the previous call
        """
        subSet = [ i for i in  np.arange(len(self.mostProbableC)) if i != self.mostProbableC[i] ]
        return subSet

    def countBigrams(self, A):
        return A.sum()
    
    def bigrams_occurrences(self, A):
        return mt.countMatrixEntrances(A)

    def __reducedMatrix(self, A, n_tresh = 5):
        """
        this fiction returns a reduced for of the bigram matrix. Only those elemnts with less more that n_tresh counts.
        """
        
    def getElementsLargerThan(self, array1D, N_tresh):
        """
        returns the indexes of the items with a value larger than N_tresh
        """
        Mindx = [i for i in range(len(array1D)) if array1D[i] > N_tresh]
        return Mindx
    
    def elements22Dindexes(self, items_idx):
        """
        items to 2 dimentional index array
        converts the items indexes into 2dim indexes
        """
        M_elements = list(it.product(*[items_idx, items_idx]))
        i = [item[0] for item in M_elements]
        j = [item[1] for item in M_elements]
        Ndim = len(set(i))
        return (i,j, Ndim)
    
    def maskedMatrix(self, A2D, items_idx):#, index_array):
        """
        returns the reduced matrix
        """
        (i,j,Ndim) = elements22Dindexes(items_idx)
        return A2D[i, j].reshape(Ndim, Ndim)



'''
class chart2sec:
    """
    this objec transform a chart into a sequence archive
    """
    def __init__(self, inF, cut_time):
        print "creatinng a chart object"
    
        f = open(inF, 'r')
        str1 = file.read(f) # reading file
        newStr = str1.rstrip() # remove last \n

    def __dayTapeDictIni(self):


### INICIALIZATIONS
        dayTape = {}
        
### READ FILE
        lines = str1.split('\n') # vector of lines
        Nlines = len(lines)
        tStamp = np.zeros([Nlines,1])
        
        lIndex = 0
        for j in lines: #inicialize dictionary, leve put the last element
            col = j.split('\t') # vector of the elements in the line       
            if(len(col)>1):
                dayTapeL = ''.join([col[2],col[1]])
                dayTape[dayTapeL] = []
                hora, minuto, segundo = col[0].split('_')
                tStamp[lIndex] = int(hora)*60*60 + int(minuto)*60 + int(segundo)
                lIndex += 1
            #print int(hora), int(minuto), segundo
            #timeStamp[lIndex] = 
            #print dayTapeL
            else:
                lIndex += 1
                print j
                
                lIndex = 0
                for j in lines: #create dictionary with the elements with the same day and tape label
                    col = j.split('\t') # vector of the elements in the line
                    
                    if(len(col)>1):
                        dayTapeL = ''.join([col[2],col[1]])
                        dayTape[dayTapeL].append(lIndex)
                        lIndex+=1
                    else:
                        lIndex+=1
                        print j


class bigram:
    """
    Creates an object with pca properties
    Inicialize it with an array 
    """

    def __init__(self, data, constrain = ["day", "tape"]):
       """
       creates an object with the PCA relevant properties
       """        
       self.__data = data
       [wg , vec] = self.full_pca(self.__data)
       self.__weights = wg
       self.__vectors = vec
       Dim = len(wg) + 1
       self.colorVec=['b','g','r','c','m','y','k','w']

'''

'''
    def full_pca(self, data):

        """
        Performs the complete eigendescomposition of the covatiance matrix
        traslading the data so that <x>=0
        """
       # print "sin trasladar los datos a media cero"
        cov = np.cov(data.T)
#        cov = np.cov(data.transpose())
        w,u = eigh(cov, overwrite_a = True)
        return w[::-1], u[:,::-1]    

    def weights(self):
        return self.__weights

    def vectors(self):
        return self.__vectors

    def dataInMyBasis(self, data0, basisDim = -1 ):
        """
        this function gives the data in the main features PCA basis
        args:
        * data0 =  the matrix with te data
        * basisDim =  number by default we set it to the complete basis
        """
        fV = self.__vectors[:, 0:basisDim]        
        nD = np.dot(data0, fV )
        #print "data0=", np.shape(data0), "data1", np.shape(fV)
        return nD
    
   

    def featureVector(self, Ifraction):
        """
        gets the feature basis, the vectors whos weight is above th
        args:
        * inverse of fraction the basis vector we want to keep (>1)
        """

        if Ifraction > 1:
            Ifraction = 1/Ifraction

        PCDim =  len(self.__weights)*Ifraction
        print "\nreturning a", PCDim, "dimensional basis\n"
        return self.__vectors[:,:PCDim]


    def dataInPCBasis(self, data0, basisDim = dim): #self.__dim):
        """
        this function gives the data in the main features PCA basis
        args:
        * data0 =  the matrix with te data
        * basisDim =  number of elements
        """
        nD = np.dot(data0, self.__vectors[:, 0:basisDim-1] )
        #print "data0=", np.shape(data0), "data1", np.shape(fV)
        return nD

    def newDataOld(self,Ifraction):
        """
        this function gives the data in the main features PCA basis
        """
        
        fV = self.featureVector(Ifraction)
        nDt = np.dot( np.transpose(fV), np.transpose(self.__data) )
        nD = np.dot(self.__data, fV )
#        print nDt.T-nD
        return nD

    def fullNewData(self):
        """
        returns all the data in the PC basis
        """
        ffV = self.__vectors
        fnD = np.dot(np.transpose(ffV), np.transpose(self.__data))
        return fnD

    def multiBarPl(self, plMatrix, plScriptName): ### under construction
        """
        plMAtrix has nD rows and nF columns
        nD = dimension of the complete space
        nF = number of features 
        """
        myfile = open(plScriptName, 'w')
        myfile.write("#!/usr/bin/python\n\n")
        myfile.write("import pylab as pl\n\n")
        nD, nF = plMatrix.shape

        for i in np.arange(nF):
            myfile.write("pl.subplot(%d,1,%d)\n" %(nF, i+1))

            x = np.arange(len(plMatrix[:,i]))
            print "equis: ", x
            myfile.write("pl.bar") #, plMatrix[:,i] ))

    def nLargestStdValuesAndIdx(self, n):
        """
        Args:
        * n, the n largest elemts of the 1 dim array (obje 

ct)
        Returns:
        * x,y = the largest values and their indexes
        """
        std_dev = self.__data.std(axis=0)
        my_nLargest = np.sort(std_dev)[-n:]
        my_nLargest_idx = np.where( std_dev >= my_nLargest[0] )
        return my_nLargest, my_nLargest_idx

    def plInMyBasis(self, TheDatas, plot_name):
        
        for i in np.arange(len(TheDatas)):
            newData = self.dataInMyBasis( TheDatas[i], 2 )
            pl.plot(newData[:,0], newData[:,1], "%so"% self.colorVec[ i%len(self.colorVec) ], label = '%d'%i, alpha=0.4 )
         #   print "i", i, "de", len(TheDatas),"\ncolor:" , "%so"% self.colorVec[ i%len(self.colorVec) ], label='data%d'%i,"\nnewdata" , np.shape(newData)
            
        pl.legend(loc='upper right')
        pl.savefig(plot_name+".ps")
        pl.clf()
        print "%s.ps"%plot_name


    def plInMyBasis0(self, TheDatasNames, plot_name):
### plots the data in the self basis
        
        TheDatas=[]
        print "\nLEN", len(TheDatasNames), TheDatasNames, "\n "

        for i in np.arange(len(TheDatasNames)):
            print i,"file name", TheDatasNames[i], "\n "
            TheDatas.append( np.loadtxt( TheDatasNames[i] ) )
            newData = self.dataInMyBasis( TheDatas[i], 2 )
            
            pl.plot(newData[:,0], newData[:,1], "%so"% self.colorVec[ i%len(self.colorVec) ], label = TheDatasNames[i].split('/')[-2], alpha=0.4 )

        pl.title(TheDatasNames[i].split('/')[-1])         
        pl.legend(loc='lower right')
        pl.savefig(plot_name+".eps")
        pl.clf()
        
        print "%s.eps"%plot_name
#        return myPCApl

    def saveDataInMyBasis(self, TheDatasNames, file_name, basis_dimension):
        """ save the data in the self basis """
        
        TheDatas=[]
        newData=[]
      #  print "\nLEN", len(TheDatasNames), TheDatasNames, "\n "

        for i in np.arange(len(TheDatasNames)):
            print i, TheDatasNames[i], "\n "
            TheDatas.append( np.loadtxt( TheDatasNames[i] ) )
            newData = self.dataInMyBasis( TheDatas[i], basis_dimension) 

            data_name = TheDatasNames[i].split('/')[-2]
            out_name = file_name+data_name+'dim_'+str(basis_dimension)+'.dat'
            print "out file: %s.ps"%out_name
            np.savetxt(out_name, newData) 
            del newData
            

class pca0mean(pca):

    """
    Creates a subobject with pca properties
    Inicialize it with an array 
    """

    def __init__(self, data):
        """
        creates an object with the PCA relevant properties
        """  
	pca.__init__(self, self.z_mean(data))#, z_mean(self,self.__data)):	      
        print "zero mean \n" 
#        print "you created a PCA object from\
 #a",np.shape(data),"dimensional data\n zero mean\n", 
   #     [wg , vec] = self.full_pca(self.__data)
  #      self.__weights = wg
 #       self.__vectors = vec
        
   
 def z_mean(self,data):
#    def z_mean(self):

        """
        this method traslades the data so that the mean of each columnn is zero
        """
        mu = data.mean(axis=0)
        data0 = data-mu
        return data0

 

'''
'''
def repetitionPL(self, A):
nC = np.shape(A)[1]
P = np.zeros(nC)
N = np.zeros(nC)
for i in range(nC):
if A[:,i].sum() != 0:
P[i] = 1.0*A[i,i]/A[:,i].sum()
N[i] = A[:,i].sum()
else:
print i, "zero L"            
return P, N

def diffPR(self, A):
nC = np.shape(A)[1]
P = np.zeros(nC)
for i in range(nC):
if A[:,i].sum() != 0:
P[i] = 1 - 1.0*A[i,i]/A[:,i].sum()
else:
P[i] = 0

return P

def diffPL(self, A):
nL = np.shape(A)[0]
P = np.zeros(nL)
for i in range(nL):
if A[i,:].sum() != 0:
P[i] = 1 - 1.0*A[i,i]/A[i,:].sum()
return P

'''
'''

class bigram:
    """
    this class reads a sequence file and return the unnormalized bigram matrix
    """
    def __init__(self, fName):
        print "bigram object"
        f = open(fName, 'r')
        fStr = file.read(f)
        fStr = fStr.strip() #chomp
        self.sentences = fStr.split('\n')
        self.words = self.vocabulary().keys() # index to word
        self.vocabularySize = len(self.words)
        self.index = self.word2Index() # word to index
        self.fullBiG = self.fullBiGMatrix()

    def vocabulary(self):
        """
        creates a dictionary of the words and sets to it the frequency
        """
        vocFrec = {}
        vocFrec['0INI'] = 0
        vocFrec['0FIN'] = 0

        for i in self.sentences: # inzialize dictionary
            palabras = i.split(' ')
            for j in palabras:
                vocFrec[j]=0
                
        # WORD FREQUENCY
        for i in self.sentences:
            vocFrec['0INI']+=1
            vocFrec['0FIN']+=1
            palabras = i.split(' ')
            for j in palabras:
                vocFrec[j]+=1
                
        return vocFrec

    def word2Index(self):
        """
        creates a maping from the call to the bigram matrix index
        """
        w2i = { self.words[j] : j  for j in range(self.vocabularySize)} #inicialize dictionary
        return w2i

 m   def fullBiGMatrix(self):
        """
        returns the bigram matrix
        """
        #  INICIALIZE 2-GRAM DICTIONARY

        print "full bigram"
        A = np.zeros((self.vocabularySize, self.vocabularySize))
#        adj2G = []
        
        for i in self.sentences:
            palabras = i.split(' ')
            preWord = '0INI'

            for j in palabras:
               # print j, self.index[j], preWord
                A[ self.index[preWord], self.index[j] ] += 1
                preWord = j
        
            A[ self.index[preWord], self.index['0FIN'] ] += 1
                
        return  A


    def BiGMatrix(self):
        """
        returns the bigram matrix effective unormalized bigram matrix
        having zeroed the INI line and the FIN column
        """        
        print "effective bigram"
        
        #  INICIALIZE 2-GRAM DICTIONARY
        biGmat = 1*self.fullBiG #multiply for one so that it creates a new memory instance. Remember that we equaling to variables, python asigns the same memory space. So eveting that is done to one of the variable will also apply for the other one.
        biGmat[self.index["0INI"], :] = 0
        biGmat[:, self.index["0FIN"]] = 0
                
        return  biGmat



'''



