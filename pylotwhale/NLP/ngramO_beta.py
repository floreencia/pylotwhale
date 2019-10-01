from __future__ import print_function, division

import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from collections import Counter, defaultdict

import ast
import pandas as pd
import nltk
from nltk.probability import ConditionalProbDist, MLEProbDist

import pylotwhale.utils.dataTools as daT
import pylotwhale.NLP.annotations_analyser as aa

import pylotwhale.utils.matrixTools as mT

"""
    Module for doing statistics on ngrams; counting ngrams, conditional frequencies
    and conditional probabilities
    florencia @ 16.05.14
"""


############################    PLOTTING    ################################


def barPltsSv(
    y,
    labs,
    figN="",
    figSz=(10, 3),
    yL="# bigrams",
    plTit="",
    plLegend=0,
    maxNyTicks=0,
    yTickStep=50,
):
    """
    plots stocked histograms
    i.e. the number of bigrams repetitions/differences
    y can be an array of the bar values of the plot
    * plLegend list with legends of the plots, i.e ['different call', 'repetition'] 
    * maxNyTicks - maximum number of y ticks. 0- default (matplotlib decides)
        not working
    """

    colors = ["b", "g", "r", "c", "m", "y", "k"]
    fig = plt.figure(figsize=figSz)
    ax = fig.add_subplot(111)
    y0 = np.zeros(len(y[0]))
    p = []
    maxY = 0  # for adjusting the y scale
    for i in range(len(y)):
        assert len(y0) == len(y[i])  # all the arr must have the same dimension
        p.append(ax.bar(range(len(y[i])), y[i], color=colors[i], bottom=y0))
        y0 = y[i]
        maxY += y0[0]

    ax.set_xticks(np.arange(len(y0)) + 0.5)
    ax.set_xticklabels(labs, rotation="vertical")
    ax.set_ylabel(yL)
    if plTit:
        ax.set_title(plTit)  # , size = 16)

    # print np.nanmax(y), y0
    if yTickStep:
        ax.set_yticks(np.arange(0, int(maxY), yTickStep))

    if isinstance(plLegend, list):
        if len(plLegend) == len(p):  # set labels
            ax.legend(tuple(p), tuple(plLegend))
            # print "LABELS:", plLegend

    if figN:
        fig.savefig(figN, bbox_inches="tight")
        # print figN


##### NLTK - pandas - related ngram code  #####


def strSeq2bigrams(strSeq):
    """decomposes a sequence of strings into its bigrmas
    Parameters
    ----------
    strSeq: list
        list of strings, eg. ['b', 'a', 'c']
    """
    return list(nltk.bigrams(strSeq))


def bigrams2Dict(bigrams_tu):
    """
    DEPRECATED
    USE bigrams2cfd
    converts a 2D-tuples list into a conditionalFreqDist (~2D-dictionary)
    eg. [(a,b) ... ] --> Di[a][b] = #(a,b)
    :bigrams_tu: bigrams as a list of tuples
    """
    cfd = nltk.ConditionalFreqDist(bigrams_tu)
    return cfd


def bigrams2cfd(bigrams_tu):
    """
    converts a list of 2D-tuples into a nltk.conditionalFreqDist (~2D-dictionary)
    eg. [(a,b) ... ] --> Di[a][b] = #(a,b)
    Parameters
    ----------
    bigrams_tu : bigrams as a list of tuples
    Returns
    -------
    cfd : nltk.ConditionalFreqDist(
    """
    cfd = nltk.ConditionalFreqDist(bigrams_tu)
    return cfd


def bigramsdf2bigramsMatrix(df, conditionsList=None, samplesList=None):
    """returns the bigram matrix of the conditionsList and samplesList, with:
        conditions as the rows
        and the samples as columns
    Parameters
    -----------
    df : conditional data frame (output of kykyDict2DataFrame)
    conditionsList : list/np.array of conditions to read (None reads all)
    samplesList : list/np.array of samples to read (None reads all)
    Returns
    -------
    M : matrix representations of the df values
    conds : labels of the rows of the matrix
    samps : labels of the columns of the matrix
    """
    if conditionsList is None:
        conditionsList = df.columns
    if samplesList is None:
        samplesList = df.index

    bigrsDF = df[samplesList].reindex(conditionsList)
    conds = bigrsDF.index.values
    samps = bigrsDF.columns.values
    M = bigrsDF.as_matrix()  # .T  # transpose to have conditions as rows
    return M, conds, samps


def bigramsDict2countsMatrix(bigramsDict, conditionsList=None, samplesList=None):
    """DEPRECATED, USE kykyCountsDict2matrix INSTEAD"""
    return kykyCountsDict2matrix(bigramsDict, conditions=conditionsList, samples=samplesList)


def cfdBigrams2countsMatrix(bigramsDict, conditionsList=None, samplesList=None):
    """
    DEPRECATED! use kykyCountsDict2matrix    
    
    2dim bigram counts dict --> bigrams matrix
    Parameters
    ----------
    bigramsDict : nltk.ConditionalFreqDist
        two entry dict,  eg. D[a][b]
    conditionsList : list/np.array of conditions to read (None reads all)
    samplesList : list/np.array of samples to read (None reads all)
    Returns
    -------
    M : matrix representations of the frequency values, with:
            conditions as the rows and the samples as columns
    conds : labels of the rows of the matrix
    samps : labels of the columns of the matrix
    """
    ## copy values of the cfd not to affect the mutable cfd outside
    bigramsD = dict(bigramsDict)
    # When given conditionsList and/or samplesList
    # make sure all keys are present in bigramsDict
    if conditionsList is not None or samplesList is not None:
        kySet = set(conditionsList) | set(samplesList)  # union
        bigramsD = fill2KyDict(bigramsD, kySet)  # fill missing keys with nan
        # print('filling missing keys')

    ## convert 2ble-ky-dictionary into dataFrame
    df = kykyDict2DataFrame(bigramsD)
    return bigramsdf2bigramsMatrix(df, conditionsList, samplesList)


def bigrams2countsMatrix(bigrams_tu, conditionsList=None, samplesList=None):
    """bigrams --> bigrams matrix"""
    return kykyDict2matrix(
        (bigrams2cfd(bigrams_tu)), conditionsList=conditionsList, samplesList=samplesList
    )


### matrix <--> samps index utilities for H0


def get_insignificant_bigrams(
    p_values, samps, conds, pc=0.1, condition=lambda p_val, pc: p_val > pc
):
    """Get bigrams that violate the null hypothesis
    Parameters
    ----------
    p_values : 2darray
        bigram's p-values (probability that H0 is true)
    samps : list like
    conds : list like
    returns a list of tuples with the bigrams that cannot reject H0"""
    bigrams_list = []

    for (r, c), p_val in np.ndenumerate(p_values):
        if condition(p_values[r, c], pc):  # p_values[r, c] > pc:
            bigrams_list.append((conds[r], samps[c]))
    return bigrams_list


def get_insignificant_bigram_indices(p_values, pc=0.1):
    """Get null hypothesis' non rejecting bigram indices
    Parameters
    ----------
    p_values: 2darray
        bigram's p-values (probability that H0 is true)
    returns a list of tuples with the bigrams indices that cannot reject H0
        ([r, c), ...]
        r = samp index
        c = cond indes
    """
    index_list = []

    for (r, c), p_val in np.ndenumerate(p_values):
        if p_values[r, c] > pc:
            index_list.append((r, c))
    return index_list


### + CONDITIONAL PROBABILITIES


def condFreqDictC2condProbDict(condFreqDict, conditions=None, samples=None):
    """estimates the conditional probabilities (dict) form cond freq dist"""

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


def kykyDict2matrix(kykyDict, conditions, samples):
    """
    return the matrix of conditional probabilities
    Parameters
    ----------
    kykyDict : dict
        dictionary of counts
    conditions, samples : list of strings
        keys of the dictionary that are considered in the matrix
    Returns
    -------
    M : 2darray
        values
    conds : labels of the rows of the matrix
    samps : labels of the columns of the matrix
    """
    df = kykyDict2DataFrame(kykyDict)
    return bigramsdf2bigramsMatrix(df, conditionsList=conditions, samplesList=samples)


def kykyCountsDict2matrix(kykyDict, conditions, samples):
    """
    DEPRECATED, use kykyDict2matrix
    """
    df = kykyDict2DataFrame(kykyDict)
    return bigramsdf2bigramsMatrix(df, conditionsList=conditions, samplesList=samples)


def condProbDict2matrix(cpd, conditions, samples):
    """
    DEPRECATED USE: kykyDict2matrix
    return the matrix of conditional probabilities
    Parameters
    ----------
    cpd: nltk.conditional_probability_distribution
    M, x_tick_labels, y_tick_labels
    """
    return bigramsdf2bigramsMatrix(
        kykyDict2DataFrame(cpd), conditionsList=conditions, samplesList=samples
    )  # , condsLi, samplesLi)


def condFreqDict2condProbMatrix(cfd, conditions, samples):
    """
    return the matrix of conditional probabilities
    Parameters
    ----------
    cfd: nltk.conditional_frequency_distribution
    > M, x_tick_labels, y_tick_labels
    """
    cpd = condFreqDictC2condProbDict(cfd)
    return kykyCountsDict2matrix(cpd, conditions, samples)


### + GENERAL


def kykyDict2DataFrame(kykyDict, fillna=0):
    """
    Transforms kyky dictionary into pandas dataframe

    Parameters
    ----------
    kykyDict : dict
        dict of counts dict, e.g. cfd or a cpd
        D[condition][sample] = n
        P (sample = row | condition = column),
        columns are the condition, rows are samples
    fillna : int, float, str
        value used to fill empty cells
    Returns
    -------
    df : pandas DataFrame
        df.loc[cond, samp] = kykyDict[cond][samp]
    """
    # T ranspose the DataFrame to have the conditions as rows (index of the df)
    # and the saples as columns
    return pd.DataFrame(kykyDict).T.fillna(fillna)


def kykyCountsDict2DataFrame(kykyDict, fillna=0):
    """
    DEPRECATED, use kykyDict2DataFrame
    """
    return kykyDict2DataFrame(kykyDict)


def matrix2kykyDict(M, rows, columns):
    """converts matrix (M) into kyky dictionary (DICT)
    where the rows (r) and columns (c) of M are mapped into DICT
    M[r, c] = DICT[r][c]
    """
    df = matrix2DataFrame(M, rows=rows, columns=columns)
    return DataFrame2kykyDict(df)


def DataFrame2kykyDict(df):
    """converts pandas DataFrame into kykyDict
    df: DataFrame
        conditions as index and
        samples as rows
    """
    # transpose the DataFrame so that when converting to dictionary
    # the first key corresponds to the condition and the second to the sample
    # DICT[cond][samp] = df.loc[cond, sample]
    thisdf = df.T
    return thisdf.to_dict()


def matrix2DataFrame(M, rows=None, columns=None):
    """converts an 2daray into a pandas DataFrame
    Parameters
    ----------
    M : 2d array
        rows = conditions and columns = samples
    rows, columns : list like
        indices (r) of the matrix, conditions
        names of the columns (c), samples
    """
    return pd.DataFrame(M, columns=columns, index=rows)


def twoDimDict2DataFrame(kykyDict):
    """
    DEPRECATED, USE kykyDict2DataFrame
    """
    return pd.DataFrame(kykyDict).fillna(0)


def fill2KyDict(kykyDict, kySet):
    """fills a conditional frequency distribution with keys
    Parameters
    ----------
    kykyDict : nltk.probability.FreqDist
    kySey : set with the keys that
    """
    missingSet = set(kySet) - set(kykyDict.keys()).intersection(set(kySet))
    # print(missingSet)
    for ky in missingSet:
        kykyDict[ky] = nltk.FreqDist()
    return kykyDict


### bigrams and time


def dictOfBigramIcTimes(listOfBigrams, df, ict_XY_l=None, label="call", ict_label="ict"):
    """searches sequences (listOfBigrams) of type <label> in the dataframe and returns a 
    dictionary with the ict_label values of the sequences
    Parameters
    ----------
    listOfBigrams : list of bigrams
    df : pandas data frame
    ict_XY_l : dictionary of lists
        bigrams as keys and the ict of the bigrams as values
    label : type of sequence, or name of the column in df where to look for the sequences
    ict_label : str
        name of the column
    Return
    ------
    ict_XY : dictionary with lists
        ICIs by bigram
    """
    if ict_XY_l is None:
        ict_XY_l = defaultdict(list)
    for seq in listOfBigrams:
        try:
            seqdf = daT.returnSequenceDf(df, seq, label="call")
        except ValueError:  # sequence not found, continue with the next seq
            continue
        ky = "".join(seq)  # seqdf.head(20)
        ict_XY_l[ky].extend(seqdf[ict_label].values)

    return ict_XY_l


def ICI_XY_list2array(ici_XY_l):
    ### transform ICI list into ICI numpy array filtering nans and infs
    ici_XY = {}
    for k in ici_XY_l.keys():
        arr = np.array(ici_XY_l[k])
        ici_XY[k] = arr[np.isfinite(arr)]
    return ici_XY


def dfDict2dictOfBigramIcTimes(dfDict, listOfBigrams, ict_XY=None, label="call", ict_label="ict"):
    """ICI of the bigrams
    searches sequences (listOfBigrams) in the dataframes from dfDict"""
    for thisdf in dfDict.values():
        ict_XY = dictOfBigramIcTimes(
            listOfBigrams, thisdf, ict_XY_l=ict_XY, label=label, ict_label=ict_label
        )
    return ict_XY


def selectBigramsAround_dt(ictDict, dt=None, minCts=10, metric=np.median):
    """takes a dictionary of ict-bigrams (ict) and returns the keys of the elements 
    with at least <minCts> counts within the dt interval
    Parameters
    ----------
    ictDict: dict of lists
    df: 2d-tuple
    minCts: int
    metric: callable
    returns a list with the keys of ictDict
    """
    if dt is None:
        dt = (None, np.inf)
    collector = []
    ict_mean = dict(
        [(item, metric(ictDict[item])) for item in ictDict.keys() if len(ictDict[item]) >= minCts]
    )
    for ky in ict_mean.keys():
        if ict_mean[ky] > dt[0] and ict_mean[ky] < dt[1]:
            collector.append(ky)
    return collector


def dfDict_to_bigram_matrix(
    df_dict,
    Dtint,
    timeLabel="ici",
    callLabel="call",
    startTag="_ini",
    endTag="_end",
    return_values="probs",
    minCalls=1,
):
    """Bigrams counts/probs as matrix from DataFrame
    Parameters
    ----------
    df_dict: DataFrame
    Dtint: tuple
    timeLabel: str
    callLabel: str
    return_values: str
        {probs, counts}
    Returns
    -------
    (matrix, sampsLi, condsLi)
    """
    cfd = nltk.ConditionalFreqDist()  # initialise cond freq dist
    calls0 = []
    for t in df_dict.keys():  # for reach tape
        thisdf = df_dict[t]
        # define the sequences
        sequences = aa.seqsLi2iniEndSeq(
            aa.df2listOfSeqs(thisdf, Dt=Dtint, l=callLabel, time_param=timeLabel),
            ini=startTag,
            end=endTag,
        )
        my_bigrams = nltk.bigrams(sequences)  # tag bigrams
        cfd += bigrams2cfd(my_bigrams)  # count bigrams
        calls0 += list(thisdf[callLabel].values)

    # calls order
    calls = [
        item[0]
        for item in sorted(Counter(calls0).items(), key=lambda x: x[1], reverse=True)
        if item[1] >= minCalls
    ]  # order calls
    samplesLi = calls[:] + [endTag]  # None #[ 'A', 'B', 'C', 'E', '_ini','_end']
    condsLi = calls[:] + [startTag]

    if return_values == "counts":
        return kykyCountsDict2matrix(cfd, condsLi, samplesLi)

    if return_values == "probs":
        cpd = condFreqDictC2condProbDict(cfd)  # , condsLi, samplesLi)
        return kykyCountsDict2matrix(cpd, condsLi, samplesLi)


#############################    LISTS AND ARRAYS    ##################################


def bigrams2matrix(A, n):
    """
    translates the bigram counts (tuple dictionary* ) into a bigram matrix 
    < A, tuple dictionary, *output of sequencesO_beta.bigramCounts
    < n, number of calls. len(i2c), len(c2i)
    > 2D matrix

    """
    # n = len(c2i)
    M = np.zeros((n, n))

    for c1 in A.keys():
        # print c1, c1[0], A[c1]
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
    """
    normalizes the rows of a matrix
    < M, matrix
    > N, row normalized of M
    """
    return np.divide(M.T, M.sum(axis=1)).T


def getElementsLargerThan(array1D, N_tresh, sort="yes"):
    """
    Index selector
    returns the indexes of the items with a value larger than N_tresh
    < array 1D is the array that we will use as a condition to sort the indexes
    * the elements can be sorted descendingly by:
      - the elements can be sorted descendingly with the values of array1D, or not
    """
    if sort == "yes":
        """sorting elemets descendingly"""
        return np.array(
            [
                i[0]
                for i in sorted(enumerate((array1D)), key=lambda x: x[1], reverse=True)
                if i[1] > N_tresh
            ]
        )
        # enumerate, enumerates the elements in the list
        # then we sort them according to the second element, the value of the array
        # and filter out elements with a value smaller than N_tresh
    else:
        return np.array(
            [i for i in range(len(array1D)) if array1D[i] > N_tresh]
        )  # iterate over the elements of the array


def elements22Dindexes(items_idx):
    """
        items to 2 dimensional index array
        converts the items indexes into 2dim indexes
        > i, row indexes
        > j, column indexes
        > Ndim, dimension of the new matrix
        """
    M_elements = list(
        it.product(*[items_idx, items_idx])
    )  # returns a list of all the combinations of the given arrays
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
    # print("TEST", i, j, Ndim)
    return A2D[i, j].reshape(Ndim, Ndim)


def reduceDict(anti_di, items_idx):
    """
    returns the reduced
    di - dictionary with the call labels
    anti_di
    """
    red_di = {anti_di[item]: item for item in items_idx}
    return red_di


def call2mkUpIx(mkUp_arr, c2i, call):
    """
    == get makeup indexes from call ==
    When selecting the calls to work with (make-upping)
    bigrams are sorted in the desired way (see getElementsLargerThan()).
    However, sometimes we want to make up the indexes according to the 
    name of the call, i.e. eliminate the index that corresponds to the pseudo-call
    __INI or __END. In this case we need to map the "call" back to the make-up
    index. and this is what this function.
    < mkUp_arr, 1D *numpy array* with the makeup-indexes
    < c2i, call to index *dictionary*
    < call, call *string*
    > $0 make up index, 
    > $1 the index that leads to the make-upped in the mkUp_arr
    """
    if np.sum(mkUp_arr == mkUp_arr[mkUp_arr == c2i[call]]):
        "if the call is in the mkUP-indexes"
        mkUpIx = mkUp_arr[mkUp_arr == c2i[call]][0]
        return mkUpIx, np.where(mkUp_arr == mkUpIx)[0][0]
    else:
        print("%s does't exists" % call)
        return None


def rmCallFromMkUpArr(mkUp_arr, c2i, call):
    """
    remove call index form make-up array
    """
    ix = call2mkUpIx(mkUp_arr, c2i, call)
    if ix:
        return np.delete(mkUp_arr, ix[1])
    else:
        print("%s is not in the array\nIdentical array returned!" % call)
        return mkUp_arr


#########   hypothesis test    >>>>>>>>>>>>>>>>


def binomialProportions(i, j, M):
    """
    Binomial distribution (p1, p2) of the (i,j)-bigrams given the bigrams matrix.
    Where p1 is the number of times the bigram (i, j) was observed and 
    p2 is the number to times a call different form j, followed i.
    p1 = #(i, j)
    p2 = #(i, not j)
    > (p1, p2)    
    """
    return M[i, j], np.nansum(M[i, :]) - M[i, j]


def df_li2mtx(df, N, i):
    """
    take the i-th line of the a data frame with columns (n, m) (i.e. the 
    shuffled probabilities dataFrame) and transform it into a matrix
    < df, data frame with (n, m) column names (ex. shuffledProbDist_NPWVR-seqsData...)
    < N, number of call types. len(i2c) = len(c2i)
    < i, line of the data frame we want to take
    > bigrams matrix
    """
    df_li_str = dict(df.ix[i])  # 2-grams counts dict
    df_li_tu = {ast.literal_eval(ky): df_li_str[ky] for ky in df_li_str.keys()}  # keys str --> tu
    return bigrams2matrix(df_li_tu, N)  # 2-grams matrix


########################################################
##################    BIGRAMS    #######################
########################################################


class bigramProbabilities:
    """
    reads the the matrix with the word frequencies and returns the bigram probabilities
    """

    def __init__(self, frequencyMatrix):

        self.freqMatrix = frequencyMatrix
        self.bigramMatrix, self.Nbigrams = self.__lineNormalized(self.freqMatrix)
        self.repetitionP = self.__repetition(self.bigramMatrix)  # probability to get the same call
        self.differentP = self.__different(self.bigramMatrix)  # probability to get a diff call
        self.mostProbableC = self.__mostProbable(self.bigramMatrix)
        self.MPrepetitionCallSet = self.__MPCrepetitionSet()
        self.MPdiffCallSet = self.__MPCdiffCallSet()
        self.numberOfBigrams = self.countBigrams(self.freqMatrix)

    def __lineNormalized(self, A):
        norm = np.nansum(A, axis=1)
        return (1.0 * A.T / norm).T, norm

    def __colNormalized(self, A):
        norm = np.nansum(A, axis=0)
        return (1.0 * A / norm), norm

    def __repetition(self, A):
        Nwords = len(A)
        rep = [A[i, i] for i in range(Nwords)]
        return np.asarray(rep)

    def __different(self, A):
        Nwords = len(A)
        diffc = [1 - A[i, i] for i in range(Nwords)]
        return np.asarray(diffc)

    def __filterZeros(self, v):
        vz = []
        vdz = []
        for i in range(len(v)):
            if v[i] > 0:
                vdz.extend([i])
            else:
                vz.extend([i])
        return (vdz, vz)

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
        return np.argmax(A, axis=1)

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
        subbiG_i = []
        for i in np.arange(len(self.bigramMatrix)):
            if self.Nbigrams[i] > n:  # filter low frequency events
                subbiG_i.append(i)
        return subbiG_i

    def __MPCrepetitionSet(self):
        """
        returns the set of calls whose MPC is the same as the previous call
        """
        subSet = [i for i in np.arange(len(self.mostProbableC)) if i == self.mostProbableC[i]]
        return subSet

    def __MPCdiffCallSet(self):
        """
        returns the set of calls whose MPC is the same as the previous call
        """
        subSet = [i for i in np.arange(len(self.mostProbableC)) if i != self.mostProbableC[i]]
        return subSet

    def countBigrams(self, A):
        return A.sum()

    def bigrams_occurrences(self, A):
        return mT.countMatrixEntrances(A)

    def __reducedMatrix(self, A, n_tresh=5):
        """
        this fiction returns a reduced for of the bigram matrix. Only those elements with less more that n_tresh counts.
        """

    def getElementsLargerThan(self, array1D, N_tresh):
        """
        returns the indexes of the items with a value larger than N_tresh
        """
        Mindx = [i for i in range(len(array1D)) if array1D[i] > N_tresh]
        return Mindx

    def elements22Dindexes(self, items_idx):
        """
        items to 2 dimensional index array
        converts the items indexes into 2dim indexes
        """
        M_elements = list(it.product(*[items_idx, items_idx]))
        i = [item[0] for item in M_elements]
        j = [item[1] for item in M_elements]
        Ndim = len(set(i))
        return (i, j, Ndim)

    def maskedMatrix(self, A2D, items_idx):  # , index_array):
        """
        returns the reduced matrix
        """
        (i, j, Ndim) = elements22Dindexes(items_idx)
        return A2D[i, j].reshape(Ndim, Ndim)
