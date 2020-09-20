# -*- coding: utf-8 -*-
"""
Functions for manipulating data (pandas dataframes, dictionaries,
lists): sorting, splitting, searching
"""

import numpy as np
from collections import Counter


def arrShuffleSplit(arr, frac=0.5):
    """samples an array into two arrays shuffling the elements"""
    n = len(arr)
    indices = np.arange(n)
    np.random.shuffle(indices)
    idx = int(n*0.5)
    return arr[indices[:idx]], arr[indices[idx:]]


#### DATAFRAMES

def dictOfGroupedDataFrames(df0, groupingKey='tape'):
    '''groups a dataframe according to groupingKey
    and returns a dictionary of data frames'''
    df_dict = {}
    keysSet = set(df0[groupingKey])
    for t in keysSet:
        df_dict[t] = df0[df0[groupingKey] == t].reset_index(drop=True)
    return df_dict


def subsetdf(df, cs):
    '''subset dataframe
    Parameters
    -----------
    df: pandas dataframe
    cs: dict
        constrains
        cs = {'sex': 'm', 'predator': 'leo'}
    Return
    ------
    constrained dataframe
    '''
    assert set(df.columns) > set(cs.keys()), "cs's keys should be columns of df \n{}".format(cs.keys())
    df_temp = []
    lastdf = df
    for k, v in cs.items():
        df_temp.append(lastdf[lastdf[k] == v])
        lastdf=df_temp[-1]

    return df_temp[-1].reset_index(drop=True) 


def groupedCountsInDataFrame(df, group_key, count_key):
    '''group dataframe by group_key and count the frequencies of group_key
    returns a Counters dictionary with group_key as keys
    Parameters:
    -----------
        df : data frame
        group_key : grouping key
        count_key : key to count
    '''
    cd = {}
    for gr in set(df[group_key]):
        cd[gr] = Counter(df[df[group_key] == gr][count_key])
    return cd


def get_indexColFromDataFrame(df, condFun):
    '''returns the coordinates (index, column) of the values in
    a DataFrame that satisfy a condition
    Parameters
    ----------
    df: DataFrame
    condFun: callable
        e.g. lambda x: x >= 3
    Return
    ------
    indexCol: set of tuples
        set of index col tuples, {(i1, c1), ..., (in, cn)}
        '''
    indexCol = set()
    for i, r in df.iterrows():
        for j, V in r.iteritems():
            if condFun(V):
                indexCol |= {(i, j)}
                #print(i,j, V)
    return indexCol


### search sequence in PANDAS dataframe

def search_sequence_numpy(arr, seq):
    """ Find sequence in an array

    Parameters
    ----------
    arr: input 1D array
    seq: input 1D array

    Return
    ------
    Output : 1D Array of indices in the input array that satisfy the
        matching of input sequence in the input array.
        In case of no match, empty array is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create 2D array of sliding indices across entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
    else:
        return np.array([])


def filterIndicesForIct(ixArray, seqSize=2, diffCall=True):
    """get the indexes of a dataframe corresponding to the ict
    because the ict are defined by two consecutive calls,
    we are interested in the index of the first element of a size 2 sequence
    Parameters
    ----------
    ixArray: an array with the indices of a dataframe we are interested in
        we filter this array to keep their corresponding
        ict (dismiss last index of a sequence)
    seqSize: int
    diffCall: bool
        True =  different calls (XY), False = repetition (XX)"""

    if diffCall is True:  # different calls
        return ixArray[::seqSize]
    else:  # same call
        idx = np.array(ixArray[1:] - ixArray[:-1] == 1)
        return ixArray[:-1][idx]


def returnSequenceDf(df0, seq, label='call'):
    '''returns the dataframe with the sequences (type label) of interest
    Parameters
    -----------
    df0: pandas dataframe
    seq: np array with the seq. of interest
    label: column name in the df0 of the seq
    Returns
    -------
    a pandas dataframe containing only the sequence'''
    ## Determine whether the call are different or the same
    if seq[0] == seq[1]:
        diffCall = False
    else:
        diffCall = True
    ## search indices with seq
    arr = df0[label].values
    ix = filterIndicesForIct(search_sequence_numpy(arr, np.array(seq)),
                           diffCall=diffCall)
    return df0.loc[ix].reset_index()


##### DICTIONARIES

def stringiseDict(di, distr):
    '''converts a dictionary into string, supports dictionaries as values'''
    for ky, val in di.items():  # for each element in the dictionary
        if isinstance(val, dict):
            distr += '-'+stringiseDict(val, ky)
        else:
            distr += '-{}_{}'.format(ky, val)
    return distr


def arrangeDict(di, ordering_keys):
    """returns a numpy array with the values of the keys
    ordered according to ordering_keys
    if oredering key not in di.keys(), then the entry will be 0"""
    return np.array([di[ky] if ky in di.keys() else 0 for ky in ordering_keys])


def ignoredKeys(di, ignoreKeys):
    return [item for item in di.items() if item[0] not in ignoreKeys]


def returnSortingKeys(di, minCounts=None, reverse=True):
    '''keys that sort a dictionary
        di : key --> num dictionary, eg. Counter dictionary'''
    return np.array([item[0] for item in
                    sorted(di.items(), key=lambda x: x[1], reverse=reverse)
                    if item[1] >= minCounts])

### LISTS


def isArraySorted(arr):
    """checks is arr is sorted increasingly
    arr: array like"""
    x = [arr[i + 1] > arr[i] for i in np.arange(len(arr)-1)]
    return(all(x))


def flattenList(L):
    '''flattens a list
    Parameters
    ----------
    L : list
        list of lists
    Returns
    -------
    flattedList : list
    '''
    flattedList = []
    for sublist in L:
        flattedList += list(sublist)
    return flattedList


def sliceBackSuperSequence(superSeq, seqSlicer):
    '''
    Parameters
    ----------
    superSeq : ndarray
        sequence to slice into sub sequences
    seqSlicer : ndarray
        indices for slicing superSeq
    Returns
    -------
    seqOfSeqs: list
        list of lists
    '''
    i0 = 0
    seqOfSeqs = []
    for i in seqSlicer:
        seqOfSeqs.append(superSeq[i0:i])
        i0 = i
    return seqOfSeqs


def removeFromList(l0, l_ignore=['_ini', '_end']):
    '''Removes l_ignore from l0
    l0, l_ignore : list/np.array'''
    return [item for item in l0 if item not in l_ignore]


def removeElementWith(l0, l_ignore=['_ini', '_end']):
    '''Removes any element containing any intersection with l_ignore
    l0 : list of lists, l_ignore : list/np.array'''
    return [item for item in l0 if not any(set(l_ignore).intersection(item))]


### matrices \\ 2dim numpy array

def matrixSubsample(M, rwl, cll, rwsbset, cllsbset):
    '''Returns a subset of a matrix
    Parameters:
    -----------
        M : matrix
        rwl, cll : rows and column labels
        rwsbset, cllsbset : row and label subsets
    Returns: subset (Matrix, rw_labels, col_labels)'''
    IOrw = np.array([True if item in rwsbset else False for item in rwl])
    IOcl = np.array([True if item in cllsbset else False for item in cll])
    return M[IOrw, :][:, IOcl], rwl[IOrw], cll[IOcl]
