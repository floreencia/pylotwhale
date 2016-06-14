# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:34:06 2016

@author: florencia

Various handy functions (sorting, spliting, searching) for working with data
    - pandas dataframes
    - dictionaries
    - lisits
"""

import numpy as np
from collections import Counter

def dictOfGroupedDataFrames(df0, groupingKey='tape'):
    '''groups a dataframe according to groupingKey and returns a dictionary of data frames'''
    df = {}
    keysSet = set(df0[groupingKey])
    for t in keysSet:
        df[t] = df0[df0[groupingKey] == t].reset_index(drop=True)
    return df

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

def arrangeDict(di, ordering_keys):
    '''returns a numpy array with the values of the keys ordered according to ordering_keys
    if oredering key not in di.keys(), then the entry will be 0'''
    return np.array([di[ky] if ky in di.keys() else 0 for ky in ordering_keys ])

def ignoredKeys(di, ignoreKeys):
    return [item for item in di.items() if item[0] not in ignoreKeys]
    
def removeFromList(l0, l_ignore=['_ini', '_end']):
    return [item for item in l0 if not any(set(l_ignore).intersection(item))]    

def returnSortingKeys(di, minCounts=None):
    '''keys that sort a dictionary'''
    return np.array([item[0] for item in 
            sorted(di.items(), key = lambda x:x[1], reverse=True) if item[1] > minCounts])
            
### search sequence in PANDAS dataframe           
            
def search_sequence_numpy(arr,seq):
    """ Find sequence in an array

    Parameters:
    ----------    
        arr    : input 1D array
        seq    : input 1D array

    Return:
    -------   
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
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any>0:
        return np.where(np.convolve(M, np.ones((Nseq),dtype=int)) > 0 )[0]
    else:
        return np.array([])
    
def filterIndexesForIct(ixArray):
    '''get the indexes of a dataframe corresponding to the ict
        ixArray  : an array with the indexes of a dataframe we are interested in
        we filter this array to keep their corresponding ict (dismis last index of a sequence)'''
    return ixArray[ixArray[1:] - ixArray[:-1] == 1]

def returnSequenceDf(df0, seq, label='call'):
    '''returns the dataframe with the sequences (type label) of interest
    Parameters:
    -----------
        < df0 : pandas dataframe
        < seq : np array with the seq. of interest
        < label : column name in the df0 of the seq.
    Returns:
    --------
        > a pandas dataframe containing only the sequence'''
    arr = df0[label].values
    ix=filterIndexesForIct(search_sequence_numpy(arr, np.array(seq)))
    return df0.loc[ix].reset_index()
            