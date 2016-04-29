# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:34:06 2016

@author: florencia
"""

import numpy as np
from collections import Counter


def groupedCountsInDataFrame(df, group_key, count_key):
    '''group data frame by group_key and count the frequencies of group_key
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
    '''returns a numpy array with the values of the keys ordered according to ordering_keys'''
    return np.array([di[ky] if ky in di.keys() else 0 for ky in ordering_keys ])

def ignoredKeys(di, ordering_keys):
    return [item for item in di.items() if item[0] not in ordering_keys]

def returnSortingKeys(di):
    return np.array([item[0] for item in 
            sorted(di.items(), key = lambda x:x[1], reverse=True)])