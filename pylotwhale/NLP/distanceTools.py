#!/usr/mprg/bin/python

from __future__ import print_function
import pylotwhale.utils.dataTools as daT
import pandas as pd

"""
    Extracting features from annotations
    florencia @ 16.05.16

"""


def df2groupedCorpus(df, groupingKey='tape', call='note', sep='\s'):
    """returns the feature list from a dataframe grouping  
    the value of the grouping key
    Parameteres
    -----------
    df: pandas dataframe
    groupingKey: str
    call: str
    Returns
    --------
    X_str: list of strings (n_instances)
        feateure matrix to use as input of CountVectorizer
    y: list, (n_instances)
        labels of the instances from the grouping key
    """
    ## group dataframe
    df_dict = daT.dictOfGroupedDataFrames(df, groupingKey=groupingKey)
    X_str = [] # feature list
    y = [] # group label
    for ky in df_dict.keys():
        y.append(ky) # gro
        X_str.append(sep.join( df_dict[ky][call].values))
    return X_str, y
