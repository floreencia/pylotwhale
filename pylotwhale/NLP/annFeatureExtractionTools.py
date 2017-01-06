#!/usr/mprg/bin/python

from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import pylotwhale.utils.dataTools as daT


"""
    Extracting features from annotations
    florencia @ 16.05.16
"""


def df2X_stry(df, groupingKey='tape', call='note', sep='\s'):
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


def df2Xy(df, groupingKey='tape', call='note', sep='\s',
          token_pattern=u'(?u)\\b\\w+\\b' ):
    """extracts bag of features from data frame grouping them
    X: array (n_instances, n_features)
    y: list (n_instances)"""
    
    X_str, y = df2X_stry(df, groupingKey=groupingKey,
                         call=call, sep=sep)
    vectorizer = CountVectorizer(lowercase=False, token_pattern=token_pattern)
    X_sparse = vectorizer.fit_transform(X_str)
    X =  X_sparse.toarray()
    return X, np.array(y)


