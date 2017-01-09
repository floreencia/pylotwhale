# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:45:10 2015
@author: florencia
"""

from __future__ import print_function
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

#import os

#import pylotwhale.signalProcessing.signalTools_beta as sT
#import pylotwhale.utils.whaleFileProcessing as fp
import pylotwhale.NLP.annotations_analyser as aa


d = {'call' : ['A', 'B', 'C', 'D', 'A', 'C', 'E'],
    't' : [1.0, 1.5, 2.0, 2.0, 0.5, 10.0, np.nan]}
te_df = pd.DataFrame(d)

df_100 = pd.DataFrame(np.hstack((np.random.randint(0,5,size=(100, 1)), np.random.rand(100, 1))), columns=list('AB'))


def test_df2listOfSeqs():
    ## NAN ending
    Dt = (None, np.max(te_df['t'])) # one big sequence
    assert len(aa.df2listOfSeqs(te_df, Dt, time_param='t')[0]) == len(te_df)
    
    Dt = (None,0) # each element by its own
    assert len(aa.df2listOfSeqs(te_df, Dt, time_param='t')) == len(te_df)

    ## float ending
    Dt = (None, np.max(te_df['t'])) # one big sequence
    te_df2 = te_df[:-1]
    assert len(aa.df2listOfSeqs(te_df2, Dt, time_param='t')[0]) == len(te_df2)
    
    Dt = (None,0) # each element by its own
    assert len(aa.df2listOfSeqs(te_df2, Dt, time_param='t')) == len(te_df2)

    
def test_chunkStruct():
    Dtvec = np.linspace(0,1,10)
    # ngrams distribution for different Dt values
    Ngrams_dist = aa.Ngrams_distributionDt_ndarray({'a':df_100}, 
                                                    Dtvec, seqLabel='A', 
                                                    time_param='B' )
    # calls in ngrams
    calls_dist = aa.NgramsDist2callsInNgramsDist(Ngrams_dist)
    ## the number of call is conserved
    totCalls = calls_dist.sum(axis=1)
    assert( all(totCalls == totCalls[0]))
