# -*- coding: utf-8 -*-
"""
Created on Fri 

@author: florencia


function tools
"""
from ..dataTools import *
import numpy as np

def test_search_sequence_numpy():
    """find sequence in numpy array
    special case"""
    arr = np.array(list('abcbdababz'))
    seq = np.array([list('ab')])
    np.testing.assert_equal(search_sequence_numpy(arr, seq), np.array([0, 1, 5, 6, 7, 8]))

    
def test_filterIndexesForIct():
    ### different calls XY
    seqSize=2
    np.testing.assert_equal(filterIndexesForIct(np.arange(10)), np.arange(10)[::seqSize])
    ### repetitions: XX
    arr = np.array([2,3,4,10,11,12,13,20,21])
    idx_0 = arr[:-1][np.array(arr[1:] - arr[:-1] == 1)]
    idx = filterIndexesForIct(arr, diffCall=False)
    np.testing.assert_equal(idx_0, idx)





