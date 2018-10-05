# -*- coding: utf-8 -*-
"""
Created on Fri 

@author: florencia


function tools
"""
from ..dataTools import *
import numpy as np
import pandas as pd

s = {('w', 'a'),
     ('w', 'b'),
     ('x', 'a'),
     ('x', 'b'),
     ('y', 'a'),
     ('y', 'b'),
     ('z', 'a'),
     ('z', 'b')}

thisdf_r = pd.DataFrame(np.random.randint(1,5, (4,2)), columns=list('ab'), index=list('wxyz'))
thisdf_c = pd.DataFrame(3, columns=list('ab'), index=list('wxyz'))


def test_search_sequence_numpy():
    """find sequence in numpy array
    special case"""
    arr = np.array(list('abcbdababz'))
    seq = np.array([list('ab')])
    np.testing.assert_equal(search_sequence_numpy(arr, seq), np.array([0, 1, 5, 6, 7, 8]))

    
def test_filterIndicesForIct():
    ### different calls XY
    seqSize=2
    np.testing.assert_equal(filterIndicesForIct(np.arange(10)), np.arange(10)[::seqSize])
    ### repetitions: XX
    arr = np.array([2,3,4,10,11,12,13,20,21])
    idx_0 = arr[:-1][np.array(arr[1:] - arr[:-1] == 1)]
    idx = filterIndicesForIct(arr, diffCall=False)
    np.testing.assert_equal(idx_0, idx)

def test_flattenList():
    li = [['a', 'b', 'c'], ['t'], ['x', 'y']]
    assert(flattenList(li) == ['a', 'b', 'c', 't', 'x', 'y'])

    li = [['a', 'b', 'c'], ['t'], ['x', 'y', ['hey']]]   
    assert(flattenList(li) == ['a', 'b', 'c', 't', 'x', 'y', ['hey']])


def test_sliceBackSuperSequence():
    tSeq = list('abcdefgxyz')
    seqSlicer=np.array([3,6, len(tSeq)])

    assert (sliceBackSuperSequence(tSeq, seqSlicer)
            == [['a', 'b', 'c'],
                ['d', 'e', 'f'],
                ['g', 'x', 'y', 'z']])
    

def test_get_indexColFromDataFrame():
    condFun = lambda x: x >= 2
    assert(s == get_indexColFromDataFrame(thisdf_c, condFun))
    condFun = lambda x: x >= 5
    assert(set() == get_indexColFromDataFrame(thisdf_c, condFun))




def test_isArraySorted():
	x = np.arange(10)
	assert(isArraySorted(x) is True)
	assert(isArraySorted(x[::-1]) is False)
	x[-1] = 2
	assert(isArraySorted(x) is False)
	x[-1] = 20
	assert(isArraySorted(x) is True)




