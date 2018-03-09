# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:45:10 2015
@author: florencia
"""

from __future__ import print_function
import numpy as np
from pylotwhale.NLP.myStatistics_beta import *

n1 = 10
test_seq1 = [list(np.repeat('a', n1)), list(np.repeat('b', n1))]
n2=10
test_seq2 = [list('ab'*n2)]


def test_repsProportion_in_listOfSeqs():
    s = [['c', 'c', 'c', 'c', 'b', 'b']]
    assert repsProportion_in_listOfSeqs(s) == (4, 5)
    s = [['c', 'c']]
    assert repsProportion_in_listOfSeqs(s) == (1, 1)
    s = [['c']]
    assert repsProportion_in_listOfSeqs(s) == (0, 0)
    return True
    
def test_sequenceBigrams():
    seqO = sequenceBigrams(test_seq1)
    assert seqO.cfd['a']['a'] == n1-1
    assert len(seqO.seqOfSeqs) == 2

    seqO = sequenceBigrams(test_seq2)
    assert seqO.cfd['a']['a'] == 0
    assert seqO.cfd['a']['b'] == n2
    assert len(seqO.seqOfSeqs) == 1


def test_randomisation_test4bigrmas_inSequences():
    test_seq = test_seq1 + ['a', 'b']
    seqO = sequenceBigrams(test_seq)
    calls = [item[0] for item in seqO.sortedCallCounts if item[1] >= 0]
    samplesLi = calls[:] + ['_end'] #None #[ 'A', 'B', 'C', 'E', '_ini','_end']
    condsLi = calls[:] + ['_ini']
    note2samp_i = {s: i for i, s in enumerate(samplesLi)}
    note2cond_i = {c: i for i, c in enumerate(condsLi)}
    Mp, samps, conds =  ngr.kykyCountsDict2matrix(seqO.cpd, conditions=condsLi, samples=samplesLi)
    p_values, sh_dists = randomisation_test4bigrmas_inSequences(seqO.seqOfSeqs, 
                                                                Mp, 100, 
                                                                condsLi=condsLi,
                                                                sampsLi=samplesLi)
    assert (p_values[note2cond_i['a'], note2samp_i['a']] < 0.1)
    assert (p_values[note2cond_i['a'], note2samp_i['b']] > 0.5)
    assert (p_values[note2cond_i['b'], note2samp_i['a']] > 0.5)
    assert (p_values[note2cond_i['b'], note2samp_i['b']] < 0.1)

