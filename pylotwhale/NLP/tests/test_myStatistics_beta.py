# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:45:10 2015
@author: florencia
"""

from __future__ import print_function
import numpy as np
from pylotwhale.NLP.myStatistics_beta import *

n1 = 10
test_seq1 = [list(np.repeat("a", n1)), list(np.repeat("b", n1))]
n2 = 10
test_seq2 = [list("ab" * n2)]


def test_repsProportion_in_listOfSeqs():
    s = [["c", "c", "c", "c", "b", "b"]]
    assert repsProportion_in_listOfSeqs(s) == (4, 5)
    s = [["c", "c"]]
    assert repsProportion_in_listOfSeqs(s) == (1, 1)
    s = [["c"]]
    assert repsProportion_in_listOfSeqs(s) == (0, 0)
    return True


def test_sequenceBigrams():
    seqO = sequenceBigrams(test_seq1)
    assert seqO.cfd["a"]["a"] == n1 - 1
    assert len(seqO.seqOfSeqs) == 2

    seqO = sequenceBigrams(test_seq2)
    assert seqO.cfd["a"]["a"] == 0
    assert seqO.cfd["a"]["b"] == n2
    assert len(seqO.seqOfSeqs) == 1


def test_randtest4bigrmas_inSequences():
    test_seq1.append(["a", "b"])
    seqO = sequenceBigrams(test_seq1)

    p_values, sh_dists = randtest4bigrmas_inSequences(
        seqO.seqOfSeqs, 100, condsLi=seqO.conditionsLi(), sampsLi=seqO.samplesLi()
    )
    assert p_values.loc["b", "b"] < 0.1
    assert p_values.loc["a", "b"] > 0.1
    assert p_values.loc["b", "a"] > 0.1
    assert p_values.loc["a", "a"] < 0.1


def test_normalisePDF():
    x = np.ones(10)
    r = (-1, 1)
    assert np.sum(normalisePDF(x, r) / len(x) * (r[-1] - r[0])) == 1
