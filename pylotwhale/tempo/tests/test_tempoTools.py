# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:45:10 2015
@author: florencia
"""

from __future__ import print_function
import numpy as np
from pylotwhale.NLP.tempoTools import *

X = np.array([3, 4, 8, 10, 11, 17, 20])


def test_binary_time_series():

    X = np.array([3, 4, 8, 10, 11, 17, 20])
    _, IO, _ = binary_time_series(X, 1)
    Y = np.zeros(X[-1] + 1)
    Y[X] = 1
    np.testing.assert_array_equal(IO, Y)


def test_binarise_times_in_window():
    Dt = 0.5
    t0 = 0
    winsz = X[-1]
    _, IO2 = binarise_times_in_window(X, t0, t0 + winsz, Dt)
    _, IO1, _ = binary_time_series(X, Dt)

    np.testing.assert_array_equal(IO1, IO2)
