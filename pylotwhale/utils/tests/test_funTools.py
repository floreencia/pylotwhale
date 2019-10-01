# -*- coding: utf-8 -*-
"""
Created on Fri 

@author: florencia


function tools
"""
from ..funTools import *
import numpy as np


def test_compose2():
    """composes two functions"""
    x = np.random.rand()
    f = compose2(lambda x: x ** 2, lambda x: np.sqrt(x))
    np.testing.assert_almost_equal(x, f(x))


def composeFunctions(*functions):
    """composes a list of functions"""
    funLi = [lambda x: x ** 2, lambda x: np.sqrt(x), lambda x: x ** 2, lambda x: np.sqrt(x)]
    f = composeFunctions(*funLi)
    np.testing.assert_almost_equal(x, f(x))
