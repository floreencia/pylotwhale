# -*- coding: utf-8 -*-
"""
Created on Fri 

@author: florencia


function tools
"""

import functools

def compose2(f, g):
    """composes two functions"""
    return lambda x: f(g(x))

def composeFunctions(*functions):
    """composes a list of functions"""
    return functools.reduce(compose2, functions, lambda x: x)
