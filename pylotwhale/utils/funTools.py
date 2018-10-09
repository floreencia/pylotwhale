# -*- coding: utf-8 -*-
"""
Created on Fri 

@author: florencia


function tools
"""

import functools

def compose2(f, g):
    """compose two functions
	based https://stackoverflow.com/a/24047214/10310793"""
    return lambda *x, **kw: f(g(*x, **kw))

def composeFunctions(*functions):
    """compose a list of functions
	based https://stackoverflow.com/a/24047214/10310793"""
    return functools.reduce(compose2, functions, lambda x: x)
