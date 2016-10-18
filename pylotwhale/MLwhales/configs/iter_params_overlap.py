#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:05:03 2015
#!/usr/bin/python
@author: florencia
"""
#####
import numpy as np 
import os
import pylotwhale.MLwhales.experimentTools as exT

####### Iter parameters

parameter = 'overlap'
paramKey = 'overlap'
N0 = 0
Ndelta = 0.1
N = 1
amp = np.arange(N0, N, Ndelta) # np.linspace(a0, a, n_amps) 

def updateParamInDict(paramDict, paramKey, param):
    paramDict['featExtFun'][paramKey] = param 
    return paramDict

updateTestSet = True # because features change

preproStr="{}_{}".format(parameter, '{}-{}'.format(N0,N))


