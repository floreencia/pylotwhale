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

###### Iter parameters

parameter = 'Nslices'
paramKey = 'Nslices'
Nslices = 10
amp = np.arange(1,12) # np.linspace(a0, a, n_amps) 

def updateParamInDict(paramDict, paramKey, param):
    paramDict['featExtFun'][paramKey] = param 
    return paramDict

updateTestSet = True # feture extraction changes

preproStr="{}_{}".format(parameter, Nslices)



