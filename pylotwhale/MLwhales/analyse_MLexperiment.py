# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:13:27 2015

@author: florencia
"""

from __future__ import print_function, division  # py3 compatibility
#import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
#import sys
import featureExtraction as fex
import pylotwhale.signalProcessing.signalTools as sT
import pylotwhale.utils.annotationTools as annT
import pylotwhale.MLwhales.MLtools_beta as myML
import pylotwhale.signalProcessing.audioFeatures as auf
import pylotwhale.utils.annotationTools as annT

### WSD


def featureVariation(df, score, param_col_indices=None):
    if param_col_indices is None: param_col_indices = np.array([8, 9, 11, -1])
    cols = df.columns.values
    params = cols[param_col_indices]
    s=''
    for p in params:
        thisdf = df.groupby(p)
        m = thisdf.mean()
        s += "{}\t{:.1f}+-{:.1f}".format(p, np.mean(m[score]), np.std(m[score]) )
        s += "\t{:.1f}\t{:.1f}\t{:.1f}\n".format(np.min(m[score]), np.max(m[score]), 
                                       -np.min(m[score]) + np.max(m[score]))
    return s












