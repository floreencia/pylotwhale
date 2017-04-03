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


labelsMapping = {'n_mels': '# Mel-filters',
                 'NFFT': '# FFT samples',
                 'overlap': 'overlap',
                 'Nslices': '# slices',
                 'f1_macro_CV': r'$<F_1>_m$',
                 'f1_macro_CV_std': r'$\sigma (F_1)$',
                 'ACC': 'A',
                 'PRE': 'P',
                 'REC': 'R',
                 'F1': r'$F_1$'
                } 

def featureVariation_ix(df, score, param_col_indices=None):
    """stats of the score for each of the params indicated with param_col_indices"""
    if param_col_indices is None: param_col_indices = np.array([8, 9, 11, -1])
    cols = df.columns.values
    param_cols = cols[param_col_indices]
    return featureVariation(df, score, param_cols)


def featureVariation(df, score, param_cols=None):
    """stats abut the score for each of the params indicated with param_cols"""

    if param_cols is None: param_cols = np.array(['n_mels', 'NFFT', 'Nslices'])
    s=''
    for p in param_cols:
        thisdf = df.groupby(p)
        m = thisdf.mean()
        s += "{}\t{:.1f}+-{:.1f}".format(p, np.mean(m[score]), np.std(m[score]) )
        s += "\t{:.1f}\t{:.1f}\t{:.1f}\n".format(np.min(m[score]), np.max(m[score]), 
                                       -np.min(m[score]) + np.max(m[score]))
    return s









