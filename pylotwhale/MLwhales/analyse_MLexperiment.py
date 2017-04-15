# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 11:13:27 2017

@author: florencia
"""

from __future__ import print_function, division  # py3 compatibility
import numpy as np

### WSD


labelsMapping = {'n_mels': '# Mel-filters',
                 'NFFT': '# FFT samples',
                 'overlap': 'overlap',
                 'Nceps': 'quefrency',
                 'MFCC': '# MFCC',
                 'Nslices': '# slices',
                 'f1_macro_CV': r'$<F_1 >_{call}$', #r'$<F_1>_m$',
                 'f1_macro_CV_std': r'$\sigma (F_1)$',
                 'ACC': 'A',
                 'PRE': 'P',
                 'REC': 'R',
                 'F1': r'$F_1$'                 
                }
                
def labels_mapping(key):
    if key in labelsMapping.keys():
        return labelsMapping[key]
    else:
        return key


def featureVariation_ix(df, score, param_col_indices=None):
    """stats of the score for each of the params indicated with param_col_indices"""
    if param_col_indices is None: param_col_indices = np.array([8, 9, 11, -1])
    cols = df.columns.values
    param_cols = cols[param_col_indices]
    return featureVariation(df, score, param_cols)


def featureVariation(df, score, param_cols=None, sep='\t'):
    """stats abut the score for each of the params indicated with param_cols"""

    if param_cols is None: param_cols = np.array(['n_mels', 'NFFT', 'Nslices'])
    s=''
    for p in param_cols:
        thisdf = df.groupby(p)
        m = thisdf.mean()
        s += "{}{}{:.1f}+-{:.1f}".format(p, sep, np.mean(m[score]), np.std(m[score]))
        s += "{}{:.1f}{}{:.1f}".format(sep, np.min(df[p]), sep, np.max(df[p]) )
        s += "{}{:.1f}{}{:.1f}{}{:.1f}\n".format(sep, np.min(m[score]), sep,
                                                 np.max(m[score]), sep,
                                                 np.max(m[score]) - np.min(m[score]))
    return s









