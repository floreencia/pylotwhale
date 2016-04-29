# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:56:58 2016

@author: florencia
"""

from __future__ import print_function
#import sys
import numpy as np
import matplotlib.pyplot as plt


def stackedBarPlot(freq_arr, freq_arr_names=None, ylabel=None, xlabel=None,
                   key_labels=None, figsize=None, outFigName=None, cmap = plt.cm.Accent_r):
    '''
    plot freq_arr stacked
    Parameters:
    -----------
        freq_arr : array (n_plots, n_items)
        freq_arr_names : tick names
        ylabel : labels in the y axis
        xlabel : labels in the x axis
        key_labels : 
        figsize  tuple
        outFigName string
        cmap : color map callable
    '''
    
    freq_arr=np.array(freq_arr)
    n_plots, n_items = np.shape(freq_arr)
    
    if key_labels is None: key_labels = [None]*n_plots
    if freq_arr_names is None : freq_arr_names = np.arange(n_items)
    assert n_items == len(freq_arr_names)
    
    color=iter(cmap(np.linspace(0,1,n_plots)))

    ind = np.arange(np.shape(freq_arr)[1])
    arr0 = np.zeros(n_items)
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(len(freq_arr)):
        arr = freq_arr[i]
        ax.bar( ind, arr, bottom=arr0, 
               color=next(color), label=key_labels[i])
        arr0 += arr
        
    plt.legend()
    ax.set_xticks(np.arange(n_items)+0.4)   
    ax.set_xticklabels(freq_arr_names)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel : ax.set_ylabel(ylabel)
    if outFigName:fig.save(outFigName)