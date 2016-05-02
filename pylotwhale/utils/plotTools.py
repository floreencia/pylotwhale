# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:56:58 2016

@author: florencia
"""

from __future__ import print_function
#import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
    if outFigName:fig.savefig(outFigName)
        
    return fig, ax     
        
        
### 2D plots ###

def fancyClrBarPl(X, vmax, vmin, maxN=10, cmap=plt.cm.jet, clrBarGaps=15, 
                  xTicks=None, yTicks=None,
                  tickLabsDict='', outplN='', plTitle='', xL='N', yL=r'$\tau$ (s)',
                  figureScale=(), extendCbar='both', extent=None):
    
    '''
    draws a beautiful color plot
    tickLabsDict     dictionary where the keys are the label of the cba ticks
                    and the vallues a re te postions
    Parameters:
    ------------                    
        X : 2d numpy array
        vmax : max value to plot
        vmin : cutoff min value
        maxN : maximum number of columns
        extent : scalars (left, right, bottom, top)     
        yTicks : (<tick_location>, <tick_names>), 
            <tick_location> array with the tick locations
            <tick_names>, array with the labels of the previous array
    '''
    fig, ax = plt.subplots()

    #colors setting
    #cmap = plt.cm.get_cmap('jet', clrBarGaps)    # discrete colors
    cmap.set_under((0.9, 0.9, 0.8)) #min
    cmap.set_over((1, 0.6, 0.6)) #max
    #cmap.set_nan((1, 0.6, 0.6)) #nan

    #plot
    cax=ax.imshow(X[:,:maxN], aspect ='auto', interpolation='nearest', 
               norm = colors.Normalize(vmin=vmin, vmax=vmax, clip = False),
               cmap=cmap, extent=extent)
    #labels
    ax.set_xlabel(xL)
    ax.set_ylabel(yL)
    if plTitle: ax.set_title(plTitle)
    
    # axis ticks
    if xTicks is not None: plt.xticks(xTicks)
    if yTicks is not None: plt.yticks(*yTicks)

    #clrbar
    cbar = fig.colorbar(cax, extend=extendCbar) #min, max, both
    cbar.set_clim((vmin, vmax)) # normalize cbar colors
    if not tickLabsDict: 
        tickLabsDict = {vmin: vmin, int(vmax/2):int(vmax/2), vmax:vmax} # tick labels
    cbar.set_ticks(tickLabsDict.values())        
    cbar.set_ticklabels(tickLabsDict.keys())
    
    #figScale
    if len(figureScale)==2: fig.set_size_inches(figureScale)        
    
    if outplN: fig.savefig(outplN, bbox_inches='tight')   

    return fig, ax     