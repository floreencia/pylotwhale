# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:56:58 2016

@author: florencia
"""

from __future__ import print_function
#import sys
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


import scipy
import pylab
import scipy.cluster.hierarchy as sch


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
                    and the values a the postions
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
    
    
def plImshowLabels(A, xTickL, yTickL, xLabel=None, yLabel=None,
              plTitle='', clrMap = 'winter_r', cbarAxSize=2, 
              cbarLim=None, cbarOrientation='vertical', Nclrs=11, 
              cbarTicks=False, cbarTickLabels=False, cbar=True, outFig='',
              figsize=None,
              underClr = 'white', badClr='gray'):
    """
    plot a matrix with ticks
    fraction=0.0
        axsize : colorbar tickness
    """
   
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.get_cmap(clrMap, Nclrs)    # 11 discrete colors
    cmap.set_under(underClr) #min
    cmap.set_bad(badClr) #nan

    im = ax.imshow(A, interpolation = 'nearest', cmap = cmap, origin='bottom')
                   # extent = [0, len(xTickL),0,len(yTickL)], origin='bottom')
    
    ax.set_yticks( np.arange( len(yTickL ) ))# + 0.5 ) #flo -> +0.1)
    ax.set_yticklabels( yTickL ) 
    ax.set_xticks( np.arange(len(xTickL)))# + 0.5 ) 
    ax.set_xticklabels( xTickL, rotation=90 ) 
    if plTitle: ax.set_title(plTitle)
    if xLabel : ax.set_xlabel(xLabel)
    if yLabel : ax.set_ylabel(yLabel)
    #COLOR BAR
    if isinstance(cbarLim, tuple): im.set_clim(cbarLim) # cmap lims

    divider = make_axes_locatable(ax)
    if cbar: 
        cax = divider.append_axes("right", size="{}%".format(cbarAxSize), pad=0.1)
        cbar = fig.colorbar(im, cax = cax)#, fraction=0.046, pad=0.04)#, extend='min')
        if isinstance(cbarLim, tuple): cbar.set_clim(cbarLim) # cbar limits
        if isinstance(cbarTicks, list): cbar.set_ticks(cbarTicks)
        if isinstance(cbarTickLabels, list): cbar.set_ticklabels(cbarTickLabels)  

        
    if outFig: 
        fig.savefig(outFig, bbox_inches = 'tight')
        #print len(i2c), "\nout:%s"%outFig
    
    return fig, ax

### Clustering plosts



def plDmatrixWDendrogram(distM, labels, cmap=pylab.cm.RdYlBu,
                         NcbarTicks=4, cbarAxis=None):
    
    Y = linkage_matrix = sch.ward(distM)
    
    fig = pylab.figure(figsize=(8,8))

    # FIRST DENDROGRAM
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6]) # left
    #Y = sch.linkage(linkage_matrix, method='centroid')
    Z1 = sch.dendrogram(Y, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # SECOND DENDROGRAM
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2]) # up
    #Y = sch.linkage(linkage_matrix, method='single')
    Z2 = sch.dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # DISTANCE MATRIX
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6]) # [x0, y0, Dx, Dy ]
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = distM[idx1,:][:,idx2]
    #D = distM[:,idx2]
    im = axmatrix.matshow(D, aspect='auto',cmap=cmap)#origin='lower'
    
    # Distance matrix tick labels
    axmatrix.set_xticks(range(len(idx1)))
    axmatrix.set_xticklabels(labels[idx1], minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()
    pylab.xticks(rotation=-90)#, fontsize=8)
    
    # image ticks
    axmatrix.set_yticks((range(len(idx2))))
    axmatrix.set_yticklabels(labels[idx2], minor=False)
    axmatrix.yaxis.set_label_position('right')
    axmatrix.yaxis.tick_right()
    
    # cbar
    if cbarAxis is None: cbarAxis = [0.91, 0.71, 0.03, 0.2 ]  
    axcolor = fig.add_axes(cbarAxis)#[0.96,0.1,0.02,0.6])
    cbar = fig.colorbar(im, cax = axcolor)#
    tick_locator = ticker.MaxNLocator(nbins=NcbarTicks)
    cbar.locator = tick_locator
    cbar.update_ticks() 
    
        