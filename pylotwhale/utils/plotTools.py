# -*- coding: utf-8 -*-
"""
Tools for plotting
"""

from __future__ import print_function, division
#import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.signal import get_window

import scipy.cluster.hierarchy as sch


def cbarLabels(minV, maxV):
    """
    Returns a tuple with three integer labels between minV and maxV
    Meant for exponents.
    """
    minV = int(np.ceil(minV))
    maxV = int(np.floor(maxV))
    ml = (minV + maxV)/2
    D = min(abs(ml-maxV), abs(minV-ml))
    ul = ml - D
    ll = ml + D
    return (ll, ml, ul)


def stackedBarPlot(freq_arr, freq_arr_names=None, ylabel=None, xlabel=None,
                   key_labels=None, figsize=None, outFigName=None,
                   cmap=plt.cm.Accent_r):
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

    freq_arr = np.array(freq_arr)
    n_plots, n_items = np.shape(freq_arr)

    if key_labels is None:
        key_labels = [None]*n_plots
    if freq_arr_names is None:
        freq_arr_names = np.arange(n_items)
    assert n_items == len(freq_arr_names)

    color = iter(cmap(np.linspace(0, 1, n_plots)))

    ind = np.arange(np.shape(freq_arr)[1])
    arr0 = np.zeros(n_items)
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(len(freq_arr)):
        arr = freq_arr[i]
        ax.bar(ind, arr, bottom=arr0,
               color=next(color), label=key_labels[i])
        arr0 += arr

    plt.legend()
    ax.set_xticks(np.arange(n_items)+0.4)
    ax.set_xticklabels(freq_arr_names)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if outFigName:
        fig.savefig(outFigName)

    return fig, ax


###### utilities ######

      
def display_numbers(fig, ax, M, fontSz, fmt=int, condition=lambda x: True):
    """
    add text to imshow matrix
    Parameters:
    ----------
    condition: must be satisfied in order to show the numbers
        eg. lambda x : x > 0
    """

    r,c = np.shape(M)
    
    ## display numbers in the matrix
    for i in range(r):
        for j in range(c):
            if condition(M[i,j]):
                ax.text(x=j, y=i, s=fmt(M[i, j]), va='center', ha='center', 
                        fontsize=fontSz )                        
    return fig, ax
        
        

def fancyClrBarPl(X, vmax, vmin, maxN=10, cmap=plt.cm.jet, clrBarGaps=15, 
                  xTicks=None, yTicks=None, figsize=None,
                  tickLabsDict='', outplN='', plTitle='', xL='N', yL=r'$\tau$ (s)',
                  figureScale=(), extendCbar='both', xextent=None, yextent=None,
                  under_clr="light grey", over_clr="neon pink"):
    
    '''
    draws a beautiful color plot
    tickLabsDict     dictionary where the keys are the label of the cba ticks
                    and the values a the positions
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
        xextent: len 2 list like
        yextent: len 2 list like
        under_clr, over_clt: str
            https://xkcd.com/color/rgb/
    '''
    fig, ax = plt.subplots(figsize=figsize)

    #colors setting
    #cmap = plt.cm.get_cmap('jet', clrBarGaps)    # discrete colors
    cmap.set_under(sns.xkcd_palette([under_clr])[0]) #min
    cmap.set_over(sns.xkcd_palette([over_clr])[0]) #max
    #cmap.set_nan((1, 0.6, 0.6)) #nan
    
    extent=[0, len(X.T), len(X), 0]
    if xextent:
        extent[:2] = xextent
    if yextent:
        extent[2:] = yextent

    #plot
    cax=ax.imshow(X[:,:maxN], aspect ='auto', interpolation='nearest', 
                  norm = colors.Normalize(vmin=vmin, vmax=vmax, clip = False),
                  cmap=cmap, extent=extent)
    ### labels
    ax.set_xlabel(xL)
    ax.set_ylabel(yL)
    if plTitle: ax.set_title(plTitle)

    ### axis ticks
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
    if len(figureScale) == 2: fig.set_size_inches(figureScale)        
    
    if outplN: fig.savefig(outplN, bbox_inches='tight')   

    return fig, ax



def plImshowLabels(A, xTickL, yTickL, xLabel=None, yLabel=None,
                   plTitle='', 
                   clrMap=plt.cm.viridis_r, cbarAxSize=2, 
                   cbarLim=None, cbarOrientation='vertical', Nclrs=11, 
                   cbarTicks=False, cbarTickLabels=False, cbar=True, outFig='',
                   figsize=None,
                   underClr = 'white', badClr='gray', **kwarg):
    """
    plot a matrix with ticks labels
    Parameters
    ----------
        **kwargs:
    norm=matplotlib.colors.Normalize(vmin=1e-14, vmax=1, clip = False)
    """
   
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.get_cmap(clrMap, Nclrs)    # 11 discrete colors
    cmap.set_under(underClr) #min
    cmap.set_bad(badClr) #nan

    im = ax.imshow(A, interpolation = 'nearest', cmap = cmap, origin='bottom', **kwarg)
                   # extent = [0, len(xTickL),0,len(yTickL)], origin='bottom')

    ax.set_yticks(np.arange(len(yTickL)))  # + 0.5 ) #flo -> +0.1)
    ax.set_yticklabels(yTickL)
    ax.set_xticks(np.arange(len(xTickL)))  # + 0.5 )
    ax.set_xticklabels(xTickL, rotation=90)
    if plTitle: ax.set_title(plTitle)
    if xLabel: ax.set_xlabel(xLabel)
    if yLabel: ax.set_ylabel(yLabel)
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
        fig.savefig(outFig, bbox_inches='tight')
        #print len(i2c), "\nout:%s"%outFig

    return fig, ax


### Clustering plots


def plDmatrixWDendrogram(distM, labels, cmap=plt.cm.RdYlBu, figsize=None,
                         NcbarTicks=4, cbarAxis=None, oFig=None, **kwargs):
    """
    kwargs for matshow
    """
    
    Y = linkage_matrix = sch.ward(distM)
    
    fig = plt.figure(figsize=figsize)

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
    idx1 = Z1['leaves']#[::-1]
    idx2 = Z2['leaves']
    D = distM[idx1,:][:,idx2]
    #D = distM[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', cmap=cmap, origin='lower', **kwargs)
    
    # Distance matrix tick labels
    axmatrix.set_xticks(np.arange(len(idx1)))
    axmatrix.set_xticklabels(labels[idx1], minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()
    plt.xticks(rotation=-90)#, fontsize=8)
    
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

    if oFig: fig.savefig(oFig, bbox_inches='tight')
    
#### spectrograms

def plspectro(waveform, sRate, outF='', N=2**9, v_cut=None, 
              overlap = 0.5, winN = 'hanning',
              xl='time (s)',  yl= 'frequency (kHz)',
              spec_fac=0.99999, plTitle='', cmN='bone_r',
              figsize=None, ax=None, fig=None, **kwargs):
    """
    plots spectrogram
    Parameters
    ----------
    v_cut: tuple
        frequency range, None sets to full spectrum (0, fs/2).
        (1000, 20*1000) works nicely for the whales
    overlap: float [0,1)
        FFT overlap
    NFFT overlap
    spec_fac: 
        cuts off the powerspectrum 0 max cutting, 1 doesn't do anything
    threshold all with variations smaller than
    """

    if v_cut is None:
        v_cut=(0, sRate/2)
    #tf = float(N)/sRate
    tf = 1.0*(len(waveform)) /sRate
    #ff = sRate/2.0
    N = int(N)
    win = get_window(winN, N)
    noverlap = int(overlap*N)
    A0 = plt.specgram(waveform, Fs = sRate, NFFT = N, noverlap = noverlap, 
                      window = win)[0]

    # Spectro editing
    A = selectBand(A0, fr_f = sRate/2, v_cut=v_cut) # band filter
    A = reScale_E(A, spec_factor=spec_fac) # zeros the the spectral energy smaller than 0.001% of <E>

    #plt.clf()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)#figsize=(max([3,int(tf/0.3)]), ff/8000))

    cax = ax.imshow(np.log(A), extent=[ 0, tf, v_cut[0]/1000, v_cut[1]/1000],
                    origin='lower', aspect = 'auto', 
                    cmap=plt.cm.get_cmap(cmN), **kwargs)#, interpolation = 'nearest')

    #labels
    ax.set_xlabel(xl)#, fontsize=16)
    ax.set_ylabel(yl)#, fontsize=16)

    '''
    if tf<1:
        plt.xticks(np.arange(0, tf, tf/2.0))
    else:
        plt.xticks(np.arange(0, tf, 1.0))
    '''

    return fig, ax

def plspectro_cbar(ax):
    """TODO... plot spectrogram with fancy cbar marks"""    
    
    #cbar
    ( ll, ml, ul ) = cbarLabels( np.log(A).min(), np.log(A).max() )
    cbar = fig.colorbar(cax, ticks=[ll, ml, ul])
    cbar.ax.set_yticklabels(['10$^{%d}$'%ll,'10$^{%d}$'%ml,'10$^{%d}$'%ul])

    return fig, ax

def selectBand(M, fr_0 = 0, fr_f = 24000, v_cut = (1.0*1000, 20.0*1000)):
    """
    selects a band on frequencies from the matrix M
    fr_0, initial frequency of the matrix
    fr_f, final frequency of the matrix, sampR/2
    cutting frequencies
    v0_cut
    vf_cut
    """
    ny, nx = np.shape(M)
    n0_cut = int( ny*v_cut[0]/( fr_f - fr_0 ) )
    nf_cut = int( ny*v_cut[1]/( fr_f - fr_0 ) )
    #print n0_cut, nf_cut, nx, ny
    return M[ n0_cut:nf_cut, : ]


def reScale_E(M, spec_factor = 1.0/3.0):
    """
    Zeroes the noise by taking only the part of the spectrum with the highest energy.
    - spec_factor \in [0,1],
    --- 0 - max cutting energy (we don't see anything)
    --- 1 - min cutting energy (returns M without doing anything )
    * M, log (spectrogram)
    """

    assert(spec_factor >= 0 and spec_factor <= 1)
    cutE = (np.min(M) - np.max(M))*spec_factor + np.max(M)
    nx, ny = np.shape(M)
    M_tr = np.array(M)
    M_tr[M < cutE] = cutE

    return M_tr
