#!/usr/bin/python

import pylab as pl
import numpy as np
import os


def plCepsFromFile(fileN, yM = 10, sRate = 192000):

    '''
    plots the data as projected in the PC, in the ceps coeffis index units
    only plots the first 10 principal components by default, assuming 128 coefficients
    
    fileN = PCT matrix
    yM = maximumvalue oy coordinate (principal component)
    
    '''

    atPCbasis = np.loadtxt(fileN).T
    baseN = os.path.splitext(fileN)[0]

    N = len(atPCbasis.T)
    tf = N/(1.0*sRate)
    
    outF = baseN+"-atPC.png"
    print outF, 'tf=', tf, "N", N
    
    fig = pl.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    cax = pl.imshow(atPCbasis, interpolation = 'nearest', extent=[0,tf,1,128], aspect = 'auto', origin='auto')
    pl.ylim(1,yM)

    # cbar
    ax.set_xlabel( 'time [s] ', fontsize=16 )
    ax.set_ylabel( 'principal component', fontsize=16 )

    # Set x ticks
    pl.yticks(np.linspace(1,yM,yM,endpoint=True))

    #ya = ax.get_yaxis()
    #ya.set_major_locator(MaxNLocator(integer=True))

    ax.tick_params(axis='both', labelsize='x-large')  #set_xticks(ticks, fontsize=24)
    m = atPCbasis.min()
    M = atPCbasis.max()
    cbar = fig.colorbar(cax, ticks=[m, M])
    cbar.ax.set_yticklabels(['%.0f'%m, '%.0f'%M], size='x-large')# vertically oriented co 
  
    pl.savefig(outF) 



    # save and show
    #fig.show() # doesn't stops the script
    #pl.show() #stops the script until you close the plot window
    
###################################

#if __name__ == '__main__':
    #pl.io()
 #   plCepsFromFile('/home/florencia/Desktop/mySamples/harmonicBeep220p10H-ts.dat-cepst.dat')
  
#raw_input()
