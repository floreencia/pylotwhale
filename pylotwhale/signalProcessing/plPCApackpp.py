#!/usr/bin/python

import pylab as pl
import numpy as np
import os
import sys

sys.path.append('/home/florencia/whales/scripts/')
import spTools as sp
import pcaO

#import scikits.audiolab as al

def pcaCalcs(fileN, plBasis = 0, plWeights = 1, plDataInPC = 1, plChopedPeak = 1, sRate=192000):

    '''
    plots the average of the variance keept by the k-th PC
    
    fileN = ceps file (cols)time x ceps "<file>-ceps.dat"
    
    '''
    cepsM = np.loadtxt(fileN)#.T # time runns horizontally

    # alighn
    Nc, Nr = np.shape(cepsM)
    if Nc>Nr:    
        cepsM = cepsM.T
        print "matrix alighned:", Nr, "cesptral coeffis", Nc, "time steps" 

    baseN = os.path.splitext(fileN)[0]
    print baseN
   
    N = len(cepsM.T) #time points
    tf = N/(1.0*sRate)
    
    pca = pcaO.pca0mean(cepsM.T) # vertical time

    if(plBasis):
        print "ploting PCA basis\n"
        Tmatrix = pca.vectors()
        plotPCbasis( Tmatrix, baseN)

    if(plWeights):
        print "ploting PCA weights\n"
        wei = pca.weights()
        plotPCweights( wei, baseN)

    if(plDataInPC):
        ccDim=len(pca.weights())
        print "ploting data in PC\n"
        timePCdata = pca.dataInMyBasis(cepsM.T,ccDim).T # transpose to leave the time run in the horizontal axis
        plotInPC(timePCdata, baseN, 30)



#------------- FUNCTIONS ----------------------

#------------- plot basis function ------------

def plotPCbasis(Tmatrix, baseN):

    fig = pl.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
#    cax = pl.imshow(Tmatrix, extent=[pc1,pcf,ccf,cci], interpolation = 'nearest', aspect ='auto')
    cax = pl.imshow(Tmatrix, interpolation = 'nearest', aspect ='auto')

    
    ax.set_xlabel( 'PC vector', fontsize=16 )
    ax.set_ylabel( 'cepstral coefficient ', fontsize=16 )

    ax.tick_params(axis='both', labelsize='x-large')  #set_xticks(ticks, fontsize=24)
   
    # cbar
    cbar = fig.colorbar(cax, ticks=[Tmatrix.min(), 0 , Tmatrix.max()])
    cbar.ax.set_yticklabels([-1,0, 1], size='x-large')# vertically oriented co
 
    # Set x ticks
    #  pl.xticks(np.linspace(1,xM,xM,endpoint=True))
    
    outF = baseN+"PCbasis.png"
    print outF
    pl.savefig(outF)
    del fig
    del ax
    del cax


#------------- plot data in PC's ----------------------

def plotInPC(data, baseN, yM=10, sRate=192000): #, pc1=1, pcf=128, cci=1, ccf=128):

    Nc, Nt = np.shape(data)
    tf = Nt/(1.0*sRate)
    
    if yM>Nc:
        yM=Nc
    
    print 'tf=', tf, "N", Nt

    fig = pl.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    cax = pl.imshow(data, interpolation = 'nearest', extent=[0,tf,1, Nc+1], aspect = 'auto', origin='auto')
    pl.ylim(1,yM)

    # cbar
    ax.set_xlabel( 'time [s] ', fontsize=16 )
    ax.set_ylabel( 'principal component', fontsize=16 )

    # Set x ticks
   # pl.yticks(np.linspace(1, yM, yM, endpoint=True))

    ax.tick_params(axis='both', labelsize='x-large')  #set_xticks(ticks, fontsize=24)
    m = data.min()
    M = data.max()
    cbar = fig.colorbar(cax, ticks=[m, M])
    cbar.ax.set_yticklabels(['%.0f'%m, '%.0f'%M], size='x-large')# vertically oriented co 
    
    outF = baseN+"atPCbasis.png"
    print outF
    pl.savefig(outF)
    del fig
    del ax
    del cax


#------------- plot weights ----------------------

def plotPCweights( vec, baseN, r_tresh=0.9): #, pci=1, pcf=128):
    deno =  vec.sum() # norm
    
    for i in np.arange(1, len(vec)): # percent variance
        vec[i] = vec[i] + vec[i-1]# + vec[i] 
        
    vec /= deno # normalize
        
    # find i such that 99% of variance is preserved
    i = -1
    while True:
        i = i+1
        r = vec[i]
           # print i
        if ( r >= r_tresh):
            break
        
    # plot or spit
    if(i>1):
        fig = pl.figure(figsize=(7,5))
        ax = fig.add_subplot(111)

        x = np.linspace(1,i+1,i+1,endpoint=True)
        cax = pl.plot(x[:i], vec[:i])

        ax.set_xlabel( 'PC index', fontsize=16 )
        ax.set_ylabel( 'r', fontsize=16 )
        ax.tick_params(axis='both', labelsize='x-large')  #set_xticks(ticks, fontsize=24)
   
        pl.xlim(0,x[i])
        pl.ylim(0,1)

        # xM = len(vec[:i])-1
        # pl.xticks(np.linspace(1,xM,xM,endpoint=True))

        outF = baseN+"PCweights.png"
        print outF
        pl.savefig(outF)
        
        del fig
        del ax 
        del cax

        print "i =", i, vec[i]
        
    else:
        print "i =", i, vec[i]
        
        

 
