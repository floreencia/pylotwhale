#!/usr/bin/python

import pylab as pl
import numpy as np
import os
import sys

sys.path.append('/home/florencia/whales/scripts/')
import spTools as sp
import pcaO

#import scikits.audiolab as al
pl.rcParams.update({'figure.autolayout': True}) # exit plot layout

def pcaCalcs(fileN, plBasis = 0, svBasis = 0, # basis
             plWeights = 1, svWeights = 0, # weights
             plDataInPCB = 1, svDataInPCB = 0, numPC = 10,  # data
             plChopedPeak = 0, sRate=192000):

    '''
    plots the average of the variance keept by the k-th PC
    fileN = ceps file (cols)time x ceps "<file>-ceps.dat"
    '''
    
    cepsM = np.loadtxt(fileN)#.T # time runns horizontally

    # alighn, so that time runs horizontally, this is the way we plot cepstrograms
    Nc, Nt = np.shape(cepsM)
    if Nc>Nt: 
        '''
        the number of time steps should be larger that the number of cc    
        if this is not the case we invert the ceps matrix
        '''
        cepsM = cepsM.T
        print "matrix alighned:", Nt, "cesptral coeffis", Nc, "time steps" 

    baseN = os.path.splitext(fileN)[0]
    print baseN
   
    N = len(cepsM.T) #time points
    tf = N/(1.0*sRate) # time units
    
    pca = pcaO.pca0mean(cepsM.T) # vertical time

    if(plBasis):
        print "ploting PCA basis\n"
        Tmatrix = pca.vectors()
        plotPCbasis( Tmatrix, baseN)

    if(plWeights):
        print "ploting PCA weights\n"
        wei = pca.weights()
        plotPCweights( wei, baseN)

    if(plDataInPCB):
        print "ploting data in PC\n"
        timePCdata = pca.dataInMyBasis(cepsM.T).T # transpose to leave the time run in the horizontal axis
        plotInPC(timePCdata, baseN, numPC, sRate = sRate)


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
    pl.clf()



#------------- plot data in PC's ----------------------

def plotInPC(data, baseN, yM=10, sRate=192000): #, pc1=1, pcf=128, cci=1, ccf=128):
    '''
    
    '''
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

    pl.clf()


#------------- plot weights ----------------------

def plotPCweights( vec, baseN, r_tresh=0.9): #, pci=1, pcf=128):
    '''
    this function plots the comulative fraction of variance that is preserved as a function if the PC 
    '''

    deno =  vec.sum() # norm
    
    for i in np.arange(1, len(vec)): # percent variance
        vec[i] = vec[i] + vec[i-1]# + vec[i] 
        
    vec /= deno # normalize
    y_low = 0
    print "vec", vec.max(), vec[0], vec[1], len(vec), vec[-1], deno
        
    # find i such that 99% of variance is preserved
    i = -1
    r = 0
    while True: # run over the weights
        i = i+1
        r = vec[i]
           # print i
        if ( r >= r_tresh and i<2): # less that 2 PC, then plot all
            i=len(vec)-1 # plot all the weights
            y_low = r
            print "i", i,r, "y_low", y_low
            break
        if ( r >= r_tresh and i>1): # more than 2 PC, pl only these two
            y_low = vec[0]
            print 'i', i, r, y_low
            break
        
    # plot 
    #  pl.clf()
    fig = pl.figure(figsize=(6,5))
    fig.add_subplot(111)
    pl.plot(vec[:i])
    pl.xlabel("PC index", fontsize=18)
    pl.ylabel("r", fontsize=18)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.ylim(y_low,1)
    xM = i-1
   
    print "i =", i, vec[i]
    
    outF = baseN+"PCweitghts.png"
    print outF
    pl.savefig(outF)
    pl.clf()
    del vec