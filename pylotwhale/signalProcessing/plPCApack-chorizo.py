#!/usr/bin/python

import pylab as pl
import numpy as np
import os
import sys

sys.path.append('/home/florencia/whales/scripts/')
import pcaO
import plotTools

#import scikits.audiolab as al

pl.rcParams.update({'figure.autolayout': True}) #leave space for the Big labels

def pcaCalcs(fileN, plBasis = 1, plWeights = 1, r_tresh = 0.99999, sRate=192000):

    '''
reads the cepstrum
    plots the average of the variance keept by the firts k PC
    
    fileN = ceps file (cols)time x ceps "<file>-ceps.dat"
    
    '''
    cepsM = np.loadtxt(fileN).T # time runns horizontally (as in the spectrograms)
    baseN = os.path.splitext(fileN)[0]
    print baseN
   
    N = len(cepsM.T) #time points
    tf = N/(1.0*sRate)
    
    pca = pcaO.pca0mean(cepsM.T) # vertical time


#------------- plot basis ----------------------
    if(plBasis):
	print 'ploting basis'
        Tmatrix = pca.vectors()
        
        fig = pl.figure(figsize=(6,5))
        ax = fig.add_subplot(111)
        cax = pl.imshow(Tmatrix, extent=[1,128,1,128], interpolation = 'nearest', aspect ='auto')
        
        ax.set_xlabel( 'PC vector', fontsize=16 )
        ax.set_ylabel( 'cepstral coefficient ', fontsize=16 )
        ax.tick_params(axis='both', labelsize='x-large')  #set_xticks(ticks, fontsize=24)

   # Set x ticks
      #  pl.xticks(np.linspace(1,xM,xM,endpoint=True))
    
        outF = baseN+"PCbasis.png"
        print outF
        pl.savefig(outF)
        
        pl.clf()

#------------- plot weights ----------------------
    if(plWeights):
	print 'ploting weights'
       
        vec = pca.weights()
        deno =  vec.sum()
       
        for i in np.arange(1,len(vec)):
            vec[i] = vec[i] + vec[i-1]# + vec[i] 
          
        vec /= deno
        y_low = 0
        
	print "vec", vec.max(), vec[0], vec[1], len(vec), vec[-1], deno
       # find i such that 99% of variance is preserved

        i=-1
        r=0
        while True: # run over th e weights
            i = i+1
            r = vec[i]
           # print i
            if ( r >= r_tresh and i<2): # less that 2 PC
                i=len(vec)-1 # plot all the weights
                y_low = r
                print "i", i,r, "y_low", y_low
                break
            if ( r >= r_tresh and i>1): # more than 2 PC
                y_low = vec[0]
                print 'i', i, r, y_low
                break
           
        # plot or spit
#        if(i>1):
        pl.plot(vec[:i])
        pl.xlabel("PC index", fontsize=18)
        pl.ylabel("r", fontsize=18)
        pl.xticks(fontsize=18)
        pl.yticks(fontsize=18)
        pl.ylim(y_low,1)
        xM = i-1
            #pl.xticks(np.linspace(1,xM,xM,endpoint=True))
        print "i =", i, vec[i]
            
 #       else:
  #          print "i =", i, vec[i]
           
    #pl.show() #stops the script until you close the plot window
        outF = baseN+"PCweihgts.png"
        print outF
        pl.savefig(outF)
        pl.clf()
	del vec


