#!/usr/bin/python
from __future__ import print_function
import numpy as np
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib

sys.path.append(os.path.abspath(os.path.expanduser('~/whales/scripts/NLP/')))
import sequencesO_beta as seqs
#import ngramO_beta as gr2
#sys.path.append(os.path.abspath(os.path.expanduser('~/python/plottingTools/')))
#import colormap_adjust as cbAdj  # set zero to white

"""
this script computes the bigram probability matrix and the shuffled
 version and computes the diference among this tow
"""

parser = argparse.ArgumentParser(description='This script is for exploring the ' 
'sequences of calls in terms of \tau. More details in /wPandas/B_sequence_statistics.ipynb')  # takes the input and interprets it

parser.add_argument("csvInF", type=str, help="data frame file name [.csv], "
"Ex: /home/florencia/whales/NLP/NPWvocalRepertoire/wPandas/NPWVR-seqsData.csv")

parser.add_argument("groupNs", type=list, help="group names continuously, ex: FGH")

parser.add_argument("-T", "--tauInt", type=str, default="0:10", 
                    help="time interval treshold for the sequences definition")

parser.add_argument("-f", "--feature", type=str, default='call', 
                    help="feature to count")

parser.add_argument("-ta", "--tape", type=str, default = "all", 
                    help="tape (recording) name, default, no constraint"
                    "all - plots all the tapes separately, <tape>")
                    
parser.add_argument("-s", "--plFontSize", type=int, default=20, 
                    help = "Default font size for the plots")                    

parser.add_argument("-vmin", "--vmin", type=int, default=1, 
                    help = "lower threshold of the clr map")    

parser.add_argument("-mN", "--maxNgram", type=int, default=10, 
                    help = "largest N (Ngram) that is plotted")    
                    
parser.add_argument("-tr", "--thresholdFrac", type=int, default=4, 
                    help = "the histogram will be cutted according to this tr"
                    "all the ngrams with more than Ncalls/tr will have the"
                    "same color")    
###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
###########################################################################

##### ASSIGMENTS #####
args = parser.parse_args()
inF = args.csvInF
groupNs = args.groupNs  # list
feature = args.feature
clrTrFrac = args.thresholdFrac
vmin = args.vmin
maxNgram = args.maxNgram
#date = args.date
tape = args.tape
t0, tf = [int(item) for item in args.tauInt.split(":")]
tau  = np.arange(t0, tf)

matplotlib.rcParams.update({'font.size': args.plFontSize})

constraintType = 'recording'

print( "groups:", groupNs, "\ntime:", tau)
print( "\ntape:", tape)
#saveD = args.saveDat ## save data

##### FILE HANDLING #####
outDN = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(inF))), 'tauIter')
fileBN = os.path.basename(inF).split('.')[0]  # take the base name & remove the extension
outF0 = os.path.join(outDN, fileBN)
print( "out dir:", outDN, "\nfile BN:", fileBN)

## Check dir or create
if not os.path.isdir(outDN):
    print( "creating out dir:", outDN)
    os.mkdir(outDN)

##### COMPUTATIONS #####


def fancyClrBarPl(X, vmax, vmin, maxN=10, clrMapN='jet', clrBarGaps=15,
                  tickLabs=("min", "middle", "max"), outplN='',
                    plTitle=''):
    
    '''
    tickLabs=("<%d"%vmin, int(vmax), '>%d/%d'%(Ncalls, frac))
    '''

    fig, ax = plt.subplots()

    #colors setting
    cmap = plt.cm.get_cmap('jet', clrBarGaps)    # discrete colors
    cmap.set_under((0.9, 0.9, 0.8)) #min
    cmap.set_over((1, 0.6, 0.6)) #nan

    #plot
    cax=ax.imshow(X[:,:maxN], aspect ='auto', interpolation='nearest', 
               norm = colors.Normalize(vmin=vmin, vmax=vmax, clip = False),
               cmap=cmap)
    #labels
    ax.set_xlabel('N')
    ax.set_ylabel(r'$\tau$')
    if plTitle: ax.set_title(plTitle)

    #clrbar
    cbar = fig.colorbar(cax, extend='both')
    cbar.set_clim((vmin,vmax))
    cbar.set_ticks((vmin, int(vmax/2), vmax))
    cbar.set_ticklabels(tickLabs)
    
    if outplN: fig.savefig(outplN)


def runIter(df0, constraintType, tau, baseName, frac, constrLi=[]):
    '''
    all getrent
    iterates over the constrain list
    calculates the .
    If the constrLi is empty, then we iterate over all the posibe constrains of
    the given constrintType. Ex: all the posible tapes
    < constrLi, is a list with the constrints we want to apply
    '''
    
    # create constraint list
    if len(constrLi)==0: constrLi = list(set(df0[constraintType]))
    
    # iterate over the constraints (tapes)
    for case in constrLi:
        df, baseN = seqs.constrainDF(df0, constraintType, case, baseName)
        print(len(df))
        
        if (len(df)>50):
            X, Ncalls = seqs.tauNgramsMatrix(df, tau[0], tau[-1])
                
            vmax = Ncalls / frac
                #print(vmax, vmin, np.shape(X))
            ### clr plot
            tickLabs = ("<%d"%vmin, int(vmax/2), '>%d/%d'%(Ncalls, frac))
            outplN = baseN+'.pdf'
            seqs.fancyClrBarPl(X, vmax, vmin, maxN=maxNgram, tickLabs=tickLabs, 
                               outplN=outplN, plTitle='%s, #calls=%d'%(case, Ncalls))
            ### change plot
            diffX=np.abs(X[0:-1,:]-X[1:,:]) # where reordering happens
            e_vec = np.arange(len(X[0,:]))
            alpha = np.dot(diffX, e_vec)
            fig, ax = plt.subplots()
            ax.bar(np.arange(1,len(alpha)+1),alpha/2)
            ax.set_xlabel(r'$\tau$')
            ax.set_ylabel('#(reordered calls)')
            ax.set_title('%s'%case)
            outplN = baseN+'-reeordering.pdf'
            print(outplN)
            fig.savefig(outplN)



            
        else:
            print("to few calls in this tape")
            
            

### read data
datB0 = pd.read_csv(inF, dtype = {'date' : str})

for groupN in groupNs:
    outF = outF0 + '_GR%s' % (groupN)
    df = datB0.loc[datB0['group'] == groupN]
    print("#calls:", len(df))
    print(groupN, len(df), outF)
    if tape=='all':
        #print("all tapes", tau, tau[-1])
        runIter(df, 'recording', tau, outF, frac=clrTrFrac)
    elif tape:
        print("tape:", tape, tau)
        runIter(df, 'recording', tau, outF, [tape], frac=clrTrFrac)        
    

                             
                              
sys.exit()                             
                
      