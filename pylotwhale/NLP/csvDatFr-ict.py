#!/usr/bin/python
from __future__ import print_function, division
import matplotlib
#import numpy as np
import argparse
import os
import sys
import pandas as pd


sys.path.append(os.path.abspath(os.path.expanduser('~/whales/scripts/NLP/')))
import sequencesO_beta as seqs
#import ngramO_beta as gr2
#sys.path.append(os.path.abspath(os.path.expanduser('~/python/plottingTools/')))
#import colormap_adjust as cbAdj  # set zero to white

"""
this script computes the bigram probability matrix and the shuffled
 version and computes the diference among this tow
"""

parser = argparse.ArgumentParser(description='plots the intercall times')  # takes the input and interprets it

parser.add_argument("csvInF", type=str, help="data frame file name [.csv], "
"Ex: /home/florencia/whales/NLP/NPWvocalRepertoire/wPandas/NPWVR-seqsData.csv")

parser.add_argument("groupNs", type=list, help="group names continuously, ex: FGH")

parser.add_argument("-f", "--feature", type=str, default='call', 
                    help="feature to count")
                    
parser.add_argument("-minC", "--minNumCalls", type=int, default = 5,
                    help = "minimum numberof calls to preceed")

#parser.add_argument("-m", "--more", type=int, help = "feature to count")

parser.add_argument("-d", "--date", type=str, default = "", help="date yymmdd")

parser.add_argument("-ta", "--tape", type=str, default = "", 
                    help="tape (recording) name, default, no constraint"
                    "all - plots all the tapes separately, <tape>")
                    
parser.add_argument("-s", "--plFontSize", type=int, default=20, 
                    help = "Default font size for the plots")
                    
parser.set_defaults(ictimes=False) # default 


###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
###########################################################################

##### ASSIGMENTS #####
args = parser.parse_args()
inF = args.csvInF
groupNs = args.groupNs  # list
feature = args.feature
date = args.date
tape = args.tape

matplotlib.rcParams.update({'font.size': args.plFontSize})

constraintType = 'recording'

print( "groups:", groupNs)
print( "\ntape:", tape)
#saveD = args.saveDat ## save data

##### FILE HANDLING #####
outDN = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(inF))), 'ict')
fileBN = os.path.basename(inF).split('.')[0]  # take the base name & remove the extension
outF0 = os.path.join(outDN, fileBN)
print( "out dir:", outDN, "\nfile BN:", fileBN)

## Check dir or create
if not os.path.isdir(outDN):
    print( "creating out dir:", outDN)
    os.mkdir(outDN)

##### COMPUTATIONS #####

def runIter(df0, constraintType, ouF, constrLi=[]):
    '''
    all getrent
    prints the shuffled disttributions for different constraint combinations.
    If the constrLi is empty, then we iterate over all the posibe constrains of
    the given constrintType. Ex: all the posible tapes
    < constrLi, is a list with the constrints we want to apply
    '''
    if len(constrLi)==0: constrLi = list(set(df0[constraintType]))
        
    for case in constrLi:
        df, baseN = seqs.constrainDF(df0, constraintType, case, ouF)

        outF = baseN + '-ictimes.pdf'
        print(outF)
        if len(df)>10:
            print("#calls", len(df))
            seqs.plTimeIntervals(df, frac=0.75, outFig=outF)

     
### read data
     
datB0 = pd.read_csv(inF, dtype = {'date' : str})

for groupN in groupNs:
    outF = outF0 + '_GR%s' % (groupN)
    df = datB0.loc[datB0['group'] == groupN]
    print("#calls:", len(df))
    print(groupN, outF)
    
    if tape=='all':
        #print("all tapes", tau, tau[-1])
        runIter(df, 'recording', outF)
    elif tape:
        print("tape:", tape)
        runIter(df, 'recording', outF, [tape])        
    

                             
                              
sys.exit()                             
                
      