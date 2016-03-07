#!/usr/bin/python
from __future__ import print_function # compatibility with python 3
import numpy as np
#import pylab as pl
import matplotlib
#import matplotlib.pyplot as plt
import pandas as pd
#import scipy.stats as st
#import sets
import ast

import argparse
import os
import sys

sys.path.append('/home/florencia/whales/scripts/NLP/')
#import sequencesO_beta as seqs 
import ngramO_beta as ngr
#import myStatistics_beta as sts

sys.path.append('/home/florencia/python/plottingTools/')
#import colormap_adjust as cbAdj # set zero to white @density plot

"""
this script computes the chissquare test for the bigrams
"""

parser = argparse.ArgumentParser(description = 'computes chi-square test')  
parser.add_argument("csvInF", type=str, help = "data frame file name [.csv]."
                    "Ex: ~/whales/NLP/NPWvocalRepertoire/wPandas-shuffled/"
                    "shuffledProbDist1000_B.csv")
parser.add_argument("dictF", type=str, help = "data frame file name [.dat]."
                    "Ex: ~/whales/NLP/NPWvocalRepertoire/wPandas-shuffled/"
                    "dictionary_shuffledProbDist1000_B.dat")
parser.add_argument("-tz", "--toleranceZero", type=float, default = 1e-5, 
                    help = "tolerance zero.")
parser.add_argument("-ch", "--checkIn", type=int, default = 1, 
                    help = "checks the input files")
parser.add_argument("-o", "--outFile", type=str, default = '', 
                    help = "Output file name if other than the automatic"
                    "outherwise a file nam will be generated from the input.")
#parser.add_argument("-pc", "--pcValue", type = float, default = 0.01, 
 #                   help = "crtitical p-value 0.01 (1% confidence)")

## RENAME CALLS BY: --flag and --no-flag
parser.add_argument("-rnCalls","--renameCalls", dest='renameCalls', action='store_true')
parser.add_argument("-no-rnCalls", "--no-renameCalls", dest='renameCalls', action='store_false')
parser.set_defaults(renameCalls=False)                    

parser.add_argument("-plH","--plDiffRepHist", dest='plDiffRepHist'
                    'bigram histogram', action='store_true')
parser.add_argument("-no-plH", "--no-plDiffRepHist", dest='plDiffRepHist', action='store_false')
parser.set_defaults(plDiffRepHist=True)
                    
parser.add_argument("-s", "--plFontSize", type=int, default=20, 
                    help = "Default font size for the plots")
parser.add_argument("-pT", "--plTresh", type=int, default=10, 
                    help = "plotting treshold")

###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
########################## #################################################

##### ASSIGMENTS #####
args = parser.parse_args()
inF = os.path.abspath(os.path.expanduser(args.csvInF))
inDict = os.path.abspath(os.path.expanduser(args.dictF))
outFile = args.outFile
tolZero = args.toleranceZero
checkIn = args.checkIn
#pcValue = args.pcValue
renameCalls = args.renameCalls
plHist = args.plDiffRepHist
plTresh = args.plTresh

matplotlib.rcParams.update({'font.size': args.plFontSize})

# CHECK INPUT, that 'dists' and 'dict' come from the same group
if checkIn:
    assert(os.path.dirname(inF) == os.path.dirname(inDict))
    com1 = os.path.splitext( os.path.basename (inF))[0]
    com2 = os.path.splitext(os.path.basename(inDict))[0]
    print("\ncommon end:\n", com1,"\n", com2)
    assert(com2.endswith(com1))

##### FILE HANDLING #####
outDN = os.path.join( os.path.dirname(inF), 'bigramMatrices/')
fileBN = os.path.basename(inF).split('.')[0] # take the base name and remove the extension

## Check dir or create it ##
if not os.path.isdir( outDN ):
    print( "creating out dir:", outDN )
    os.mkdir( outDN )

# this script outputs two types of plots: outFile and outFile2
print("out file nothing expected:", outFile)    
if( outFile ): 
    baseN, Fext = os.path.splitext(outFile) 
    outFile = baseN + "-freqs" + Fext
else:
    outFile  = os.path.join(outDN, fileBN + "-freqs.pdf")
    if renameCalls: outFile = outFile.replace( "-freqs", "-freqs-rn")

outFile2 = outFile.replace( "-freqs", "-probs")
    #os.path.join(outDN, fileBN+"-obsVsShuff.pdf")


##### READ DATA #####
shuffDF = pd.read_csv(inF) # bigrams
i2c = [line.strip() for line in open(inDict, 'r')] # dict
c2i = {i2c[i] : i for i in range(len(i2c))}
IO_diffThanRandom = {} # dictionary of the hypothesis rejection
IO_assim = {} # symmetry hypothesis dictionary

##### COMPUTATIONS #####
### observed frequencies -all
F_obs_all = np.nan_to_num(shuffDF.ix[0])

bigramNames = shuffDF.columns.values
bigramNames_tu = np.array([ast.literal_eval(item) for item in bigramNames])
no_INI = [False if c2i['__INI'] in item else True for item in bigramNames_tu]
no_END = [False if c2i['__END'] in item else True for item in bigramNames_tu]
no_INI_no_END = np.logical_and(no_END, no_INI)

F_obs_no_INI_no_END = dict(F_obs_all[no_INI_no_END])    
F_obs_no_INI_no_END_tu = {ast.literal_eval(k) : F_obs_no_INI_no_END[k] 
                          for k in F_obs_no_INI_no_END.keys() }

freq_mtx = ngr.bigrams2matrix(F_obs_no_INI_no_END_tu, len(c2i))

### make-up
N_2grms_st = np.nansum(freq_mtx, axis = 1 )
mkup_arr = ngr.getElementsLargerThan(N_2grms_st, plTresh) 
i2c_mkUp = [i2c[i] for i in mkup_arr]
if renameCalls: i2c_mkUp = ["c%d"%i[0] for i in enumerate(mkup_arr)]

### frequencies
redA = ngr.reducedMatrix(freq_mtx, mkup_arr) # evaluate reduced matrix
ngr.plBigramM(redA, i2c_mkUp, cbarLim=(1, np.max(redA)), cbar=True, 
              outFig = outFile)
print("OUT:", outFile)

### probabilities
P_obs_all = np.nan_to_num(shuffDF.ix[1])
P_obs_no_INI_no_END = dict(P_obs_all[no_INI_no_END])
P_obs_no_INI_no_END_tu = { ast.literal_eval(k): P_obs_no_INI_no_END[k] for 
                            k in F_obs_no_INI_no_END.keys()}
P_mtx = ngr.bigrams2matrix(P_obs_no_INI_no_END_tu, len(c2i))
redA = ngr.reducedMatrix(P_mtx, mkup_arr) # evaluate reduced matrix
ngr.plBigramM(redA, i2c_mkUp, cbarLim=(np.min(redA[redA != 0]), 1) , outFig=outFile2)
print("minP (not 0):",np.min(redA[redA != 0]))
print("OUT:", outFile2)

### rep/diff-bigrams-histogram
if(plHist):
    A_diag = [ freq_mtx[i,i] for i in range(len(freq_mtx)) ] # repetitions
    A_diag_red = [ A_diag[i] for i in mkup_arr ] # mkUp reps

    # off diagonal  
    A_st = np.vstack( [ np.nansum(freq_mtx, axis = 1), np.multiply(-1, A_diag) ] )
    A_offD = np.nansum(A_st, axis = 0)   # changes
    A_offD_red = [ A_offD[i] for i in mkup_arr ] # mkUp diff
    outFile3 = outFile.replace( "-freqs", "-diffRepHist")

    ngr.barPltsSv([A_offD_red[:], A_diag_red[:]], i2c_mkUp[:],
                figN = outFile3, plLegend = 0, yTickStep=None)# ['different call', 'repetition'] ) # 
                #   plTit = r'%s,$\tau $= %s'%(groupN, tau), 

sys.exit()


