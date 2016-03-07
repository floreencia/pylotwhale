#!/usr/bin/python
from __future__ import print_function # compatibility with python 3
import numpy as np
#import pylab as pl
import matplotlib
import matplotlib.pyplot as plt
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
import myStatistics_beta as sts

#sys.path.append('/home/florencia/python/plottingTools/')
#import colormap_adjust as cbAdj # set zero to white @density plot

"""
this script computes the chissquare test for the bigrams
"""

parser = argparse.ArgumentParser(description = 'computes difference of'
                             'proportions test')  
parser.add_argument("csvInF", type=str, help = "data frame file name [.csv]."
                    "Ex: ~/whales/NLP/NPWvocalRepertoire/wPandas-shuffled/"
                    "shuffledProbDist1000_B.csv")
parser.add_argument("dictF", type=str, help = "data frame file name [.dat]."
                    "Ex: ~/whales/NLP/NPWvocalRepertoire/wPandas-shuffled/"
                    "dictionary_shuffledProbDist1000_B.dat")
parser.add_argument("-tz", "--toleranceZero", type=float, default = 1e-5, 
                    help = "tolerance zero.")
parser.add_argument("-ch", "--checkIn", type=bool, default = True, 
                    help = "checks the input files")
parser.add_argument("-o", "--outFile", type=str, default = '', 
                    help = "Output file name if other than the automatic"
                    "outherwise a file nam will be generated from the input.")
parser.add_argument("-pc", "--pcValue", type = float, default = 0.01, 
                    help = "crtitical p-value 0.01 (1% confidence)")
                    
## RENAME CALLS BY: --flag and --no-flag
parser.add_argument("-rnCalls","--renameCalls", dest='renameCalls', action='store_true')
parser.add_argument("-no-rnCalls", "--no-renameCalls", dest='renameCalls', action='store_false')
parser.set_defaults(renameCalls=False) # default uses Heikes call names

parser.add_argument("-s", "--plFontSize", type=int, default = 10, 
                    help = "Default font size for the plots")
parser.add_argument("-nt", "--nTresh", type=int, default = 1, 
                    help = "Minumium number of observations in"
                    "of a bigram in order to perform the test.")
parser.add_argument("-pT", "--plTresh", type=int, default = 10, 
                    help = "plotting treshold. output plots (bigram matrices)"
                    "will only show the calls appearing in more than"
                    "plTresh bigrams")
parser.add_argument("-ext", "--figExt", type=str, default = '.pdf', 
                    help = "extension of the figures")

###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
###########################################################################

##### ASSIGMENTS #####
args = parser.parse_args()
inF = os.path.abspath(os.path.expanduser(args.csvInF))
inDict = os.path.abspath(os.path.expanduser(args.dictF))
outFile =  args.outFile
tolZero = args.toleranceZero
checkIn = args.checkIn
pcValue = args.pcValue
renameCalls = args.renameCalls
nTresh = args.nTresh
plTresh = args.plTresh
figExt = args.figExt

matplotlib.rcParams.update({'font.size': args.plFontSize})

# CHECK INPUT, that 'dists' and 'dict' come from the same group
if checkIn:
    assert(os.path.dirname(inF) == os.path.dirname(inDict))
    com1 = os.path.splitext( os.path.basename (inF))[0]
    com2 = os.path.splitext(os.path.basename(inDict))[0]
    print("\ncommon end:\n", com1,"\n", com2)
    assert(com2.endswith(com1))

##### FILE HANDLING #####
outDN = os.path.join( os.path.dirname(inF), 'shuffledVsObs_diffProportions/')
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
    outFile  = os.path.join(outDN, fileBN + "-freqs" + figExt)
    ## renamed calls
    if renameCalls: outFile = outFile.replace( "-freqs", "-freqs-rn")
        
outFile2 = outFile.replace( "-freqs", "-jointProbs")

outFile3 = outFile.replace( "-freqs", "-condProbs")


##### READ DATA #####
shuffDF = pd.read_csv(inF, dtype = {'date': str}) # bigrams
i2c = [line.strip() for line in open(inDict, 'r')] # dict
c2i = {i2c[i] : i for i in range(len(i2c))}
IO_diffThanRandom = {} # dictionary of the hypothesis rejection
IO_assim_joint = {} # symmetry hypothesis dictionary for joint pobs
IO_assim_cond = {} # symmetry hypothesis dictionary for conditional probs.


##### COMPUTATIONS #####
### observed frequencies -all
F_obs_all = np.nan_to_num(shuffDF.ix[0])
F_obs_all_tu = { ast.literal_eval(k): F_obs_all[k] for k in dict(F_obs_all).keys()}
N_obs_all = int(np.sum(F_obs_all))
N_types_obs_all  = len(F_obs_all[F_obs_all > 0]) # count the types of bigrams observed
bigramNames = shuffDF.columns.values
N_bigramsCombos_OUE = len(bigramNames)

### observed probabilities
P_obs_all = pd.Series(np.nan_to_num(shuffDF.ix[1]))
P_obs_all_tu = { ast.literal_eval(k): P_obs_all[k] for k in F_obs_all.keys()}

### sort bigrams
bigramNames_tu = np.array([ast.literal_eval(item) for item in bigramNames])
no_INI = [False if c2i['__INI'] in item else True for item in bigramNames_tu]
no_END = [False if c2i['__END'] in item else True for item in bigramNames_tu]
no_INI_no_END = np.logical_and(no_END, no_INI)
F_obs_no_INI_no_END = dict(F_obs_all[no_INI_no_END])
sortedBigrams = sorted(F_obs_no_INI_no_END, key = F_obs_no_INI_no_END.get, 
                       reverse = True) # sort the bigrams by frequency

### expected probs -all
shDF = shuffDF.ix[2:]
shDF.fillna(0, inplace=True) # nan --> 0
Nsh = len(shDF)  # N_shufflig iterations
total_sh = np.sum(shDF)  # sum over all the observatins from the shuffling
print("Number of shufflings:", Nsh)  #
#print("", total_sh)

for thisBigram  in sortedBigrams:
    if( not F_obs_no_INI_no_END[thisBigram] >= nTresh): # all bigrams w/ freq smaller than 5 or nans
        IO_diffThanRandom[ast.literal_eval(thisBigram)] = np.nan
    else:
        # get all the bigrams starting with A
        thisBigram_tu = ast.literal_eval(thisBigram)
        IO_N_obs = [item[0] == thisBigram_tu[0] for item in bigramNames_tu]
        #print("# bigram-types starting w/ call "
         #   "\"{}\": {}".format(i2c[thisBigram_tu[0]], sum(IO_N_obs)))
        N_starting_wThisBigram = np.sum(F_obs_all[IO_N_obs])
        #print("# bigrams starting", N_starting_wThisBigram)
        
        ### (2): H0: P(A|B) = P*(A|B)
        n1 = np.sum(F_obs_all[IO_N_obs])
        p1 = F_obs_all[thisBigram]/n1
        n2 = np.sum(total_sh[IO_N_obs])
        p2 = total_sh[thisBigram]/n2
        #print("p1:", p1, "\nn1:", n1, "\np2:", p2, "\nn2", n2)
        IO_diffThanRandom[ast.literal_eval(thisBigram)] =  sts.testDiffProportions(p1, p2, n1, n2, pcValue=0.99)[0]
        
        ### (4-join) H0: P(A,B) = P(B,A)
        IO_N_obs1 = [item[0] == thisBigram_tu[0] for item in bigramNames_tu]
        IO_N_obs2 = [item[0] == thisBigram_tu[1] for item in bigramNames_tu]
        n1 = N_obs_all  # np.sum(F_obs_all[IO_N_obs1])
        p1 = 1.0*F_obs_all_tu[thisBigram_tu[0], thisBigram_tu[1]]/N_obs_all
        n2 = N_obs_all  # np.sum(F_obs_all[IO_N_obs2])
        if (thisBigram_tu[1], thisBigram_tu[0]) in F_obs_all_tu.keys():
            p2 = 1.0*F_obs_all_tu[thisBigram_tu[1], thisBigram_tu[0]]/N_obs_all
        else:
            p2 = 0
        IO_assim_joint[ast.literal_eval(thisBigram)] = \
                    sts.testDiffProportions(p1, p2, n1, n2, pcValue=0.99)[0]
        
        ### (4-cond) H0: P(A|B) = P(B|A)
        n1 = np.sum(F_obs_all[IO_N_obs1])
        p1 = P_obs_all_tu[thisBigram_tu[0], thisBigram_tu[1]]
        n2 = np.sum(F_obs_all[IO_N_obs2])        
        if (thisBigram_tu[1], thisBigram_tu[0]) in P_obs_all_tu.keys():
            p2 = P_obs_all_tu[thisBigram_tu[1], thisBigram_tu[0]]        
        else:
            p2 = 0
        IO_assim_cond[ast.literal_eval(thisBigram)] = \
                    sts.testDiffProportions(p1, p2, n1, n2, pcValue=0.99)[0]

### reject/accept H0 matrix
IO_mtx = ngr.bigrams2matrix(IO_diffThanRandom, len(c2i))
### plotting make-up
F_obs_no_INI_no_END_tu = { ast.literal_eval(k): F_obs_no_INI_no_END[k] for k in F_obs_no_INI_no_END.keys() }
freq_mtx = ngr.bigrams2matrix(F_obs_no_INI_no_END_tu, len(c2i))
N_2grms_st = np.nansum(freq_mtx, axis = 1 )
mkup_arr = ngr.getElementsLargerThan(N_2grms_st, plTresh) # hyp const n >= 30
i2c_mkUp = [i2c[i] for i in mkup_arr]
if renameCalls: i2c_mkUp = ["c%d"%i[0] for i in enumerate(mkup_arr)]
redA = ngr.reducedMatrix(IO_mtx, mkup_arr) # evaluate reduced matrix
ngr.plBigramM(redA, i2c_mkUp, clrMap=plt.cm.bwr_r, cbarLim = (-1, 1), 
              cbar=False, outFig = outFile)
print("OUT:", outFile)

asm_mtx_j = ngr.bigrams2matrix(IO_assim_joint, len(c2i))
redA = ngr.reducedMatrix(asm_mtx_j, mkup_arr) 
ngr.plBigramM(redA, i2c_mkUp, clrMap=plt.cm.bwr_r, cbarLim = (-1, 1), 
              cbar=False, outFig = outFile2)
print("OUT:", outFile2)

asm_mtx_c = ngr.bigrams2matrix(IO_assim_cond, len(c2i))
redA = ngr.reducedMatrix(asm_mtx_c, mkup_arr) 
ngr.plBigramM(redA, i2c_mkUp, clrMap=plt.cm.bwr_r, cbarLim = (-1, 1), 
              cbar=False, outFig = outFile3)
print("OUT:", outFile3)



sys.exit()
