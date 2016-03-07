#!/usr/bin/python
from __future__ import print_function, division # compatibility with python 3
import numpy as np
#import pylab as pl
#import matplotlib
#import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
#import sets
import ast

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.expanduser('~/whales/scripts/NLP/')))
#import sequencesO_beta as seqs 
#import ngramO_beta as gr2
import myStatistics_beta as sts

#sys.path.append(os.path.abspath(os.path.expanduser('~/python/plottingTools/')))
#import colormap_adjust as cbAdj # set zero to white @density plot

"""
this script computes the chissquare test for the bigrams
 / NLP / hypothesisTesting / notebook /chiSquaretest.ipynb
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
parser.add_argument("-ch", "--checkIn", type=bool, default = True, 
                    help = "checks the input files")
parser.add_argument("-o", "--outFile", type=str, default = '', 
                    help = "Output file name if other than the automatic"
                    "outherwise a file name will be generated from the input.")
parser.add_argument("-pc", "--pcValue", type = float, default = 0.01, 
                    help = "crtitical p-value 0.01 (1% confidence)")
parser.add_argument("-af", "--appFile", type = str, default = '',
                    help = "file into which we append the summary")

###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
###########################################################################

##### ASSIGMENTS #####
args = parser.parse_args()
inF = os.path.abspath(os.path.expanduser(args.csvInF))
inDict = os.path.abspath(os.path.expanduser(args.dictF))
outFile =  args.outFile # don't expand utill later, to chek if string is empty
tolZero = args.toleranceZero
checkIn = args.checkIn
pcValue = args.pcValue
appFile = os.path.abspath(os.path.expanduser(args.appFile))
#test = 

# CHECK INPUT, that 'dists' and 'dict' come from the same group
if checkIn:
    assert(os.path.dirname(inF) == os.path.dirname(inDict))
    com1 = os.path.splitext( os.path.basename (inF))[0]
    com2 = os.path.splitext(os.path.basename(inDict))[0]
    print("\ncommon end:\n", com1,"\n", com2)
    assert(com2.endswith(com1))

##### FILE HANDLING #####
outDN = os.path.join( os.path.dirname(inF), 'confidenceIntervalsChi/')
fileBN = os.path.basename(inF).split('.')[0] # take the base name and remove the extension

## Check dir or create it ##
if not os.path.isdir( outDN ):
    print( "creating out dir:", outDN )
    os.mkdir( outDN )

if( not outFile ): outFile  = os.path.join(outDN, fileBN+"-chi2Test.txt")


##### READ DATA #####
print("OPENING", inF)
shuffDF = pd.read_csv(inF) # bigrams
i2c = [line.strip() for line in open(inDict, 'r')] # dict
c2i = {i2c[i] : i for i in range(len(i2c))}

##### COMPUTATIONS #####
f = open(outFile, 'w')
### observed frequencies -all
F_obs_all = np.nan_to_num(shuffDF.ix[0])
N_obs_all = int(np.sum(F_obs_all))
N_types_obs_all = len(F_obs_all[F_obs_all > 0]) # count the types of bigrams observed
bigramNames = shuffDF.columns.values
N_bigramsCombos_OUE = len(bigramNames)

### expected probs -all
shDF = shuffDF.ix[2:]
shDF.fillna(0, inplace=True) # nan --> 0
Nsh = len(shDF)
f.write("Number of shufflings: %s" % Nsh)
print( np.nansum(shuffDF.ix[2]))

### expected freqs -all
F_exp_all = 1.0 * np.sum(shDF) / Nsh
N_exp_all = np.sum(F_exp_all)
f.write("\nN_exp, _obs: %.1f %.1f"%( N_exp_all, N_obs_all))
f.write("\n# bigram types O but not E: %d"%(len(F_exp_all[F_exp_all == 0])))
assert(int(N_exp_all) - int(N_obs_all) < tolZero)  # The number of bigrams should be the same

### find the INI and END labels
bigramNames_tu = np.array([ast.literal_eval(item) for item in bigramNames])
no_INI = [False if c2i['__INI'] in item else True for item in bigramNames_tu]
no_END = [False if c2i['__END'] in item else True for item in bigramNames_tu]
## mask INI nor END bigrams
no_INI_no_END = np.logical_and(no_END, no_INI)
bigramNames_se = pd.Series(bigramNames)

### find rare bigrams
IO_rareBigrams_obs = F_obs_all < 5
IO_rareBigrams_exp = F_exp_all < 5
IO_rareBigrams = np.logical_or(IO_rareBigrams_exp, IO_rareBigrams_obs) # mask

### set observed and expected frequencies for the test
F_obs = F_obs_all[np.logical_and(no_INI_no_END, ~ IO_rareBigrams)]
F_exp = F_exp_all[np.logical_and(no_INI_no_END, ~ IO_rareBigrams)]
# define rare bigrams class
rareBigramsClass_obs = np.sum(
                    F_obs_all[np.logical_and(no_INI_no_END, IO_rareBigrams)])
rareBigramsClass_exp = np.sum(
                    F_exp_all[np.logical_and(no_INI_no_END, IO_rareBigrams)])
F_obs["rareBigr"] = rareBigramsClass_obs
F_exp["rareBigr"] = rareBigramsClass_exp
assert(len(F_obs) == len(F_exp)) # check lenghts of the obs and exp frequencies
assert( np.sum(F_obs) - np.sum(F_exp) < tolZero)

f.write("\nbigrams: %s"%" ".join(F_obs.keys()))

### test
df = len(F_obs) - 1 # the number of observed frequencies minus one
f.write("\ndf %d" % df)

### chi square
X2, p_chi = st.chisquare(F_obs.values, F_exp.values)
f.write("\n--- chi-square {} ---".format(X2))
f.write("\np-value {}".format(p_chi))
f.write("\nchi2 tabular value (for p=%.3f) =  %.2f" % (pcValue, sts.tabularChiSquare(pcValue, df)))
f.write("\nREJECT H0!\n") if pcValue > p_chi else f.write("\nCannot reject H0!!!\n")

### G-test
G, p_G = sts.Gstatistics(F_obs.values, F_exp.values)
f.write("\n--- G-test {} ---".format(G))
f.write("\np-value {}".format(p_G))
f.write("\nchi2 tabular value (for p=%.3f) =  %.2f" % (pcValue, sts.tabularChiSquare(pcValue, df)))
f.write("\nREJECT H0!\n") if pcValue > p_G else f.write("\nCannot reject H0!!!\n")

### sumary file
if args.appFile: 
    g = open(appFile, "a")
    g.write("{0} df={1}, p-value(c, chi, G)={2}, {3:0.3f}, {4:0.3f} => ".format(com1, df, pcValue, p_chi, p_G))
    #g.write("chi-p-value={0:0.3f} ".format(p_chi))
    #g.write("G-p-value={0:0.23} ".format(p_G))
    g.write("1") if pcValue > p_chi else g.write("0")
    g.write("1\n") if pcValue > p_G else g.write("0\n")



print("OUT FILE:", appFile)
print("OUT FILE:", outFile)

f.close()
g.close()
sys.exit()
