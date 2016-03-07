#!/usr/bin/python

import numpy as np
#import pylab as pl
import matplotlib

import argparse
import os
import sys
import pandas as pd
import sets
import ast

sys.path.append('/home/florencia/whales/scripts/NLP/')
import sequencesO_beta as seqs 
import ngramO_beta as gr2
import myStatistics_beta as sts

sys.path.append('/home/florencia/python/plottingTools/')
import colormap_adjust as cbAdj # set zero to white @density plot

"""
this script computes the bigram probability matrix and the shuffled version and computes the diference among this tow
"""

parser = argparse.ArgumentParser(description = 'Reads the shuffled distributions csv'
                                ' (li 1 - bigram counts, l2 - bigram probs, l3:end - shuffled distributions.'
                                'Computes the confidence intervals and determines the relevance of the'
                                'probabilities give the sampling set. Plots a binary matrix with the '
                                'relevant probabilities.')  # takes the input and interprets it
parser.add_argument("csvInF", type=str, help = "data frame file name [.csv]. Ex: ~/whales/NLP/NPWvocalRepertoire/wPandas-shuffled/shuffledProbDist1000_B.csv")

parser.add_argument("dictF", type=str, help = "data frame file name [.dat]. Ex: ~/whales/NLP/NPWvocalRepertoire/wPandas-shuffled/dictionary_shuffledProbDist1000_B.dat")

parser.add_argument("-a", "--atLeastN2gr", type=int, default = 5, help = "Filter out bigrams with less that a occurences. Only while pltting.")

parser.add_argument("-s", "--plFontSize", type=int, default = 20, help = "DEfault font size for the plots")

parser.add_argument("-cb", "--plcbar", type = str, default = '', help = "do you whant to plot the cbar?")

parser.add_argument("-ex", "--imgExt", type = str, default = 'pdf', help = "format of the output images")


###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
###########################################################################

##### ASSIGMENTS #####
args = parser.parse_args()
inF = args.csvInF
inDict = args.dictF
minN2grms = args.atLeastN2gr
plcbar = args.plcbar
imExt = args.imgExt


matplotlib.rcParams.update({'font.size': args.plFontSize})


###mkUP_tresh = args

# CHECK INPUT, that 'dists' and 'dict' come from the same group
tau1 = inF.split('.')[-2][-1]
tau2 = inDict.split('.')[-2][-1]
print tau1, tau2
assert( tau1 == tau2 )
tau = tau1

group1 = inF.split('_')[-2]
group2 = inDict.split('_')[-2]
print group1, group2
assert( group1 == group2 )
groupN = group1

##### FILE HANDLING #####
outDN = os.path.abspath(os.path.expanduser(os.path.dirname(inF)))+'/confidenceIntervals/'
fileBN = os.path.basename(inF).split('.')[0] # take the base name and remove the extension
print "out dir:", outDN

## Check dir or create it
if not os.path.isdir( outDN ):
    print "creating out dir:", outDN 
    os.mkdir( outDN )

##### COMPUTATIONS #####

# data
df_distsAll = pd.read_csv(inF) 

# dict
i2c = [line.strip() for line in open(inDict, 'r')]
c2i = {i2c[i] : i for i in range(len(i2c))}

IO2grm_str = {item : 0 for item in df_distsAll.columns.values} # binary bigrams dictionary inicialization

relevantCols = seqs.select2grmsDict_lrgr_NotIni( df_distsAll.ix[0], 
                                                removeSet=[c2i['__INI'], c2i['__END']], 
                                                lgr=minN2grms ) # labels of the cols

print "HI", len(IO2grm_str), relevantCols

for thisCol in relevantCols:

    # computations
    realData = df_distsAll[ str(thisCol) ][1] # real value, unshuffled data
    shuffledDists = df_distsAll[ str(thisCol) ][2:] # shuffled distributions
    # yes or not 
    mu, h = sts.mean_confidence_interval(shuffledDists, confidence=0.999)
    x = sts.inORout(realData, mu, h)
    IO2grm_str[str(thisCol)] = x
    print "this col", thisCol, x, mu, h, realData


####### TUP DIC --> MATRIX #####
### I/O
IO2grm_tu = {ast.literal_eval(ky): IO2grm_str[ky] for ky in IO2grm_str.keys()} # di ct string keys --> tuple
IO_mtx = gr2.bigrams2matrix( IO2grm_tu, c2i ) # IO - matrix
### Probs
biGrms_P_str = dict(df_distsAll.ix[1]) # dict, 2-grams counts
biGrms_P = {ast.literal_eval(ky): biGrms_P_str[ky] for ky in biGrms_P_str.keys()} # keys str --> tu
### counts
biGrms_counts_str = dict(df_distsAll.ix[0]) # 2-grams counts dict
biGrms_counts = {ast.literal_eval(ky): biGrms_counts_str[ky] for ky in biGrms_counts_str.keys()} # keys str --> tu
biGrms_counts_mtx = gr2.bigrams2matrix( biGrms_counts, c2i) # 2-grams matrix
## ignore spurius bigrms _X, X_
biGrms_counts_mtx[c2i['__INI'], :]=0
biGrms_counts_mtx[:, c2i['__INI']]=0
biGrms_counts_mtx[c2i['__END'], :]=0
biGrms_counts_mtx[:, c2i['__END']]=0
# print IO2grm_tu, "\n", c2i

### make-up indexes
N_2grms_arr = np.nansum( biGrms_counts_mtx, axis = 1 ) # mk-UP arr
mkUP_indexes = gr2.getElementsLargerThan( N_2grms_arr, minN2grms) # mk-UP indexes
### leave __INI and __END out of the bigram counts
if c2i['__INI'] in mkUP_indexes: mkUP_indexes.remove(c2i['__INI']) # remove __INI
if c2i['__END'] in mkUP_indexes: mkUP_indexes.remove(c2i['__END']) # remove __END

mkUP_mtx = gr2.reducedMatrix(IO_mtx, mkUP_indexes) # mk-UP matrix
mkUP_i2c = [i2c[i] for i in mkUP_indexes] # mk-IP dict
## ploting
#plName = outDN + fileBN + 'IO2grm.eps'

print "\ntest\n", len(mkUP_mtx), np.shape(mkUP_mtx), N_2grms_arr, "cbar?", plcbar

gr2.plBigramM(mkUP_mtx, mkUP_i2c, Bname = fileBN + '-IO-bar%s'%str(plcbar), 
              figDir=outDN, ext=imExt, plTitle=groupN, cbar=False )
#print( np.shape(IO_mtx), np.shape(mkUP_mtx), len(mkUP_i2c), len(c2i),
#"mkUP", mkUP_indexes, N_2grms_arr)
gr2.plBigramM(IO_mtx, i2c, Bname = fileBN+'-IO-full', figDir = outDN, ext=imExt,
              plTitle = groupN)
gr2.plBigramM(gr2.reducedMatrix(biGrms_counts_mtx, mkUP_indexes), mkUP_i2c, 
              Bname = fileBN+'-counts-cb%s'%plcbar, figDir = outDN, ext=imExt,
              plTitle = groupN, cbar=plcbar)
#gr2.plBigramM(gr2.reducedMatrix(gr2.bigrams2matrix(biGrms_P, c2i), mkUP_indexes), mkUP_i2c, Bname = fileBN+'-probss', figDir = outDN) # via data frame

gr2.plBigramM(gr2.reducedMatrix(gr2.bigramProbabilities(biGrms_counts_mtx).bigramMatrix,
                                mkUP_indexes), mkUP_i2c, Bname=fileBN + '-probs', 
                                figDir = outDN, ext=imExt, 
                                plTitle = r'%s,  $\tau$=%s'%(groupN, tau), 
                                cbarLim = (0,1), cbarTicks=[0, 0.25, 0.5, 0.75, 1], 
                                cbarOrientation ='horizontal', cbar = False) # via normaliz

### histograms #-bigrams counts
# diagonal elements
A = biGrms_counts_mtx
A_diag = [ A[i,i] for i in range(len(A)) ] # repetitions
A_diag_red = [ A_diag[i] for i in mkUP_indexes ] # mkUp reps
# off diagonal
A_st = np.vstack( [ np.nansum(A, axis = 1), np.multiply(-1, A_diag) ] )
A_offD = np.nansum(A_st, axis = 0)   # changes
A_offD_red = [ A_offD[i] for i in mkUP_indexes ] # mkUp diff
# plot
#stt = 1 if mkUP_i2c[0] == '__INI' else 0
gr2.barPltsSv([A_offD_red[:], A_diag_red[:]], mkUP_i2c[:], 
                plTit = r'%s,$\tau $= %s'%(groupN, tau), 
                figN = outDN + fileBN + '-hist%s'%imExt, plLegend = 0)# ['different call', 'repetition'] ) # 
print A_offD_red[1:], A_diag_red

# SAVE as data frame
df = pd.DataFrame( IO2grm_str, index = range(1))
outFN = outDN + fileBN + '-IO2grms.csv'
x = df.columns
y = df_distsAll.columns
print outFN, "dfs:", len(x), len(y), set(x).difference(set(y))

df = pd.concat([ df_distsAll[:2], df ], ignore_index = True)

df.to_csv(outFN, index = False)

sys.exit()
