#!/usr/bin/python
from __future__ import print_function
import numpy as np
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
 version and computes the difference among these two
"""

parser = argparse.ArgumentParser(description='Shuffles the data N times and'
' computes the bigram probalilities from group groupNs of the given csv file.'
' Generates a csv with the probablilities, being the first two comlumns:'
' the bigram counts and the bigram probabilities of the unshuffled data. '
'Also generates a .dat file with the names of the calls, to be read'
' as index-2-call dictionary.')  # takes the input and interprets it

parser.add_argument("csvInF", type=str, help="data frame file name [.csv], "
"Ex: /home/florencia/whales/NLP/NPWvocalRepertoire/wPandas/NPWVR-seqsData.csv")

parser.add_argument("groupNs", type=list, help="group names continuously, ex: FGH")

parser.add_argument("-a", "--atLeastN2gr", type=int, default=5, help="filter out bigrams"
" with less that a occurences. They are only filtered out at the ploting"
". the bigrams are computed taking them into accout.")

parser.add_argument("-N", "--Nshuffl", type=int, default=0, help="list of group names")

parser.add_argument("-T", "--time_T", type=int, default=5, help="time"
                    " treshold for the sequences definition")

parser.add_argument("-f", "--feature", type=str, default='call', 
                    help="feature to count")

parser.add_argument("-n", "--normalize", type=int, default=1, 
                    help = "shuffled bigramsare normalized by default." 
                    "If you need the frequencies instead, i.e. for the"
                    "ch^2 test, then change it to cero.")
                    
parser.add_argument("-minC", "--minNumCalls", type=int, default = 5,
                    help = "minimum numberof calls to preceed")

#parser.add_argument("-m", "--more", type=int, help = "feature to count")

parser.add_argument("-d", "--date", type=str, default = "", help="date yymmdd")

parser.add_argument("-ta", "--tape", type=str, default = "", 
                    help="tape (recording) name, default, no constraint"
                    "all - plots all the tapes separately, <tape>")

# inter call times histogram                    
parser.add_argument("-ict","--plict", dest='ictimes', action='store_true')
parser.add_argument("-no-ict", "--no-plict", dest='ictimes', action='store_false')
parser.set_defaults(ictimes=False) # default don't plot intercall times


#parser.add_argument("-s", "--saveDat", type = int, default = 0, help = "do you whant to save the cepstrum data? (0,1)")

# check input
assert( parser.parse_args().Nshuffl >= 0 )  # number of times we shuffle, Nshuffl = 0 (don't shuffle)


###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
###########################################################################

##### ASSIGMENTS #####
args = parser.parse_args()
inF = args.csvInF
groupNs = args.groupNs  # list
Nshuffl = args.Nshuffl
timeT = args.time_T
feature = args.feature
plTr = args.atLeastN2gr
normalize = args.normalize
minNumCalls = args.minNumCalls
date = args.date
tape = args.tape
ictSwitch = args.ictimes

print( "groups:", groupNs, "\ntime:", timeT, "\n#shuffle:", Nshuffl)
print("date:", date, "\ntape:a", tape)
#saveD = args.saveDat ## save data

##### FILE HANDLING #####
outDN = os.path.abspath(os.path.expanduser(os.path.dirname(inF))) + '-shuffled/'
fileBN = os.path.basename(inF).split('.')[0]  # take the base name & remove the extension
outF0 = os.path.join(outDN, fileBN)
print( "out dir:", outDN, "\nfile BN:", fileBN)

## Check dir or create
if not os.path.isdir(outDN):
    print( "creating out dir:", outDN)
    os.mkdir(outDN)

##### COMPUTATIONS #####

### read data
datB0 = pd.read_csv(inF, dtype = {'date' : str})

def printShuffledDistribution(datB, Nshuffl, baseN, minNumCalls=minNumCalls,\
                              timeT=timeT, feature=feature, normalize=normalize):

    if( len(datB) <= minNumCalls):
        print('Only %d calls (min %d calls)'%(len(datB), minNumCalls))
        return 0
    
    
    ### Non shuffled data
    liS = seqs.df2listOfSeqs(datB, timeT=timeT, feature=feature, shuffl=0)  # define seqs
    NBiG0, c2i, i2c = seqs.listOfSeqs2BigramCounts(liS)  # count bigrams
    df = pd.DataFrame(NBiG0, index=range(1))  # inicialize dataframe w/ the counts
    NBiG_normalized = seqs.normalize_2grams_dict(NBiG0)  # normalize bigrams
    newDf = pd.DataFrame(NBiG_normalized, index=np.arange(1))  # dict -> dataFrame
    df = pd.concat([df, newDf], ignore_index=True)  # concat dataFrames
    
    ### shuffled data calculations
    baseN += '_T%d_NORM%d_N%d'%(timeT, normalize, Nshuffl)
    if normalize: 
        normfun = lambda x : seqs.normalize_2grams_dict(x) 
    else:
        normfun = lambda x : x
        
    for i in range(Nshuffl):  # SHUFFLE, Nshuffl=1 was inicialization => substract 1
        liS = seqs.df2listOfSeqs(datB, timeT=timeT, feature=feature, shuffl=1)  # this group slection just double checks

        X = {j: 0 for j in NBiG0.keys()}  # inicialize adj dictionary

        print(i)
        NBiG, c2i, i2c = seqs.listOfSeqs2BigramCounts(liS, X, c2i, i2c)  # bigrams
        
        NBiG_normalized = normfun(NBiG)  # normalize
            
        newDf = pd.DataFrame(NBiG_normalized, index=np.arange(1))
        df = pd.concat([df, newDf], ignore_index=True)  # save run statistics


    ### print distributions
    outcsv = baseN +'.csv'
    print("out:", outcsv, len(df))
    df.to_csv(outcsv, index=False)

    ### print dictionary
    outdict = baseN + '.dat'   
    print( "out:", outdict, len(df))
    fi = open(outdict, 'w')
    fi.writelines(["%s\n" % item for item in i2c])
    fi.close()


### 


### RUNING CASES

def runIter(df0, constraintType, baseName, constrLi=[]):
    '''
    all getrent
    prints the shuffled disttributions for different constraint combinations.
    If the constrLi is empty, then we iterate over all the posibe constrains of
    the given constrintType. Ex: all the posible tapes
    < constrLi, is a list with the constrints we want to apply
    '''
    if len(constrLi)==0: constrLi = list(set(df0[constraintType]))
        
    for case in constrLi:
        df, baseN = seqs.constrainDF(df0, [constraintType], case, baseName)
        print("TEST:", len(df))
        seqs.printShuffledDistribution(df, Nshuffl, baseN, normalize=normalize,
                              minNumCalls=minNumCalls,
                              timeT=timeT, feature=feature)
        if ictSwitch:
             outF = baseN + '-ictimes.pdf'
             print(outF)
             if len(df)>10:
                 print("#calls", len(df))
                 seqs.plTimeIntervals(df, frac=0.75, outFig=outF)

#doDict = {'constrained': runConst, 'iterate': runIter, 'unconst': run}

for groupN in groupNs:
    outF = outF0 + '_GR%s' % (groupN)
    df = datB0.loc[datB0['group'] == groupN]
    print(groupN, len(df), outF)
    if tape=='all':
        print("all tapes")
        runIter(df, 'recording', outF)
    elif tape:
        print("tape:", tape)
        runIter(df, 'recording', outF, [tape])        
    elif date=='all':
        print("all dates")
        runIter(df, 'date', outF)
    elif date:
        print("dates:", date)
        runIter(df, 'date', outF, [date])
    else:
        print("No constraint")
        printShuffledDistribution(df, Nshuffl, baseN=outF, normalize=normalize,
                              minNumCalls=minNumCalls,
                              timeT=timeT, feature=feature)

                             
                              
sys.exit()                             
                
      