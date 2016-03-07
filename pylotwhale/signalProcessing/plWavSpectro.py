#!/usr/bin/python

import numpy as np
#import pylab as pl
import matplotlib
import scipy.signal as sig
from scipy.io import wavfile
import argparse
import os
import sys
#import plSpecgrams as pspec # for the plots

#sys.path.append('./signal-processing/')
import signalTools_beta as sT

parser = argparse.ArgumentParser() #takes the input and interprets it
parser.add_argument("wavFN", type=str, help = "wav file name")
parser.add_argument("-o", "--overlap", type = float, default = 0.5, help = "overlap of the NFTT")
parser.add_argument("-w", "--winPow", type = int, default = 9, help = "Exponent of the FFT window size, NFFT = 2^{windowPow}")

parser.add_argument("-sd", "--saveDat", action='store_true', help = "include this option if you want to save the data")
parser.set_defaults(saveDat=False)                    

parser.add_argument("-f0", "--frec0", type = int, default = 1000, help = "frequency fileter. Lower frequency.")

parser.add_argument("-ff", "--frecf", type = int, default = 20*1000, help = "Frequency fileter. Upper frequency")

parser.add_argument("-e", "--figExt", type = str, default = 'png', help = "Extention of the output figure")

parser.add_argument("-plT", "--plTitle", action='store_true', help = "writes the the wav file name in the plot title")
parser.set_defaults(plTitle=False)                    

parser.add_argument("-s", "--plFontSize", type=int, default=18, 
                    help = "Default font size for the plots")

# check input
assert( parser.parse_args().overlap >= 0 and parser.parse_args().overlap < 1 ) #overlap \in [0,1)
assert( parser.parse_args().winPow < 15 ) #overlap \in [0,1) 

###########################################################################
###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
###########################################################################


##### ASSIGMENTS #####
args = parser.parse_args()
fileN = args.wavFN ## file name
NFFTpow = args.winPow ## power
overlap = args.overlap ## overlap
saveD = args.saveDat ## save data
figExt = args.figExt ## save data
v0_c = args.frec0
vf_c = args.frecf
plFontSize = args.plFontSize

matplotlib.rcParams.update({'font.size': plFontSize})

### FFT settings
N = 2**NFFTpow
winN = 'hanning'

##### FILE HANDLING #####
outDN = os.path.dirname(fileN)+'-specs'
fileBN = os.path.basename(fileN).split('.')[0] # take the base name and remove the extension

## plot title
plTitle = fileBN if args.plTitle else ''

fileBN = fileBN + '-spec_wsPwp%d_over%d_cut%d-%d'%(NFFTpow, int(100*overlap), v0_c, vf_c)
outF = os.path.join(outDN, fileBN+ '.' + figExt)
print "base name:" , outF

## Check dir or create
if not os.path.isdir(outDN):
    print "creating out dir:", outDN 
    os.mkdir(outDN)



##### COMPUTATIONS #####
## get waveform
sRate, waveForm = wavfile.read(fileN)

## compute spectro
sT.plspectro(waveForm, outF=outF, sRate=sRate, N = N, v0_cut = v0_c, vf_cut = vf_c,\
             overFrac = overlap, plTitle=plTitle, plTitleFontSz=plFontSize-2)

sys.exit()

## plot the cesptrum
#plNumCeps(myCeps, tf, Nceps, Bname) 

## save cepstrum data
if( saveD ):
    outFN = Bname+'.dat'
    print "saving ceps data", outFN, np.shape(myCeps)
    plt.savetxt( outFN, abs(myCeps) )
