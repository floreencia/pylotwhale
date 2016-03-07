#!/usr/bin/python
#!/usr/mprg/bin/python

import numpy as np
import pylab as pl
import argparse
import os
import sys
import scipy.signal as sig
from scipy.io import wavfile

#import scikits.audiolab as al

#sys.path.append('/home/florencia/whales/scripts/signal-processing/')
import signalTools_beta as preP

parser = argparse.ArgumentParser(description = 'looks into all the wav files of a given directory and generates a csv file with information form the wav.') #takes the input and interprets it

parser.add_argument("inDir", type=str, help = "directory with wav files")
parser.add_argument("-o", "--outDir", default = '', type=str, help = "output directory.")

parser.add_argument("-N", "--Nbits", type=int, default = 4, help = "number of bits we whant to use in the binarization")


###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
###########################################################################

##### ASSIGMENTS #####
args = parser.parse_args()
inDir = os.path.expanduser(os.path.abspath(args.inDir))
outDir = args.outDir
nBits = args.Nbits

power = 9
N = 2**power
over = int(N*0.9)
winN = 'hanning'
win =  sig.get_window(winN, N)#np.hamming(N)
v0_c = 1000
vf_c = 20000

if not outDir: outDir = inDir.rstrip('\\')+'-binMatrices'

## Check dir or create it
if not os.path.isdir( outDir ):
    print "creating out dir:", outDir
    os.mkdir( outDir )

for wavF in os.listdir(inDir):
    if wavF.endswith(".wav"):
        print wavF
        #waveform, sampRate, x = al.wavread(os.path.join(inDir, wavF))
	sampRate, waveform = wavfile.read(os.path.join(inDir, wavF))
        WF_len = len(waveform)

        S = pl.specgram(waveform, Fs = sampRate, NFFT = N, noverlap = over, window = win)
        A = np.log(S[0])
        A_cuted = preP.selectBand(preP.reeScale_E(A, spec_factor = 0.7), v0_cut=v0_c, vf_cut=vf_c)
        A_cuted = preP.reeSize_t( preP.allPositive_andNormal( A_cuted ))
        A_bin = preP.myBinarize(A_cuted)
        outFN =  wavF.split('.wav')[0] + '_%s.dat'%nBits
        print outFN
        np.savetxt(os.path.join(outDir, outFN), A_bin)






