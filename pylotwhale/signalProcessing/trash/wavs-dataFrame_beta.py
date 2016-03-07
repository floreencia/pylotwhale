#!/usr/bin/python

import numpy as np
import pylab as pl
import argparse
import os

import scikits.audiolab as al


"""
this script computes the bigram probability matrix and the shuffled version and computes the diference among this tow
"""

parser = argparse.ArgumentParser(description = 'looks into all the wav files of a given directory and generates a csv file with information form the wav.') #takes the input and interprets it

parser.add_argument("inDir", type=str, help = "directory with wav files")

###########################################################################
######################  NOW WE PASS THE ARGUMENTS  ########################
###########################################################################

##### ASSIGMENTS #####
args = parser.parse_args()
inDir = os.path.expanduser(os.path.abspath(args.inDir))

outF = inDir + 'wavs.csv'
#outF = '~/whales/final_test.dat'
oF = open(outF, 'w')
oF.write( "wav_file,samp_rate,time_duration,length\n")

for wavF in os.listdir(inDir):
    if wavF.endswith(".wav"):
        print wavF
        waveform, sampRate, x = al.wavread(os.path.join(inDir, wavF))
        WF_len = len(waveform)
        tf = 1.0*WF_len/sampRate
        oF.write( "%s, %d, %s, %d\n"%(wavF, sampRate, tf, WF_len))

oF.close()
print outF





