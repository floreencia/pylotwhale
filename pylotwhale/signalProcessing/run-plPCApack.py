#!/usr/bin/python

import sys
import os.path
import plPCApack as plPCA
import scikits.audiolab as al

'''
this script calls tha pakage plSpecgram for ploting spectrogtrams from wav files
we are wvaring the window feng of the fft
'''

###### check input ######
if (len(sys.argv) != 2):
    print 'usage: %s <directory with the ceps files>' % sys.argv[0]
    print "ERROR: bad input"
    sys.exit()

###### sampling rate ######
#!!!!

##### modeule import #####
#scriptsDir = os.path.abspath(os.path.dirname(os.path.expanduser(sys.argv[0])))
#sys.path.append(scriptsDir)
#import plSpecgrams as pspec

##### dir managing #####
filesDir = os.path.abspath(os.path.expanduser(sys.argv[1]))

##### ploting #####
for file in os.listdir(filesDir): 

    if file.endswith("-cepst.dat"): # para cada archivo wav en el dir
        print file
        cepsF = filesDir +"/"+file
        #for N in (8,10,11,12,13):
         #   pspec.saveSpecgramWav(wavF,N)

	plPCA.pcaCalcs(cepsF, plBasis = 1, sRate=48000)

