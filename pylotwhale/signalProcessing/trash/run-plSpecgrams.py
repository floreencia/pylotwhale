#!/usr/bin/python

import sys
import os.path

'''
this script calls tha pakage plSpecgram for ploting spectrogtrams from wav files
we are wavaring the window feng of the fft
'''

###### check input ######
if (len(sys.argv) != 2):
    print 'usage: %s <directory with the wav files>' % sys.argv[0]
    print "ERROR: bad input"
    sys.exit()

##### modeule import #####
scriptsDir = os.path.abspath(os.path.dirname(os.path.expanduser(sys.argv[0])))
sys.path.append(scriptsDir)
import plSpecgrams as pspec

##### dir managing #####
filesDir = os.path.abspath(os.path.expanduser(sys.argv[1]))

##### ploting- #####
N=10
for myfile in os.listdir(filesDir): 
    if myfile.endswith(".wav"): # para cada archivo wav en el dir
        print myfile
        wavF = filesDir +"/" + myfile
        pspec.saveSpecgram(wavF,N)
        

'''
##### ploting-for severan window sizes #####
for file in os.listdir(filesDir): 
    if file.endswith(".wav"): # para cada archivo wav en el dir
        print file
        wavF =filesDir +"/"+file
        for N in (11, 12):
            pspec.saveSpecgram(wavF,N)

'''