#!/usr/bin/python

import subprocess as sp
import sys
import os

###### check input ######
if (len(sys.argv) != 2):
    print 'usage: %s <directory with the wav files>' % sys.argv[0]
    print "ERROR: bad input"
    sys.exit()


##### modeule import #####
#scriptsDir = os.path.abspath(os.path.dirname(os.path.expanduser(sys.argv[0])))
#sys.path.append(scriptsDir)

##### dir managing #####
filesDir = os.path.abspath(os.path.expanduser(sys.argv[1]))

##### ploting #####
for file in os.listdir(filesDir): 
    if file.endswith(".wav"): # para cada archivo wav en el dir       
        wavF = filesDir +"/"+file
        print wavF
        p=sp.Popen([os.path.abspath(os.path.expanduser("~/whales/scripts/signal-processing/plWavSpectro.py")), wavF, '--plTitle'])
        p.wait()

        '''
        for N in [9]:
            print "N", N, wavF
            sp.Popen(["mySpectro.py", wavF, '-w', str(N), '-o', str(0.5), '-s', str(0)])

'''
print "FIN"

