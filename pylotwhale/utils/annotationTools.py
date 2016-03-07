#!/usr/bin/python

from __future__ import print_function
#import sys
import numpy as np
import re
import os
import scikits.audiolab as au

"""
tools for the preparation of annotated files
"""
    
#### AUP annotations <---> mtl (marsyas) annotations
    
def parseAupFile(inFilename, sep='.'):
    """ 
    parses an audacity text file
    Parameters:
    -----------
    inFilename : file with audacity-like annotations (t0 \t tf \t label)
    sep : decimal separator, by default "." but often i.e. the case of my
        aup the decimal separator is a ","
    Returns:
    --------
    data : list of dictionaries with the aup annotations format
            { startTime  endTime  label } 

    """
    #print(inFilename)
    with open(inFilename, "r") as f:
        lines = f.read().splitlines() 
    
    data = []
    
    for line in lines:
        m = re.search('([-0-9%s]*)\t([-0-9%s]*)\t(\w*)'%(sep, sep), line)
        assert m, "lines don't match the RE"
        startTime = float(m.group(1).replace(sep, '.')) # replaces separator for dot
        endTime = float(m.group(2).replace(sep, '.'))
        label = m.group(3)
        item = { 'startTime' : startTime, 'endTime' : endTime, 'label' : label }
        data.append(item)
        
    return data      


def readCols(fName, colIndexes, sep='\t'):
    '''
    Read the columns "colIxes" of a file "fName"
    
    np.loadtxt(file, delimiter='\t', dtype='|S') can be used instead!
    Parameters
    ----------
    fName : file to read
    colIndexes : list with the indexs of the columns to read
    sep : column separator
    Returns
    -------
    collectionList : list of tuples with the wav - annotation files
                        tu[0] : path to wav file
                        tu[1] : path to annotation file
    '''
    with open(fName) as f:
        lines = f.readlines()
    
    #li = [tuple([line.strip().split(sep)[ci] for ci in colIndexes]) for line in lines if not line.startswith('#')]
    li = [[line.strip().split(sep)[ci] for ci in colIndexes] for line in lines if not line.startswith('#')]
    return li  
    

### text file editing
     
def replaceInFile(textFile, string, replaceString, newTextFile=None):
    '''
    replaces all occurrences of a string in a file
    textFile : file to edit
    string : string to replace 
    replaceString : replacing string
    '''
    if newTextFile == None : newTextFile = textFile
    try:
        with open(textFile, 'r') as f: ## read sv-lines
            s = f.read().replace( string, replaceString)
    except IOError:
        print("WARNING! cannot open: \n\t {}".format(textFile))
        return None        
        
    #s = open(textFile).read().replace( string, replaceString)
    with open( newTextFile, 'w') as f:
        f.write(s)    
    return newTextFile    
    
def commas2periods(textFile, newTextFile=None):
    '''
    replaces all commas for dots
    '''    
    return replaceInFile(textFile, ',', '.', newTextFile=newTextFile)    
    
def svAnn2t0_tf_label(svAnnFile, newAnnFi=None):
    """converts sonic visualizer annotations to aup annotations
        sv : [t0, freq, Delta t, label]
        aup: [t0, tf, label]
    """
    if newAnnFi == None : newAnnFi = svAnnFile.replace('.txt', '-aupAnn.txt')
    try: ## remove annotations file if already exists
        os.remove(newAnnFi)
    except OSError:
        pass
    
    try:
        with open(svAnnFile, 'r') as f: ## read sv-lines
            lines = f.read().splitlines()
    except IOError:
        print("WARNING! cannot open: \n\t {}".format(svAnnFile))
        return None
            
    for li in lines:
        try: # check format
            t0, _, dt, l = li.strip().split() 
            with open(newAnnFi, 'a') as g:
                g.write("{}\t{}\t{}\n".format(t0, float(t0)+float(dt), l))
        except:
            continue
    return newAnnFi
    
    
def filterAnotations_minTime(annFi, outFi=None, minTime=0.01):
    '''
    Filters out annotations shorter that minTime
    '''
    
    if outFi is None: outFi = annFi.replace('.txt', '-timeFiltered.txt')
       
    with open(annFi) as f:
        lines = f.readlines()
        
    try:
        os.remove(outFi)
    except OSError:
        pass
        
    for li in lines:
        t0, tf, label = li.split('\t')
        if float(tf)-float(t0) > minTime:
            with open(outFi, 'a') as g:
                g.write(li)
                
    return(outFi)    

                    
#### back to time stamps -- audio frame analyser (walking)

def getSections(y, tf=None):
    '''
    takes labels (numeric) vector and returns a dictionary of the sections
    if tf is given the sections will have time units otherwise they'll have
    sample units
    Parameters
    ----------
    < y : labels (np.array)
    < tf : time at y[-1]
    Returns
    -----------
    > sections dictionary in analiysed fft-samples
        (s0, sf) : label
        
    ASSUMPTIONS:
        - no time overlappong regions in the annotations
        
    '''
    scaleFac = 1.*tf/len(y) if tf else 1
    
    sectionsDict = {}
    s0 = 0
    
    for tr in np.where([y[i] != y[i-1] for i in range(len(y)) ] )[0]:
        sectionsDict[(s0*scaleFac, tr*scaleFac)] = y[s0+1]
        s0 = tr

    # write the last interval if there is still space for one 
    #print("interval:", s0, len(y)) 
    if s0+1 <= len(y)-1: 
        sectionsDict[(s0*scaleFac, len(y)*scaleFac)] = y[s0+1]
    return(sectionsDict)


def predictions2txt(y, outTxt, tf, sections='default'):
    '''
    this functions transformed the predicted labels y into a txt annotation file
    that can be read by audacity.
    < y : predicted samples
    < outTxt : output file name
    < sR : sampling rate of the wave file corresponding to the annotation
    < ws : size of the fft-analysis window
    < sections : an array with the sections (num) we are interested
        'default', means section '1'
    > txtFile : audacity annotations file
    '''
    
    if sections == 'default': sections = [1] # only call, not background
        
    sectionsD = getSections(y, tf)  ## find sections
    ky = sorted(sectionsD.keys(), key = lambda x: x[0]) ## sort the stamps
    
    with open(outTxt, 'w') as f: ## write
        for (t0, tf) in ky:
            #print("keys:", t0, tf, sectionsD[(t0, tf)], sections)
            if sectionsD[(t0, tf)] in sections:
                f.write("%f\t%f\t%s\n"%(t0, tf, sectionsD[(t0, tf)] ) )   
