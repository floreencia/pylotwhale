#!/usr/bin/python

from __future__ import print_function
#import sys
import numpy as np
import re
import os
#import scikits.audiolab as au
#import warnings


"""
tools for the preparation of annotated files
"""
    
#### AUP annotations <---> mtl (marsyas) annotations
    
def anns2array(annF):
    '''loads annotations file into ndarray'''
    return np.genfromtxt(annF, dtype=None)
    
def loadAnnLabels(fi, cols=(2,)):
    """Loads labels from annotations file (3rd column)"""
    return np.loadtxt(fi, dtype=str, usecols=cols, ndmin=1)
    
    
def anns2TLndarrays(fi):
    """like anns2array but returns 2 ndarrays T (n, 2) and L (n,)"""
    T = np.loadtxt(fi, usecols=(0,1), ndmin=2)
    L = loadAnnLabels(fi)
    return T, L
    
def save_TLannotations(T, L, outF, opening_mode='w'):
    """saves T, L as an annotations file"""
    
    assert len(T) == len(L), "T and L must match in length"

    if opening_mode == 'w':
        try:
            os.remove(outF)
        except OSError:
            pass
        
    with open(outF, opening_mode) as f:  # new annotations
        for i in np.arange(len(L)):  # print new annotations
            f.write("{:5.5f}\t{:5.5f}\t{:}\n".format(T[i, 0],
                    T[i, 1], L[i]))
    return outF
    
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
        try:
            m = re.search('([-0-9%s]*)\t([-0-9%s]*)\t(\w*)'%(sep, sep), line)
            assert m, "{}\nlines don't match the RE".format(line)
            pass
        except:
            print("{}\nnot parsed: {}".format(inFilename, line))
            continue
            
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


def getLabels_from_wavAnnColl(collection, annCollLabel=1, labelCol=2):
    '''
    parses the annotations of a collection
    Parameters:
    -----------
        collection : 
        anncollLabel : number of the column with the annotations
        labelCol : column of the label in the annotation file
    '''
    labels = []
    for li in collection:
        annF=li[annCollLabel]
        l = readCols(annF, [labelCol])
        labels.extend(reduce(lambda x,y: x+y,l))
    
    return labels  
    

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
    """
    takes labels (numeric) vector and returns a dictionary of the sections
    by joining labels of the same type into sections.
    If tf is given the sections will have time units otherwise they'll have
    sample units
    Parameters
    ----------
    y: labels (np.array)
    tf: time at y[-1]
    Returns
    -------
    > sections dictionary in analysed fft-samples
        (s0, sf) : label
        
    ASSUMPTIONS:
        - no time overlapping regions in the annotations
        
    """
    scaleFac = 1.*tf/len(y) if tf else 1
    
    sectionsDict = {}
    s0 = 0
    
    for tr in np.where([y[i] != y[i-1] for i in range(1,len(y)) ] )[0]:
        sectionsDict[(s0*scaleFac, (tr+1)*scaleFac)] = y[tr]
        s0 = tr+1

    # write the last interval if there is still space for one 
    #print("interval:", s0, len(y)) 
    if s0+1 <= len(y)-1: 
        sectionsDict[(s0*scaleFac, len(y)*scaleFac)] = y[s0+1]
    return(sectionsDict)
    
def annDi2annArrays(sectionsD):
    """reformats sectionsD into T array with the times and L
    array with the labels"""
    times = sorted(sectionsD.keys(), key = lambda x: x[0]) # sort annotations
    T = np.zeros((len(times), 2))
    labels = np.empty(len(times), dtype=str)
    for i, t in enumerate(times):
        T[i, :] = t
        labels[i] = sectionsD[t]
    return T, labels
        
def predictions2annotations(y, tf=None):
    """interprets predictions as annotations: 
    Parameters
    ----------
    y: ndarray
        predictions
    tf: final time of the wavform to wich this predictions belong to
    Returns
    -------
    T: ndarray (n, 2)
        times
    L: ndarray (n,)
        labels
    """
    sectionsD = getSections(y, tf)
    return annDi2annArrays(sectionsD)


def predictions2txt(y, outTxt, tf, sections):
    '''
    transforms the predicted labels y into a txt annotation file
    that can be read by audacity or other audio software.
   
    Parameters
    ----------
    y: predicted samples
    outTxt: output file name
    tf: float
        final time
    sections: list 
        sections (num) we are interested
    Returns
    -------
    txtFile: str
        name of the output file with the annotations
    '''
        
    sectionsD = getSections(y, tf)  ## find sections
    ky = sorted(sectionsD.keys(), key = lambda x: x[0]) ## sort the stamps
    
    with open(outTxt, 'w') as f: ## write
        for (t0, tf) in ky:
            #print("keys:", t0, tf, sectionsD[(t0, tf)], sections)
            if sectionsD[(t0, tf)] in sections:
                f.write("%f\t%f\t%s\n"%(t0, tf, sectionsD[(t0, tf)] ) )   
                
           
