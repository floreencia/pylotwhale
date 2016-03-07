# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:41:04 2015
@author: florencia
"""
from __future__ import print_function
import os
import glob
#import warnings

def areListItemsUnique(mylist):
    '''checks for duplcated items in a list'''
    uniqueLi = list(set(mylist))
    if len(mylist) == len(uniqueLi):
        return True
    else:
        return False    

def uniqueItemIndexes(mylist):
    '''returns the indexes of the set of elements in the list (mulist), 
    i.e. excluding duplicates'''
    uniqueLi = list(set(mylist))
    if len(mylist) != len(uniqueLi):
        print("WARNING! Duplicated items")
        #warnings.warn("Duplicated items")
        return [mylist.index(item) for item in uniqueLi]
    else:
        print("unique items list")
        return(range(len(mylist)))
        
    
def filterListForUniqueItems(myList):
    uIdx = uniqueItemIndexes(myList)
    return [myList[item] for item in uIdx]
   
    
### collection creating    
    
def annotationsDir2wavAnnCollection(annDir, wavDir='default', outCollFile='default',
                                     str0='.txt', strRep='.wav'):
    '''
    reads the annotation files (*.txt) in annDir, 
    searches their corresponding waves in
    wavDir and saves a collection <outCollFile> of annotated wavs  
    '''    
    if wavDir == 'default': wavDir = os.path.join( annDir, '..')
    if outCollFile == 'default': outCollFile = os.path.join( annDir, '..', 'collection.txt')
    annotationsFiList = glob.glob( os.path.join(annDir, '*.txt') )  # enlist all the annotations
    return annotationsList2wavAnnCollection(annotationsFiList, wavDir, outCollFile,
                                     str0=str0, strRep=strRep )
    
def annotationsList2wavAnnCollection( annotationsFiList, wavDir, outCollFile,
                                     str0='.txt', strRep='.wav' ):
    '''
    searches the wave files in the dir <wavDir> 
    from each annotationFile in annotationsFiList
    and saves a collection <outCollFile> of annotated wavs
    Parameters
    ----------
    annotationsFiList : list of annotation files
    wavDir : directory where to find the waves
    outCollFile : out file name
    str0 : string of the template file to be replaplazed
    strRep : characteristic string of the file we are looking for
    '''
    with open(outCollFile, 'w') as g:
        for annFi in annotationsFiList:
            wavFileName = os.path.basename(annFi).replace( str0, strRep)
            #print(wavFileName)
            wavFiPath = findFilePathInDir(wavFileName, wavDir)
            if wavFiPath:
                g.write("{}\t{}\n".format( wavFiPath, annFi) )     
    return outCollFile
    

def findFilePathInDir(guessFile, searchDir):   
    '''
    looks for the path of a file, assuming this is in searchDir
    '''
    for root, dirs, files in os.walk(searchDir):
        for name in files:
            if name == guessFile:
                return os.path.abspath(os.path.join(root, name))
    return False
    
