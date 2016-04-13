# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:13:27 2015

@author: florencia
"""

from __future__ import print_function, division # py3 compatibility
#import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
#import sys
import featureExtraction as fex
import pylotwhale.signalProcessing.signalTools_beta as sT
import pylotwhale.utils.annotationTools as annT
import pylotwhale.MLwhales.MLtools_beta as myML


### split

### PREDICT WAV SECTIONS (SOUND TYPE)

def predictSoundSections(wavF, clf, lt, feExFun, 
                         outF='default', annSections='default'):
    '''
    predicts and generates the annotations of the given wavF
    PARAMETERS:
    -----------
        wavF : path to wav file
        clf : classifier object
        lt : label transformer object
        feExFun : feature extraction callable
        out : annotations out file name, default = wav base name + '-predicitons'
        annSections : setions to print, defaulf = 'c'
    '''
    if outF =='default':
        bN = os.path.basename(wavF)
        outF = os.path.join(outDir , bN.replace('.wav', '-predictions.txt'))
    if annSections == 'default': 
            annSections = ['c']

    waveForm, fs = sT.wav2waveform(wavF)
    tf = 1.0*len(waveForm)/fs
        
    M0, _, featN, fExStr =  feExFun(waveForm, fs)#, annotations=annotLi_t)
    y_pred = clf.predict(M0)
    annT.predictions2txt(lt.num2nom(y_pred), outF, tf, sections=annSections)
    return outF

### PREDICT THE LABELS OF THE ANNOTATED SECTION IN A WAV FILE  (CALL TYPE)

def predictFeatureCollectionAndWrite(inFile, clf, lt, col=0, outFile=None, sep='\t', stop=None):
    '''read a collection (indexfile) of features (*.npy) --> predict --> save predictions'''
    if outFile is None: outFile = os.path.splitext(inFile)[0] + '-predictions.txt'       
    try: # remove file if exists
        os.remove(outFile)
    except OSError:
        pass
    
    with open(inFile) as f: # read lines
        lines=f.readlines()

    with open(outFile, 'a') as g: # predict
        g.write("#{}\n".format(lt.classes_))
        for li in lines[:stop]:
            if li[0] != '#':
                li = li.strip()
                X = fex.loadAndFlattenX(li.split(sep)[col])
                y = clf.predict(X)
                y_probs = clf.predict_proba(X)
                li += '\t{}\t{}\n'.format(lt.num2nom(y)[0], y_probs[0])
            g.write(li)
    return outFile
    
def predictAnnotationSections(wavF, annF, clf, feExtParams, lt, outFile=None,
                              sep='\t', printProbs=False, header=''):
    '''
    Predicts the label (call types) of each annotated section and writes 
    the prediction into a outFile
    Parameters:
    -----------
    < wavF : wave file
    < annF : anontatiosn file
    < clf : classifier
    < featExtFun :  feature extraction function (callable)
                    or a dictionary with the feature extraction settings
                    featureExtrationParams = dict(zip(i, i))
    '''
    ### out file handling
    if outFile is None: outFile = os.path.splitext(annF)[0] + '-sectionPredictions.txt'       
    try: # remove file if exists
        os.remove(outFile)
    except OSError:
        pass
    
    ## read data
    predO = fex.wavAnn2sectionsXy(wavF, annF, featExtFun=feExtParams)
    ## predict
    predictions = np.expand_dims(lt.num2nom(clf.predict(predO.X)), axis=1)
    if printProbs:
        predictions = np.hstack((predictions, clf.predict_proba(predO.X)))
        header = '{}'.format(le.classes_)
    ## save file
    A = np.loadtxt(annF, delimiter='\t', dtype='|S', ndmin=2)#,usecols=[0,1])
    print("TEST", np.shape(A), np.shape(predictions))
    np.savetxt(outFile, np.hstack((A, predictions)), fmt='%s', 
               delimiter = '\t', header=header)
    return outFile
    
    
    
    
### GARBAGE    
    
def plConfusionMatrix(cM, labels, outFig='', figsize=None):
    '''
    MOOVED TO myML
    '''
    print("~WARNING!!! USE THE ONE IN myML")
    myML.plConfusionMatrix(cM, labels, outFig='', figsize=None)
    
    
def predictAndWrite(inFile, clf, lt, col=0, outFile=None, sep='\t', stop=None):
    ''' don't use this function
        use "predictFeatureCollectionAndWrite" instead. THIS FUNCITON WILL BE REMOVED'''
    return predictFeatureCollectionAndWrite(inFile, clf, lt, col, outFile, sep=sep, stop=stop)
        
  