# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:13:27 2015

@author: florencia
"""

from __future__ import print_function, division  # py3 compatibility
#import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
#import sys
import featureExtraction as fex
import pylotwhale.signalProcessing.signalTools_beta as sT
import pylotwhale.utils.annotationTools as annT
import pylotwhale.MLwhales.MLtools_beta as myML
import pylotwhale.signalProcessing.audioFeatures as auf
import pylotwhale.utils.annotationTools as annT


### split

### PREDICT WAV SECTIONS (SOUND TYPE)

def predictSoundSections(wavF, clf, lt, feExFun,
                         outF='default', annSections='default'):
    '''
    predicts and generates the annotations of the given wavF
    Parameters:
    -----------
    wavF : str
        path to wav file
    clf : estimator
        classifier object
    lt : label transformer object
    feExFun : callable
        feature extraction
    out : str
        annotations out file name, default = wav base name + '-predictions'
    annSections : array
        sections to print, default = ['c']
    '''
    if outF == 'default':
        bN = os.path.basename(wavF)
        outF = os.path.join(outDir, bN.replace('.wav', '-predictions.txt'))

    waveForm, fs = sT.wav2waveform(wavF)
    return predictSectionsFromWaveform_genAnnotations(waveForm, fs, clf, lt, feExFun,
                                       outF=outF, annSections=annSections)


def predictSectionsFromWaveform_genAnnotations(waveform, fs, clf, lt, feExFun, outF,
                                annSections='default'):

    """
    predicts the sections of a waveform and generates annotations
    walking along a waveform 
    Parameters:
    -----------
    waveform : ndarray
        sound waveform
    clf : estimator
        classifier object
    lt : label transformer object
    feExFun : callable
        feature extraction
    out : str
        annotations out file name, default = wav base name + '-predictions'
    annSections : array
        sections to print, default = ['c']
    """

    if annSections == 'default':
        annSections = ['c']

    tf = 1.0*len(waveform)/fs    
    M0 =  feExFun(waveform)#, annotations=annotLi_t)
    y_pred = clf.predict(M0)
    annT.predictions2txt(lt.num2nom(y_pred), outF, tf, sections=annSections)
    return outF


def WSD2predictions(wavF, annWSD1, feExtFun, lt, WSD2_clf, outF,
                    readSections='default', keepSections='default', dt=0):
    """Generate annotations using the WSD2
    reads the predicted sections from WSD1 to predicts
    the finer structure of the calls
    with clf trained with a smaller nTextWS
    Parameters
    ----------
    wavF: str
        wavefile name
    feExFun: callable
        feature extraction function
    lt: LabelEncoder
        label transformation object
    WSD2_clf: estimator
        model for estimating predictions
    outF: str
        name of the output annotations
    readSections: list like object
        array with the ann sections from WSD1 we want to reinterpret, default = ['c']
    keepSections: list like object
        array with the ann sections we want to print
    dt: float
        time buffer for reading arund the desired annotation section
    """
    try:
        os.remove(outF)
    except OSError:
        pass
    waveform, fs = sT.wav2waveform(wavF)  # load waveform
    A = annT.anns2array(annWSD1)  # load annotations
    for t0i, t0f, l0 in A[:]:  # for each ann section
        if l0 in readSections:  # if section of interest (c)
            thisWaveform = auf.getWavSec(waveform, fs, t0i - dt, t0f + dt)
            ## predict annotations
            T, L = predictAnnotations(thisWaveform, fs, feExtFun, lt,
                                      WSD2_clf,
                                      annSections=keepSections)
            newT = T + t0i - dt  # relative to the orginal ann sections

            outF = annT.save_TLannotations(newT, L, outF, opening_mode='a')
    return outF
    
    
def predictAnnotations(waveform, fs, feExFun, lt, clf, annSections='default'):

    """
    predicts annotation sections of a waveform
    walking along a waveform 
    Parameters:
    -----------
    waveform : ndarray
        sound waveform
    clf : estimator
        classifier object
    lt : label transformer object
    feExFun : callable
        feature extraction
    annSections : array
        sections to print, default = ['c']
    Returns
    -------
    T: ndarray (#annotations, 2)
        initial and final time of the annotation sections
    labels: ndarray (#anns, )
        labels of the annotation sections
    """

    if annSections == 'default':
        annSections = ['c']

    tf = 1.0*len(waveform)/fs    
    M0 = feExFun(waveform)  #, annotations=annotLi_t)
    y_pred = clf.predict(M0)
    T, labels = annT.predictions2annotations(lt.num2nom(y_pred), tf)
    return T, labels     


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
        
  