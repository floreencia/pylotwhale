# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:55:12 2015

@author: florencia
"""

from __future__ import print_function, division # py3 compatibility
import numpy as np
#from pandas import read_csv
import sys
import os
import functools

import pylotwhale.signalProcessing.signalTools_beta as sT
import pylotwhale.signalProcessing.effects as eff

import MLtools_beta as myML

import pylotwhale.utils.whaleFileProcessing as fp


def wavAnnCollection2featureFiles(collectionList, outDir, indexFile='default',
                    featExtFun=None):
    """
    Computes and saves the features of a collection of annotated wavs.
    The features are saved in the outDir and alongside an index file is created with
    the paths to the festures and the labels.
    
    ( •_•)O*¯`·. Used for call type classification .·´¯`°Q(•_• )
    
    Parameters
    ----------
    < collectionList : list of tuples with the wav - annotation files
                        tu[0] : path to wav file
                        tu[1] : path to annotation file
    < outdir : dir where the features will be saved
    < featExtFun : the feature extraction function
                    OR a dictionary with the feature extraction settings
                    featureExtrationParams = dict(zip(i, i))
    < indexFile :  file ith the paths to the featues and the labels

    Return
    ------    
    > indexFile :  a file with the paths to the features and their labels
    """
    ### out dir    
    if not os.path.isdir(outDir): 
        #os.remove( outDir ) # delet if already exists
        os.mkdir( outDir )
    ## index file 
    if indexFile == 'default' :
        indexFile = os.path.join(outDir, 'indexFile.txt')    
    if os.path.isfile(indexFile): 
        os.remove( indexFile ) # delet if already exists

    ### feature extraction settings
    if not callable(featExtFun): # dictionary or None (defaul parameters)
        fexO=wavFeatureExtractionSplit(featExtFun)
        featExtFun = fexO.featExtrFun() # default
        featStr = fexO.feature_str
    else:
        featStr='featureExtractionCallable'
        
    ### extract and save features for each annotated section
    for wavF, annF in collectionList: #  wav file loop
        bN = os.path.splitext(os.path.basename(wavF))[0]
        datO = wavAnn2sectionsXy(wavF, annF, featExtFun=None) # data object
        for i in range(datO.m_instances): # ann-section loop --> save features
            bN += '.%d_%s'%(i, datO.y_labels[i])
            outFile = os.path.join(outDir, bN)
            np.save(outFile, datO.X[i])
            # write in index file
            with open(indexFile, 'a') as g:
                g.write("%s\t%s\n"%(outFile+'.npy', datO.y_labels[i]))
                    
    ### write the feature-extraction-especifications at the bigining of the index file
    with open(indexFile, "r+") as f:
        old = f.read() # read everything in the file
        f.seek(0) # rewind
        f.write("#%s\n%s"%(featStr, old))
    
    return indexFile
    
### call type classification    
    
def wavAnn2sectionsXy(wavF, annF, featExtFun=None):
    """
    Computes the features of each annotated section in the wav file
    ment to be used with feature extraction 'split' 

    ( •_•)O*¯`·. Used for call type classification .·´¯`°Q(•_• )
    
    Parameters
    ----------
    < wavFi : path to wave file
    < featExtFun :  feature extraction function (callable)
                    or a dictionary with the feature extraction settings
                    featureExtrationParams = dict(zip(i, i))
    < indexFile :  file ith the paths to the featues and the labels

    Return
    ------    
    > X : features matrix
    > y : labels
    """

    ### check feature extraction function
    if not callable(featExtFun): # dictionary or None (defaul parameters)
        featExtFun = wavFeatureExtractionSplit(featExtFun).featExtrFun() # default
    ### check existance of provided files    
    assert os.path.isfile(wavF), "%s\ndoesn't exists"%wavF
    assert os.path.isfile(annF), "%s\ndoesn't exists"%annF
    ### extract features for each annotated section
    segmentsLi, fs = sT.getAnnWavSec(wavF, annF)    

    datO = myML.dataXy_names()     
    ## for each annotation in the wavfile compute the features
    for annIndex in range(len(segmentsLi)): 
        label = segmentsLi[annIndex]['label']
        waveform = segmentsLi[annIndex]['waveform']
        M, _, _, featStr = featExtFun(waveform, fs)
        datO.addInstances(np.expand_dims(M.flatten(), axis=0), [np.array(label)])                                           
    
    return datO
    
def wavAnnCollection2sectionsXy(wavAnnColl, featExtFun=None):
    """
    Computes the X, y for a collection of annotated wav files
    for each annotated section in the wav file
    ment to be used with feature extraction 'split' 

    ( •_•)O*¯`·. Used for call type classification .·´¯`°Q(•_• )
    
    Parameters
    ----------
    < wavAnnColl : collection of annotated wavfiles
    < featExtFun :  feature extraction function (callable)
                    or a dictionary with the feature extraction settings
                    featureExtrationParams = dict(zip(i, i))
    Return
    ------    
    > datXy_names : features object
    """
    ### check feature extraction function
    if not callable(featExtFun): # dictionary or None (defaul parameters)
        featExtFun = wavFeatureExtractionSplit(featExtFun).featExtrFun() # default

    datO_test = myML.dataXy_names() 

    for wF, annF in wavAnnColl[:]:
        datO_test_new = wavAnn2sectionsXy( wF, annF, featExtFun=featExtFun) #wavPreprocesingT = wavPreprocessingFun )
        datO_test.addInstances(datO_test_new.X, datO_test_new.y_names )
    
    return datO_test
    
def wavAnn2sectionsXy_ensemble(wavF, annF, featExtFun=None, wavPreprocesingT=None,
                               ensembleSettings=None):
    """
    Computes the features of each annotated section in the wav file
    ment to be used with feature extraction 'split' 

    ( •_•)O*¯`·. Used for call type classification .·´¯`°Q(•_• )
    
    Parameters
    ----------
    < wavFi : path to wave file
    < featExtFun :  feature extraction function (callable)
                    or a dictionary with the feature extraction settings
                    featureExtrationParams = dict(zip(i, i))
    < wavPreprocesingT : callable
    < ensembleSettings : dictionary with the instructions for ensemeble generation

    Return
    ------    
    > datXy_names : data object
    """

    ### check feature extraction function
    if not callable(featExtFun): # dictionary or None (defaul parameters)
        featExtFun = wavFeatureExtractionSplit(featExtFun).featExtrFun() # default
    if not callable(wavPreprocesingT): 
        wavPreprocesingT = lambda x, y : x
    if ensembleSettings is None:
        ensembleSettings = dict(effectName='addWhiteNoise', param_grid=np.ones(1) )
    ### check existance of provided files    
    assert os.path.isfile(wavF), "%s\ndoesn't exists"%wavF
    assert os.path.isfile(annF), "%s\ndoesn't exists"%annF
                
    ### extract features for each annotated section
    segmentsLi, fs = sT.getAnnWavSec(wavF, annF)
    #assert sr==fs, "noise and signal waves have different sampling rates"

    datO = myML.dataXy_names()     
    ## for each annotation in the wavfile compute the features
    for annIndex in range(len(segmentsLi)): 
        label = segmentsLi[annIndex]['label']
        waveform = segmentsLi[annIndex]['waveform']
        waveform = wavPreprocesingT(waveform, fs)  # preproces waveform

        Y = eff.generateWaveformEnsemble( waveform,  **ensembleSettings) ## noise
        
        for i in range(len(Y)):#np.shape(Y)[0]):
            #M, _, _, featStr = featExtFun(Y[i], fs) #
            M, _, _, featStr = featExtFun(Y[i,:], fs)
            datO.addInstances(np.expand_dims(M.flatten(), axis=0), [np.array(label)])                                           
    
    return datO    
    
def wavAnnCollection2Xy_ensemble(wavAnnColl, featExtFun=None, wavPreprocesingT=None,
                               ensembleSettings=None):
    datO = myML.dataXy_names() 
    for wavF, annF in wavAnnColl[:]:
        datO_new = wavAnn2sectionsXy_ensemble( wavF, annF, featExtFun=featExtFun, 
                                                   wavPreprocesingT=wavPreprocesingT,
                                                   ensembleSettings=ensembleSettings) 
        datO.addInstances(datO_new.X, datO_new.y_names )
    
    return datO

    
def wavCollection2datXy(wavLabelCollection, featExtFun=None, wavPreprocesingT=None):
    """
    returns the data object of a collection of annotated wavs.
            
        ( •_•)O*¯`·. call type (classification) .·´¯`°Q(•_• )

    
    Parameters
    ----------
    < wavLabelCollection : list of tuples with the wav - annotation files
                        tu[0] : path to wav file
                        tu[1] : path to annotation label
    < featExtFun : feature extraction function OR
                            dictionary with the feature extraction settings
    < wavPreprocesingT : waveform preorocessing function
                            eg. filter

    Return
    ------    
    > datO :  a file with the paths to the features and their labels
    """   
    if isinstance(featExtFun, dict):
        featExtFun = functools.partial(sT.waveform2featMatrix, **featExtFun)
    if not callable(wavPreprocesingT): 
        wavPreprocesingT = lambda x, y : x
        
    datO = myML.dataXy_names() #inicialize data object    

    for wavF, l in wavLabelCollection:
        waveForm, fs = sT.wav2waveform(wavF)
        waveForm = wavPreprocesingT(waveForm, fs)        
        M, _, _, featStr = featExtFun(waveForm, fs)
        datO.addInstances(np.expand_dims(M.flatten(), axis=0), [l])  
        #print(np.shape(M0), datO.shape, np.shape(datO.y), os.path.basename(wavF))
    return datO        
    
### whale sound detector
    
def wavAnnCollection2datXy(WavAnnCollection, featExtFun=None, wavPreprocesingT=None):
    """
    !!!! split it into wavAnn2datXy + a for loop that does it for the whole collection
    !!!! see wavAnn2secionsXy and wavAnnCollection2sectionsXy
    
    returns the data object of a collection of annotated wavs.
            
        ( •_•)O*¯`·. whale sound detector (classification) .·´¯`°Q(•_• )

    
    Parameters
    ----------
    < WavAnnCollection : list of tuples with the wav - annotation files
                        tu[0] : path to wav file
                        tu[1] : path to annotation file
    < featureExtractionFun : feature extraction function OR
                            dictionary with the feature extraction settings
    < wavPreprocesingT : waveform preorocessing function
                            eg. filter

    Return
    ------    
    > datO :  a file with the paths to the features and their labels
    """   
    if isinstance(featExtFun, dict):
        #!!! featExtFun = wavFeatureExtractionWalk(featExtFun).featExtrFun()
        featExtFun = functools.partial(sT.waveform2featMatrix, **featExtFun)
    if not callable(wavPreprocesingT): 
        wavPreprocesingT = lambda x, y : x
        
    datO = myML.dataXy_names() #inicialize data object

    for wavF, annF in WavAnnCollection:
        waveForm, fs = sT.wav2waveform(wavF)
        waveForm = wavPreprocesingT(waveForm, fs)
        annotLi_t = sT.aupTxt2annTu(annF) ## in sample units
        M0, y0_names,  _, _ =  featExtFun(waveForm, fs, annotations=annotLi_t)
        datO.addInstances(M0, y0_names) 
        #print(np.shape(M0), datO.shape, np.shape(datO.y), os.path.basename(wavF))

    return datO
    

    
    

    
##### READ DATA FILES #####

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

def loadAndFlattenX(featuresFi):
    '''loads a features files (*.npy) into X vector'''
    A = np.load(featuresFi)
    if np.ndim(A) == 1:
        r=len(A); c = 1
    else:
        r,c = np.shape(A)
    return A.reshape(1, r*c) #A.flatten()#

    
def readFeaturesFileList2Xy(featuresFiList, annotated=True):
    '''
    Read a list of path feature files (*.npy) into X, y format
    Parameters
    ----------
    featuresFiList : collection of feature files (npy) and annotation (only if annotated=True)
                        li[0] : path to feature file of ONE instance
                        li[1] : label
    Returns
    -------
    X, y :  feature matrix and labels
            np.arrays
    '''  
    m = len(featuresFiList)
    
    if annotated:
        n = np.shape(loadAndFlattenX(featuresFiList[0][0]))[1]
        X = np.zeros((m,n))
        y = np.zeros((m), dtype='|S4')
        for i in range(m):
            X[i] = loadAndFlattenX(featuresFiList[i][0])
            y[i] = featuresFiList[i][1]
        return(X, y)
    else:
        n = np.shape(loadAndFlattenX(featuresFiList[0]))[1]
        X = np.zeros((m,n))
        y = np.zeros((m,1))
        for i in range(m):
            X[i] = loadAndFlattenX(featuresFiList[i][0])
        return(X, y)
    

### WRITE to data files

def addColParsingCol(fName, colIndex=0, outFileName=None, sep='\t', comment = '#',
                key = "call", parserule = fp.parseHeikesNameConv):
    '''
    Read the columns "colIxes" of a file "fName"
    Parameters
    ----------
    fName : file to read
    colIndex : column to parse
    outFileName : out file name, 
                    None overwrites fName
    parserule : parsing rule, parses the info in column <colIndexe> giving 
                a dictionary back to be read with <key>
    key :  dictionary key from the parserule
    Returns
    -------
    outFileName : output file name
    '''    
    if outFileName is None: outFileName = fName 
        
    with open(fName,'r') as f:
        lines = f.readlines()
        
    with open(fName,'w') as f:
        for li in lines:
            if not li.startswith(comment):
                parsed = parserule(li.strip().split()[colIndex])[key]
                newLine = li.strip() + sep + parsed +'\n'
                f.write(newLine)
            else:
                f.write(li)
    return fName      
    
####Split collections    
    
def splitCollectionRandomly(collection, trainFraction = 0.75):
    '''
    splits a list
    '''
    m=len(collection)
    np.random.shuffle(collection)
    return collection[:int(m*trainFraction)], collection[int(m*trainFraction):]    
    
    
class wavFeatureExtraction():
    """class for the extraction of wav features, statring from a dictionary of settings
    bounds:
        * feature extraction function
        * feature string
        * settings dictionary
    """
    def __init__(self, feature_extr_di):
        self.newFeatExtrDi( feature_extr_di )
               
    def set_featureExtStr(self, di):
        '''defines a string with the feature extraction settings'''
        #print(di)
        featStr = di['featExtrFun']+'-'
        di.pop('featExtrFun')
        featStr += '-'.join([str(c)+str(v) for (c,v) in di.items()])
        return featStr
    
    def featExtrFun(self):
        '''returns the feature extraction callable, ready to use!'''
        return functools.partial(sT.waveform2featMatrix, **self.feature_extr_di)
    
    def newFeatExtrDi(self, feature_extr_di):
        '''updates the feature-extractio-dictionary and string'''
        if isinstance(feature_extr_di, dict): 
            self.feature_extr_di = feature_extr_di
            self.feature_str = self.set_featureExtStr(dict(self.feature_extr_di))
            
class wavFeatureExtractionWalk(wavFeatureExtraction):
    """class for the extraction of wav features 
    by framing the signal walking in steps of textWS"""
    def __init__(self, feature_extr_di=None):
        if feature_extr_di is None:
            feature_extr_di = self.defaultFeatureExtractionDi()
        wavFeatureExtraction.__init__(self, feature_extr_di) #sets string and dictionary
        self.newFeatExtrDi( feature_extr_di )
        
    def defaultFeatureExtractionDi(self):
        '''default settings'''
        feExDict = {'featExtrFun' : 'melspectro', 'textWS' : 0.1, 'n_mels' : 2**4 }
        return feExDict
        
class wavFeatureExtractionSplit(wavFeatureExtraction):
    """class for the extraction of wav features by splitting section into Nslices"""
    def __init__(self, feature_extr_di=None):
        if feature_extr_di is None:
            feature_extr_di = self.defaultFeatureExtractionDi()
        wavFeatureExtraction.__init__(self, feature_extr_di) #sets string and dictionary
        self.newFeatExtrDi( feature_extr_di )
        
    def defaultFeatureExtractionDi(self):
        '''default settings'''
        feExDict = {'featExtrFun' : 'melspectro', 'Nslices' : 10, 'n_mels' : 2**4 }
        return feExDict            
            
            
            