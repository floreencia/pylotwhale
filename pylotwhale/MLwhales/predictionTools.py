from __future__ import print_function, division  # py3 compatibility
import numpy as np
import os
import featureExtraction as fex
import pylotwhale.signalProcessing.signalTools as sT
import pylotwhale.utils.annotationTools as annT
import pylotwhale.signalProcessing.audioFeatures as auf

"""
Tools for transforming classifier predictions into annotation sections and files
"""

### WSD -- splitting -- PREDICT WAV SECTIONS (SOUND TYPE)


def predictSoundSections(wavF, clf, lt, feExFun,
                         outF='default', annSections='default'):
    '''
    predicts and generates the annotations of the given wavF walking

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

    oF = predictSectionsFromWaveform_genAnnotations(waveForm, fs, clf, lt, feExFun,
                                       outF=outF, annSections=annSections)
    return oF


def predictSectionsFromWaveform_genAnnotations(waveform, fs, clf, lt, feExFun,
                                               outF, annSections='default'):
    """
    predicts the sections of a waveform and generates annotations
    walking along a waveform
 
    Parameters
    ----------
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

    tf = 1.0 * len(waveform) / fs
    M0 = feExFun(waveform) #, annotations=annotLi_t)
    y_pred = clf.predict(M0)
    annT.predictions2txt(lt.num2nom(y_pred), outF, tf, sections=annSections)
    return outF


def predictAnnotations(waveform, fs, feExFun, lt, clf):  #, annSections=None):
    """
    predicts annotation sections of a waveform
    walking along a waveform

    Parameters
    ----------
    waveform : ndarray
        sound waveform
    clf : estimator
        classifier object
    lt : label transformer object
    feExFun : callable
        feature extraction
    annSections : array
        sections returned, if None all labels are kept

    Returns
    -------
    T : ndarray (#annotations, 2)
        initial and final time of the annotation sections
    labels: ndarray (#anns, )
        labels of the annotation sections
    """
    tf = 1.0 * len(waveform) / fs
    M0 = feExFun(waveform)  # annotations=annotLi_t)
    y_pred = clf.predict(M0)
    T, labels = annT.predictions2annotations(lt.num2nom(y_pred), tf)
    ### filter annotations to keep only annSections
    #mask = np.in1d(labels, annSections) L[mask,:], labels[mask]
    return T, labels


### WSD2

def WSD2predictionsTLanns(wavF, annWSD1, feExtFun, lt, WSD2_clf,
                         readSections, dt=0): #keepSections=None
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
    readSections: list like object
        array with the ann sections from WSD1 we want to reinterpret
    dt: float
        time buffer for reading around the desired annotation section
    keepSections: (DEPRECATED) list like object
    """
    waveform, fs = sT.wav2waveform(wavF)  # load waveform
    A = annT.anns2array(annWSD1)  # load annotations

    newT_list = []
    newL_list = []
    for t0i, t0f, l0 in A[:]:  # for each ann section
        if l0 in readSections:  # if section of interest (c)
            thisWaveform = auf.getWavSec(waveform, fs, t0i - dt, t0f + dt)
            ## predict annotations
            secT, secL = predictAnnotations(thisWaveform, fs, feExtFun,
                                            lt, WSD2_clf)
            newSectT = secT + t0i - dt  # relative to the orginal ann sections
            newT_list.append(newSectT)
            newL_list.append(secL)
            #outF = annT.save_TLannotations(newT, L, outF, opening_mode='a')
    newL = np.hstack((newL_list))
    newT = np.vstack((newT_list))
    return newT, newL


def WSD2predict(wavF, template_annF, WSD2_feExFun, lt, WSD2_clf, m, tf,
                readSections='default', labelsHierarchy='default'):
    """WSD2 predictions
    takes wavF, template_annF to predict instance labels with WSD2_clf
    Parameters
    ----------
    wavF: str
    template_annFL: str
    lt: LabelTransformer
        to map predictions
    m: int
        number of instances in wavF
    tf: float
        length of the wavF in seconds
    Returns
    -------
    y_pred_names: ndarray
        instance labels
    """
    if readSections == 'default':
        readSections = ['c']
    if labelsHierarchy == 'default':
        labelsHierarchy = ['c']
    ## ground truth vs. WSD2 annotations predictions
    T, L = WSD2predictionsTLanns(wavF, template_annF, WSD2_feExFun, lt, WSD2_clf,
                                 readSections=readSections)
    # instances
    y_pred_names = auf.annotations2instanceArray(T, L, m, tf,
                                                 labelsHierarchy=labelsHierarchy)
    return y_pred_names


def WSD2predictAnnotations(wavF, annWSD1, feExtFun, lt, WSD2_clf, outF,
                           readSections, keepSections='default', dt=0):
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
        time buffer for reading around the desired annotation section
    """
    if keepSections is 'default':
        keepSections = ['c']
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
                                      WSD2_clf) #annSections=keepSections)
            newT = T + t0i - dt  # relative to the orginal ann sections
            mask = np.in1d(L, keepSections)
            outF = annT.save_TLannotations(newT[mask, :], L[mask], outF, opening_mode='a')
    return outF

### PREDICT THE LABELS OF THE ANNOTATED SECTION IN A WAV FILE  (CALL TYPE)


def predictFeatureCollectionAndWrite(inFile, clf, lt, col=0, outFile=None,
                                     sep='\t', stop=None):
    '''read a collection (indexfile) of features (*.npy)
    --> predict --> save predictions
    one prediction per file in the collection
    Parameters
    ----------
    inFile : str
        collection of feature files, in *.npy format
    clf : sklearn estimator
    lt : label transformer
    col : int
    outFile : str
        collection with the predicted labels, path to file
    sep: str
    stop: int
    Returns
    -------
    outFile: str
        path to output file
    '''
    if outFile is None:
        outFile = os.path.splitext(inFile)[0] + '-predictions.txt'

    try:  # remove file if exists
        os.remove(outFile)
    except OSError:
        pass
    

    with open(inFile) as f:  # read lines
        lines = f.readlines()

    with open(outFile, 'a') as g:  # predict
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


def TLpredictAnnotationSections(wavF, annF, clf, featExtFun, lt,
                                printProbs=False, readSections=None,
                                printreadSectionsC=True):
    """generates annotations predicting audio section classes
    Parameters
    ----------
    wavF : str
    annF : str
        path to the file with the annotation section to predict
    clf : estimator
    featExtFun : callable
    lt : labelTransformer
    printProbs : bool
    readSections : list of str
        regions in the annF for which we predict
    printreadSectionsC : bool
    """

    ## load annotations
    waveform, fs = sT.wav2waveform(wavF)
    T, L0 = annT.anns2TLndarrays(annF)
    ## set of annotation-sections to predict
    if readSections is None:
        readSections = np.array(list(set(L0)))
    ## filter for sections of interest
    IO_sections = np.isin(L0, readSections)
    Tp = T[IO_sections]
    L = L0[IO_sections]
    Lp = np.zeros_like(L)
    ## for each annotation section
    for i, label in enumerate(L):  # for each section
        waveformSec = auf.getWavSec(waveform, fs, *Tp[i])  # load waveform section
        M0 = featExtFun(waveformSec)  # extract features
        M = np.expand_dims(M0.flatten(), axis=0)
        Lp[i] = lt.num2nom(clf.predict(M))[0]  # predict

    return Tp, Lp


def predictAnnotationSections(wavF, annF, clf, featExtFun, lt, outFile=None,
                              sep='\t', printProbs=False, header='',
                              readSections=None, printreadSectionsC=True):
    """predicts annotations for call types sections
    Parameters
    ----------
    wavF: str
    annF: str
    clf: estimator
    featExtFun: callable
    lt: labelTransformer
    outFil: str
    sep: str
    printProbs: bool
    header: str
    readSections: list of str
        regions in the annF for which we predict
    printreadSectionsC: bool
    See also
    --------
       TLpredictAnnotationSections
       TODO: recode to use TLpredictAnnotationSections
    """

    if outFile is None: outFile = os.path.splitext(annF)[0] + '-sectionPredictions.txt'

    try:  # remove file if exists
        os.remove(outFile)
    except OSError:
        pass
                                       
    ## load files
    waveform, fs = sT.wav2waveform(wavF)
    T, L = annT.anns2TLndarrays(annF)
    if readSections == None:
        readSections = list(set(L))
    ## for each annotation section
    for i, label in enumerate(L):
        if label in readSections:
            waveformSec = auf.getWavSec(waveform, fs, *T[i] )
            ## predict
            try:
                M0 = featExtFun(waveformSec)  # estract features 
                M = np.expand_dims(M0.flatten(), axis=0)
                y_pred = lt.num2nom(clf.predict(M)) # predict label
            except AssertionError:
                y_pred = [label]
            ## write
            with open(outFile, 'a') as f:
                f.write("{}\t{}\t{}\t{}\n".format(T[i, 0], T[i, 1], label, *y_pred))
        elif printreadSectionsC:
            with open(outFile, 'a') as f:
                f.write("{}\t{}\t{}\t{}\n".format(T[i, 0], T[i, 1], label, label))

    return outFile


def predictAnnotationSections0(wavF, annF, clf, featExtFun, lt, outFile=None,
                              sep='\t', printProbs=False, header=''):
    '''
    Predicts the label (call types) of each annotated section and writes 
    the prediction into outFile
    Parameters
    ----------
    wavF : str
        wave file
    annF : str
        annotation file
    clf : sklearn fitted estimator
        classifier
    featExtFun :  callable
        feature extraction function
        or a dictionary with the feature extraction settings
        featureExtrationParams = dict(zip(i, i))
    '''
    ### out file handling
    if outFile is None: outFile = os.path.splitext(annF)[0] + '-sectionPredictions.txt'

    try:  # remove file if exists
        os.remove(outFile)
    except OSError:
        pass

    ## read data
    predO = fex.wavAnn2annSecs_dataXy_names(wavF, annF, featExtFun=featExtFun)
    ## predict
    predictions = np.expand_dims(lt.num2nom(clf.predict(predO.X)), axis=1)
    if printProbs:
        predictions = np.hstack((predictions, clf.predict_proba(predO.X)))
        header = '{}'.format(le.classes_)
    ## save file
    A = np.loadtxt(annF, delimiter='\t', dtype=object, ndmin=2) # usecols=[0,1])
    np.savetxt(outFile, np.hstack((A, predictions)), fmt='%s',
               delimiter = '\t', header=header)
    return outFile

