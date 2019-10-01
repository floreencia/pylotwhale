from __future__ import print_function, division  # py3 compatibility
import numpy as np
import functools

from pylotwhale.signalProcessing.signalTools import (
    wav2waveform,
    waveformPreprocessingFun,
    audioFeaturesFun,
)

import pylotwhale.signalProcessing.audioFeatures as auf

import pylotwhale.signalProcessing.effects as eff
from pylotwhale.utils.funTools import compose2

import pylotwhale.MLwhales.MLtools_beta as myML

import pylotwhale.utils.whaleFileProcessing as fp
from pylotwhale.utils.dataTools import stringiseDict

"""
Tools for extracting audio features from audio files and and its annotations
- create transformation, TransformationsPipeline
- extract
    - walking features
    - splitting features
"""


####### TRANSFORMERS
##### Feature extraction classes


def get_transformationFun(funName=None):
    """
    Dictionary of transformations
    maps method name (str) --> estimator (callable)
    """
    D = {}
    ## waveform --> waveform
    D.update(waveformPreprocessingFun())
    ## waveform --> audio features matrix
    D.update(audioFeaturesFun())
    ## audio features matrix --> summ clf features
    D.update(auf.summarisationFun())

    if funName in D.keys():
        return D[funName]
    else:
        return D


class Transformation:
    """Creates a transformation from a transformation_name
    and a settings dictionary
    A Transformation has a callable (fun) and a settings string
    """

    def __init__(self, transformation_name, settings_di):
        assert (
            transformation_name in get_transformationFun().keys()
        ), "trans" "formation not recognised\n {}".format(get_transformationFun().keys())
        self.name = transformation_name
        self.settingsDict = dict(settings_di)  # make copy of the dictionary

    @property  # turns the method into an attr
    def string(self):
        return self.set_transformationStr(self.settingsDict, self.name)

    @property
    def fun(self):
        return self.set_transformationFun(self.name, self.settingsDict)

    def set_transformationStr(self, di, settStr=""):
        """defines a string with transformation's instructions"""
        settStr += stringiseDict(di, "")
        return settStr

    def set_transformationFun(self, Tname, settings, transformationFun=get_transformationFun):
        """returns the feature extraction callable, ready to use!"""
        return functools.partial(transformationFun(Tname), **settings)

    def define_attrs_form_dict(self):
        for k, v in self.settingsDict.items():
            setattr(self, "__{}".format(k), v)


class TransformationsPipeline:
    """pipeline of Transformations
    Parameters
    ---------
    transformationsList: list
        [(<processing_step>, Transformation), ...]
    """

    def __init__(self, transformationsList):
        self.transformationsList = transformationsList

    @property
    def string(self):
        return self.set_string(self.transformationsList)

    @property
    def fun(self):
        return self.set_fun(self.transformationsList)

    @property
    def step_sequence(self):
        """list with the sequence of processing steps"""
        return self.set_step_sequence(self.transformationsList)

    @property
    def steps(self):
        """dictionary, maps processing_step to Transformation"""
        return self.set_steps(self.transformationsList)

    def set_string(self, T_list):
        s = ""
        for (step, trO) in T_list:
            s += "-{}-{}".format(step, trO.string)
        return s

    def set_fun(self, T_list):
        f = lambda x: x
        for (step, trO) in T_list:
            f = self.composeTransformation(trO.fun, f)
        return f

    def composeTransformation(self, fun2, fun1):
        return compose2(fun2, fun1)

    def set_steps(self, T_list):
        s = {}
        for (step, trO) in T_list:
            s[step] = trO
        return s

    def set_step_sequence(self, T_list):
        s = []
        for (step, trO) in T_list:
            s.append(step)
        return s

    def processing_steps(self):
        s = ""
        for k in self.step_sequence:
            s += "{}, {}, {}\n".format(k, self.steps[k].name, self.steps[k].settingsDict)
        return s


def makeTransformationsPipeline(settings):
    """creates a transformations pipeline
    settings: list
        ["step_name", ("transformation_name", settingsDict)]
        step_name: identifier for the name of the step
        transformation_name: must be a key from get_transformationFun
        """
    transformationsList = []
    for s, (tn, sD) in settings:
        transformationsList.append((s, Transformation(tn, sD)))
    return TransformationsPipeline(transformationsList)


##### READ DATA FILES #####


def readCols(fName, colIndexes, sep="\t"):
    """
    Read the columns "colIxes" of a file "fName"

    np.loadtxt(file, delimiter='\t', dtype='|S') can be used instead!
    Parameters
    ----------
    fName : file to read
    colIndexes : list with the indexes of the columns to read
    sep : column separator
    Returns
    -------
    collectionList : list of tuples with the wav - annotation files
                        tu[0] : path to wav file
                        tu[1] : path to annotation file
    """
    with open(fName) as f:
        lines = f.readlines()

    # li = [tuple([line.strip().split(sep)[ci] for ci in colIndexes]) for line in lines if not line.startswith('#')]
    li = [
        [line.strip().split(sep)[ci] for ci in colIndexes]
        for line in lines
        if not line.startswith("#")
    ]
    return li


def loadAndFlattenX(featuresFi):
    """loads a features files (*.npy) into X vector"""
    A = np.load(featuresFi)
    if np.ndim(A) == 1:
        r = len(A)
        c = 1
    else:
        r, c = np.shape(A)
    return A.reshape(1, r * c)  # A.flatten()#


def readFeaturesFileList2Xy(featuresFiList, annotated=True):
    """
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
    """
    m = len(featuresFiList)

    if annotated:
        n = np.shape(loadAndFlattenX(featuresFiList[0][0]))[1]
        X = np.zeros((m, n))
        y = np.zeros((m), dtype="|S4")
        for i in range(m):
            X[i] = loadAndFlattenX(featuresFiList[i][0])
            y[i] = featuresFiList[i][1]
        return (X, y)
    else:
        n = np.shape(loadAndFlattenX(featuresFiList[0]))[1]
        X = np.zeros((m, n))
        y = np.zeros((m, 1))
        for i in range(m):
            X[i] = loadAndFlattenX(featuresFiList[i][0])
        return (X, y)


### WRITE to data files


def addColParsingCol(
    fName,
    colIndex=0,
    outFileName=None,
    sep="\t",
    comment="#",
    key="call",
    parserule=fp.parseHeikesNameConv,
):
    """
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
    """
    if outFileName is None:
        outFileName = fName

    with open(fName, "r") as f:
        lines = f.readlines()

    with open(fName, "w") as f:
        for li in lines:
            if not li.startswith(comment):
                parsed = parserule(li.strip().split()[colIndex])[key]
                newLine = li.strip() + sep + parsed + "\n"
                f.write(newLine)
            else:
                f.write(li)
    return fName


####Split collections


def splitCollectionRandomly(collection, trainFraction=0.75):
    """
    splits a list
    """
    m = len(collection)
    np.random.shuffle(collection)
    return collection[: int(m * trainFraction)], collection[int(m * trainFraction) :]


class wavFeatureExtraction:
    """class for the extraction of wav features, stating from a dictionary of settings
    bounds:
        * feature extraction function
        * feature string
        * settings dictionary
    """

    def __init__(self, feature_extr_di):
        self.newFeatExtrDi(feature_extr_di)
        # self.feature_str = set_featureExtStr

    def set_featureExtStr(self, di):
        """defines a string with the feature extraction settings"""
        # print("TEST", di)
        # featStr = di['featExtrFun']+'-'
        # di.pop('featExtrFun')
        featStr = stringiseDict(di, "")
        return featStr

    def featExtrFun(self):
        """returns the feature extraction callable, ready to use!"""
        return functools.partial(auf.waveform2featMatrix, **self.feature_extr_di)

    def newFeatExtrDi(self, feature_extr_di):
        """updates the feature-extraction-dictionary and string"""
        if isinstance(feature_extr_di, dict):
            self.feature_extr_di = feature_extr_di
            self.feature_str = self.set_featureExtStr(dict(self.feature_extr_di))


class wavFeatureExtractionWalk(wavFeatureExtraction):
    """class for the extraction of wav features 
    by framing the signal walking in steps of textWS"""

    def __init__(self, feature_extr_di=None):
        if feature_extr_di is None:
            feature_extr_di = self.defaultFeatureExtractionDi()
        wavFeatureExtraction.__init__(self, feature_extr_di)  # sets string and dictionary
        self.newFeatExtrDi(feature_extr_di)

    def defaultFeatureExtractionDi(self):
        """default settings"""
        feExDict = {"featExtrFun": "melspectro", "textWS": 0.1, "n_mels": 2 ** 4}
        return feExDict


class wavFeatureExtractionSplit(wavFeatureExtraction):
    """class for the extraction of wav features by splitting section into Nslices"""

    def __init__(self, feature_extr_di=None):
        if feature_extr_di is None:
            feature_extr_di = self.defaultFeatureExtractionDi()
        wavFeatureExtraction.__init__(self, feature_extr_di)  # sets string and dictionary
        self.newFeatExtrDi(feature_extr_di)

    def defaultFeatureExtractionDi(self):
        """default settings"""
        feExDict = {"featExtrFun": "melspectro", "Nslices": 10, "n_mels": 2 ** 4}
        return feExDict


#### DO WE STILL NEED THIS PART?!!!


### ensemble generation


def wavFAnnF2sections_wavsEnsemble_datXy_names(
    wavF, annF, featExtFun=None, wavPreprocessingT=None, ensembleSettings=None
):
    """
    Computes the features of each annotated section in the wav file
    ment to be used with feature extraction 'split' 

    
    Parameters:
    ----------
    wavFi: str
        path to wave file
    featExtFun: callable
        feature extraction function function
    wavPreprocessingT : callable
        applied before ensemble generation
    ensembleSettings: dict
        instructions for ensemble generation

    Return:
    ------
        > datXy_names : data object
    """

    ### check feature extraction function
    if not callable(featExtFun):  # dictionary or None (default parameters)
        featExtFun = wavFeatureExtraction(featExtFun).featExtrFun()  # default
    if not callable(wavPreprocessingT):
        wavPreprocessingT = lambda x, y: x
    if ensembleSettings is None:
        ensembleSettings = dict(effectName="addWhiteNoise", generate_data_grid=np.ones(1))

    ### extract features for each annotated section
    segmentsLi, fs = auf.getAnnWavSec(wavF, annF)
    # assert sr==fs, "noise and signal waves have different sampling rates"

    datO = myML.dataXy_names()
    ## for each annotation in the wavfile compute the features
    for annIndex in range(len(segmentsLi)):
        label = segmentsLi[annIndex]["label"]
        waveform = segmentsLi[annIndex]["waveform"]
        ##
        waveform = wavPreprocessingT(waveform, fs)  # preprocess waveform
        ## generate ensemble
        Y = eff.generateWaveformEnsemble(waveform, **ensembleSettings)  ## noise
        # Extrac
        for i in range(len(Y)):  # np.shape(Y)[0]):
            # M, _, _, featStr = featExtFun(Y[i], fs) #
            M = featExtFun(Y[i, :])
            datO.addInstances(np.expand_dims(M.flatten(), axis=0), [np.array(label)])

    return datO


def wavAnnCollection2Xy_ensemble_datXy_names(
    wavAnnColl, featExtFun, wavPreprocessingT=None, ensembleSettings=None
):

    datO = myML.dataXy_names()  # initialise data object
    for wavF, annF in wavAnnColl[:]:
        datO_new = wavFAnnF2sections_wavsEnsemble_datXy_names(
            wavF,
            annF,
            featExtFun=featExtFun,
            wavPreprocessingT=wavPreprocessingT,
            ensembleSettings=ensembleSettings,
        )
        datO.addInstances(datO_new.X, datO_new.y_names)

    return datO


def annotationsSamplesSpace(wavAnnColl):
    """create dictionary of labels and a list of waveforms from collection"""
    sampSpaceDi = {}
    for wavF, annF in wavAnnColl:
        segmentsLi, fs = auf.getAnnWavSec(wavF, annF)
        for i, segment in enumerate(segmentsLi):
            label = segment["label"]
            waveform = segment["waveform"]
            if label not in sampSpaceDi.keys():
                sampSpaceDi[label] = [waveform]
            else:
                sampSpaceDi[label].append(waveform)
    return sampSpaceDi


def waveformsLi2DatXy_names(waveformsLi, label, feExFun, nInstances):
    """Extracts features from an waveformlist and returns data object"""
    n_samps = len(waveformsLi)
    stopIdx = None
    if n_samps > nInstances:
        stopIdx = nInstances

    datO = myML.dataXy_names()  # initialise data object
    for waveform in waveformsLi[:stopIdx]:
        M = feExFun(waveform)
        datO.addInstances(np.expand_dims(M.flatten(), axis=0), [np.array(label)])
    return datO


def waveformsLi2aritificial_DatXy_names(
    waveformsLi, label, feExFun, n_instances, **ensemble_settings
):
    """takes a list of waveforms, all with the same label, generates artificial samples, 
    extracts features and returns data object
    Parameters
    ---------
    n_instances: int
        total number of artificial samples (instances) to generate
    ensemble_settings: dict
        kwargs for the generation of artificial samples
        see exT.generateData_ensembleSettings(n_artificial_samples=1)
    """
    n_samps = len(waveformsLi)
    # indices to take different waveforms until created desired number of samples
    indices = np.arange(n_instances) % n_samps
    datO = myML.dataXy_names()  # initialise data object

    for i in indices:
        waveform = waveformsLi[i]
        artificial_waveform = eff.generateWaveformEnsemble(waveform, **ensemble_settings)[0]
        art_samp = feExFun(artificial_waveform)
        datO.addInstances(np.expand_dims(art_samp.flatten(), axis=0), [np.array(label)])
    return datO


def extractFeaturesWDataAugmentation(sampSpace, feExFun, n_instances=10, **ensSettings):
    """Prepares data with the labels in wavAnnCollection, 
    balancing the classes generating artificial samples
    Parameter
    ---------
    sampSpace: dict
        labels and waveforms (samples space)
    feExfun: callable
    n_instances: int
    ensemble_settings: dict
        kwards for the generation of artificial samples
        see exT.generateData_ensembleSettings(n_artificial_samples=1)"""

    datO = myML.dataXy_names()  # data object
    for call in sampSpace.keys():
        ### extract features from original samples
        dat = waveformsLi2DatXy_names(sampSpace[call], call, feExFun, nInstances=n_instances)
        datO.addInstances(dat.X, dat.y_names)
        n_art_instances = n_instances - dat.m_instances
        ### generate artificial samples
        datArt = waveformsLi2aritificial_DatXy_names(
            sampSpace[call], call, feExFun, n_instances=n_art_instances, **ensSettings
        )
        datO.addInstances(datArt.X, datArt.y_names)
    return datO


### Extract features
### hierarchical annotations (WALKING)


def get_DataXy_fromWavFannF(wavF, annF, feExFun, labelsHierarchy):
    """
    extracts features and its labels (ground truth) from wavF and annF files
    and returns its dataXy_names instance
    ----------
    wavF: str
    annF: str
    feExFun: callable
    labelsHierarchy: list
    """
    # np.loadtxt(collFi, delimiter='\t', dtype='|S')
    # print("\n==========\nTEST\n==============",wavF)
    waveForm, fs = wav2waveform(wavF)
    tf = len(waveForm) / fs

    M0 = feExFun(waveForm)
    m = len(M0)
    y0_names = auf.annotationsFi2instances(annF, m, tf, labelsHierarchy=labelsHierarchy)
    datO = myML.dataXy_names(M0, y0_names)
    return datO


def getXy_fromWavFAnnF(wavF, annF, feExFun, labelsHierarchy, filter_classes=None):
    """Extract features from wavfile and labels from annotations
    returns feature matrix (X) and labels (y_names)"""

    datO = get_DataXy_fromWavFannF(wavF, annF, feExFun, labelsHierarchy)
    X, y_names = datO.filterInstances(filter_classes)
    return X, y_names


def wavAnnCollection2datXy(WavAnnCollection, feExFun=None, labelsHierarchy="default"):
    """
    Extracts features and labels from wav-ann collection
    Parameters
    ----------
    WavAnnCollection: list of tuples
        [(<path to wavF>, <path to annF>), ...]
    feExFun: callable
        feature extraction function
    labelsHierarchy: list
        labels in hierarchical order for setting the label of the instances (WALKING)

    Return
    ------    
    > datO :  a file with the paths to the features and their labels
    """
    if labelsHierarchy == "default":
        labelsHierarchy = ["c"]

    datO = myML.dataXy_names()  # initialise data object

    for wavF, annF in WavAnnCollection:
        X, y0_names = getXy_fromWavFAnnF(wavF, annF, feExFun, labelsHierarchy)
        datO.addInstances(X, y0_names)

    return datO


def wavAnnCollectionFi2datXy(WavAnnCollectionFi, feExFun=None, labelsHierarchy="default"):
    """
    Extracts features and labels from wav-ann collection    
    Parameters
    ----------
    WavAnnCollectionFi: str
        path to file
    feExFun: callable
        feature extraction function
    labelsHierarchy: list
        labels in hierarchical order for setting the label of the instances

    Return
    ------
    > datO :  a file with the paths to the features and their labels
    """
    coll = np.genfromtxt(WavAnnCollectionFi, dtype=object)

    return wavAnnCollection2datXy(coll, feExFun, labelsHierarchy)


### 1 section x feature vector (SPLITTING)


def wavAnn2annSecs_dataXy_names(wavF, annF, featExtFun=None):
    """
    Instantiates the annotated sections of a wavfile
    extracting a feature vector for each annotated section in the wav file
    meant to be used with feature extraction 'split'

    (n_n) used for call type classification Q (n_n) 
    Parameters
    ----------
    wavF: str
        path to wavefile
    annF: str
        path to wavfile
    featExtFun:  callable
        feature extraction function

    Returns
    -------
    datO: ML.dataXy_names
        classification features
    """

    ### extract features for each annotated section
    segmentsLi, fs = auf.getAnnWavSec(wavF, annF)

    datO = myML.dataXy_names()
    ## for each annotation in the wavfile compute the features
    for annIndex in range(len(segmentsLi)):
        label = segmentsLi[annIndex]["label"]
        waveform = segmentsLi[annIndex]["waveform"]
        M = featExtFun(waveform)
        datO.addInstances(np.expand_dims(M.flatten(), axis=0), [np.array(label)])

    return datO


def wavAnnCollection2annSecs_dataXy_names(wavAnnColl, featExtFun=None):
    """
    Computes the X, y for a collection of annotated wav files
    for each annotated section in the wav file
    meant to be used with feature extraction 'split'

    ........O Used for call type classification Q....
    
    Parameters
    ----------
    < wavAnnColl : collection of annotated wavfiles
    < featExtFun :  feature extraction function (callable)
                    or a dictionary with the feature extraction settings
                    featureExtractionParams = dict(zip(i, i))
    Return
    ------    
    > datXy_names : features object
    """

    datO = myML.dataXy_names()

    for wavF, annF in wavAnnColl[:]:
        # datO_test_new = wavAnn2sectionsXy( wF, annF, featExtFun=featExtFun) #wavPreprocessingT = wavPreprocessingFun )
        datO_new = wavAnn2annSecs_dataXy_names(
            wavF, annF, featExtFun=featExtFun
        )  # wavPreprocessingT = wavPreprocessingFun )
        datO.addInstances(datO_new.X, datO_new.y_names)

    return datO


def wavAnnCollection2datXyDict(wavAnnColl, featExtFun=None):
    """
    Computes the Xy-data-object and save it as a dictionary, 
    using the wavF and annF as dictionary keys,
    for a collection of annotated wav files
    meant to be used with feature extraction 'split' 

    .... Used for call type classification .Q....
    
    Parameters
    ----------
    < wavAnnColl : collection of annotated wavfiles
    < featExtFun : feature extraction function (callable)
                    or a dictionary with the feature extraction settings
                    featureExtractionParams = dict(zip(i, i))
    Return
    ------    
    > XyDict : dictionary of features object
    """
    ### check feature extraction function
    if not callable(featExtFun):  # dictionary or None (defaul parameters)
        featExtFun = wavFeatureExtraction(featExtFun).featExtrFun()  # default

    XyDict = {}

    for wF, annF in wavAnnColl[:]:
        datO_test_new = wavAnn2annSecs_dataXy_names(
            wF, annF, featExtFun=featExtFun
        )  # wavPreprocessingT = wavPreprocessingFun )
        XyDict["{}, {}".format(wF, annF)] = (datO_test_new.X, datO_test_new.y_names)

    return XyDict


def wavLCollection2datXy(wavLabelCollection, fs=None, featExtFun=None):
    """
    returns the data object of a collection of labelled wavs

        ..... call type (classification) ....

    Parameters
    ----------
    wavLabelCollection : list of tuples
        tu[0] : path to wav file
        tu[1] : wav label
    featExtFun : callable

    Return
    ------
    > datO: myML.dataXy_names
        data
    """

    datO = myML.dataXy_names()  # initialise data object

    for wavF, l in wavLabelCollection:
        waveForm, fs = wav2waveform(wavF, fs=fs)  # , normalize=False)
        M = featExtFun(waveForm)
        datO.addInstances(np.expand_dims(M.flatten(), axis=0), [l])

        # print(np.shape(M0), datO.shape, np.shape(datO.y), os.path.basename(wavF))
    return datO
