from __future__ import print_function, division
import os
import sys

from collections import Counter
import shutil

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import utils as sku

import numpy as np
import scipy.io.arff as arff
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import colors  # for plXy

from subprocess import call
import pandas as pd
import seaborn as sns

import pylotwhale.signalProcessing.audioFeatures as auf
import pylotwhale.signalProcessing.signalTools as sT

import featureExtraction as fex

"""
    Preprocessing function for doing machine learning
    florencia @ 16.05.15

"""

#warnings.simplefilter('always', DeprecationWarning)
#DEPRECATION_MSG = ("use MLEvalTools")

#################################################################################
##############################    FEATURES    ##################################
#################################################################################


####    preprocessing    ##################################


def rescale(M):
    '''
    rescales the columns of a matrix so that ots values lay in [0,1]
    substracting the mean and dividing by the range (max-min),
    also called normalisation
    '''
    return np.divide((1.0*M - np.min(M, axis=0)), np.max(M, axis=0) - np.min(M, axis=0) )


def standarize(M):
    '''
    standarized the columns of a matrix, so that they have mean = 0 and std = 1
    '''
    return 1.0*(M - np.mean(M, axis=0))/np.std(M-np.mean(M, axis=0), axis=0)


def colCounts2colFreqs(M):
    '''tranform column counts to column frequencies
    by dividing the matrix columns by total counts of each column'''
    return np.divide(1.*M, 1.*M.sum(axis=0))


def scale(M, normfun=rescale, axis=1):
    '''
    rescales a matrix
    axis = 0 (columns)
         = 1 (rows)
         
    normfun 'normalization function'
                {rescale, standarize, colCounts2colFreqs}
                
    ! this function replazed normalizeMatrix()
    '''
    if axis == 1:
         return normfun(M.T).T
        
    if axis == 0:
         return normfun(M)
         
    
def removeBuggs_idx(M, axis=0):
    '''     
    Returns the indexes for wich the 
    rows (instances, axis=1) or columns (features, axis=0) have finite values
    axis=0 columns
    axis=1 rows
    < M : features matrix
    '''    
    Ms = np.sum(M, axis=axis) 
    idx = np.isfinite(Ms)
    return(idx)


def removeBuggyInstances(M, y=None):
    '''
    removes the instances (rows) with numeric singularities nan, inf
    < M : feature matrix, m x n
    < y : target array, m x 1
    m : number of instances
    n :  number of features
    -->
    > M : unbugged feature matrix
    > y : unbugged target array
    > idx : unbugging indexes
    '''
    if y is None:
        y=np.zeros(np.shape(M)[0])
    ## find buggy instances    
    idx = removeBuggs_idx(M, axis=1)
    return M[idx, :], y[idx], idx


def removeBuggyFeatures(M, y=None):
    '''
    removes the features (columns) with numeric singularities nan, inf
    < M : feature matrix, m x n  (m, number of instances)
    < y : feature array, n x 1  (n,  number of features)
    -->
    > M : unbugged feature matrix 
    > y : unbugged feature array
    > idx : unbugging indexes
    '''
    if y == None: y=np.zeros(np.shape(M)[1])
    ## find buggy instances    
    idx = removeBuggs_idx(M, axis=0)
    return M[idx, :], y[idx], idx      

####    visualizing    ############################


def plXy(X, y, figsize=None, cmapName_L='gray_r', cmapName_Fig='gray_r'):
    """Plot feateure matrix with labels

    Parameters
    ----------
    X : ndarray (n, m)
        features
    y : ndarray
        labels, (n,)
    figsize : listlike
        size of the figure
    cmapName_L : str
        name of the colormap for the labels
    cmapName_Fig : str
        name of the colormap for the features matrix

    Returns
    -------
    fig : matpltlib figure
    """

    ## initialise figure
    fig = plt.figure(figsize=figsize)
    ## define axes
    left, width, height = 0.1, 0.9, 0.9
    width_y, bottom_y = 0.05, 0.1
    bottom = bottom_y + width_y
    #
    rect_X = [left, bottom, width, height]  # [x0, y0, xf, yf ] - big  plot
    rect_Y = [left, bottom_y, width, width_y]  # small plot
    # add axes
    axX = fig.add_axes(rect_X)  # big plot for instance features
    axY = fig.add_axes(rect_Y)  # small plot for instance labels
    # remove tick labels
    axX.get_xaxis().set_visible(False)
    axY.get_yaxis().set_visible(False)
    # set axis labels
    axX.set_ylabel('features')
    axY.set_xlabel('instances')

    ## define labels mapping
    label_set = list(set(y))
    n_labels = len(label_set)
    y_map = {label: i for i, label in enumerate(label_set)}
    labels_cmap = sns.color_palette(cmapName_L, n_colors=n_labels)
    ## create keys for the labels
    artistLi = []
    txtLi = []
    # draw label keys

    for label, ix_label in y_map.items():
        # line
        artistLi.append(plt.Line2D((0, 1), (0, 0), color=labels_cmap[ix_label],
                        linewidth=14))
        txtLi.append(label)  # text
    axX.legend(artistLi, txtLi)

    #### PLOTS
    ## plot features figure

    axX.imshow(X, aspect='auto', interpolation='nearest',
               cmap=plt.cm.get_cmap(cmapName_Fig))
    ## labels
    ## map labels into numbers
    yL = [y_map[item] for item in y]
    y_num = np.array(yL, ndmin=2)
    # check dimensions
    m_instances_X = np.shape(X)[1]
    m_instances_y = np.shape(y_num)[1]
    if m_instances_y != m_instances_X:
        print('WARNING! X and y have different'
              'sizes ({:d} =/= {:d})'.format(m_instances_X, m_instances_y) )
    # plot labels
    cmap = colors.ListedColormap(labels_cmap)
    axY.imshow(y_num, aspect='auto', cmap=cmap, #plt.cm.get_cmap(cmapName_L),
               interpolation='nearest')

    return fig, axX, axY


### Operations over X, y data objects

def selectData(X, y, label):
    """Returns the data with the specified labels
    label: list like object"""
    ix = y == label
    return  np.array(X[ix,:]), np.array(y[ix])

def resample(X, y, random_state=1, **options):
    """Sample with replacement (bootstrap)
       **options: 
        n_samples, int; replace, bool"""
    return sku.resample(X, y, random_state=random_state, **options)

def shuffle(X, y, random_state=1, **options):
    """Shuffle arrays (permute)
    sample without replacement"""
    return sku.shuffle(X, y, random_state=random_state, **options)

def balanceToClass(X, y, class_label, random_state=1, shuffle_samples=False):
    """Balances data Xy to a given class, 
    classes with less data then class_label are left the same
    Parameters
    ----------
    class_label: str
    random_stateL int
        this is used twice: (1) for sampling the labels data and
        (2) for shuffling the arrays X, y before being returned when shuffle_samples is True
    shuffle_samples: bool
        arrays are stacked, having them arranged by label.  
        If True, arrays are shuffle before being returned
    Returns
    -------
    balX, baly
    """
    ## get data of the class_label
    balX, baly = selectData(X, y, class_label)
    
    n_balclass = len(baly) # count samples
    ## labels of the rest of the classes
    balance_labels = set(y) - set([class_label])

    for l in balance_labels:  # for each of the other classes
        thisX, thisy = selectData(X, y, l)  # get class data
        class_n_samples = np.min((n_balclass, len(thisy)))
        ## select class samples randomly
        sX, sy = resample(thisX, thisy, random_state=random_state,
                          n_samples=class_n_samples, replace=False)
        ## stack
        balX = np.vstack((balX, sX))
        baly = np.hstack((baly, sy))

    ## shuffle again not to have X, y stacked by class
    if shuffle_samples is True:
        balX, baly = shuffle(balX, baly, random_state=random_state, replace=False)

    return balX, baly #np.array(balX), np.array(baly)


################# classes

class dataX:
    """
    features object -- unannotated data
    < X : the data matrix  ( # instnces X #features )
            or a tuple of data matrixes (X1, X2, ...)
    < attrNames : labels array ( # instances X 1 )
    annotated, tells if the data set should be trated as an annotated (True)
        containing the ground truth or not (False)
    """

    def __init__( self, X=None, attrNames=None, datStr=''):       
        #print("TEST", arffFile)
        
        self.load_X(X)
        self.nameAttribuites(attrNames)
        self.datStr=datStr
        #print( len(self.attrNames), self.n_attr)# '# targets must match n'
        
    def load_X(self, X):
        if isinstance(X, tuple): # stack new instances
            self.X = np.vstack(X)
            self.shape = self.m_instances, self.n_attr = np.shape(self.X)

        elif X is None: # no instances
            self.X = self.m_instances = self.n_attr = self.shape = None
        else: # load the first set of instances
            self.X = X
            self.shape = self.m_instances, self.n_attr = np.shape(self.X)
                
    def addInstances(self, newX):
        '''adds instances'''
        m, n= np.shape(newX)
        if self.X is None:
            self.load_X(newX)
        else:
            assert(self.n_attr is None or n == self.n_attr), "different feature spaces {} =/= {}".format(n, self.n_attr)
            self.load_X((self.X, newX))
        
    def nameAttribuites(self, attrNames):
        if self.n_attr is None: # no data
            self.attrNames = None
        elif attrNames is not None: # attr names given
            self.attrNames = attrNames # features + output
            assert len(self.attrNames) == self.n_attr, "# targets must match n"
        else: # generate names
            self.attrNames = np.arange(self.n_attr) # features + output


class dataXy_names(dataX):
    """
    features object -- annotated data
    Parameters:
    ------------
        X : the data matrix  ( # instnces X #features )
            or a tuple of data matrixes (X1, X2, ...)
        y_names : array with the names of the names ( # instances (m))
        attrNames : names array ( # features X 1 )
    """
    
    def __init__(self, X=None, y_names=None, attrNames=None, datStr=''):
        self.X = None
        dataX.__init__(self, X=X, attrNames=None, datStr='')
        self.load_y_names(y_names) # self.y_names
        
    def load_y_names(self, y_names):
        '''loads and updates y_names and the label encoder
        > self.y_names
        > self.y
        '''
        if isinstance(y_names, tuple) : # load more data (tuple w/ old and new names)
            self.y_names = np.concatenate(y_names)

        elif y_names is None: # nothing to load
            self.y_names = None
        else: # load for the first time
            self.y_names = y_names
    
    def addInstances(self, new_X, new_y_names):
        '''adds annotated instances
        new_X : feature matrix
                    np.array (m, n)
        new_y_names : targets
                    np.array (m,)
        '''
        try:
            m, n = np.shape(new_X) # m (instances) x n (features)
        except ValueError: # notthing to add new_X=None
            return None
            
        ## check dimensions
        if self.X is None: # first time we load data
            self.load_X(new_X)
            self.load_y_names(new_y_names)
        else: # stack data
            assert(self.n_attr is None or n == self.n_attr), "different feature spaces {} =/= {}".format(n, self.n_attr)
            self.load_X((self.X, new_X))
            assert(m == len(new_y_names)),  "inconsistent labeling {} =/= {}".format(m, len(new_y_names))
            self.load_y_names((self.y_names, new_y_names))
        
    def checkDimesions(self, A, a_names):
        m, n = np.shape(A) # m (instances) x n (features)
        if m == len(a_names):
            return True
        else:
            print("invalid data\n# names should match # instances :{} =/= {}".format(m, len(a_names)))
            return False
        
    def targetFrequencies(self):
        return( dict(Counter(self.y_names) ))
        
    def filterInstances(self, y_namesSet=None, A=None, a=None ):
        '''
        returns the instances with y_names in the y_namesSet
        Parameters
        ----------    
        y_namesSet: list of y_names to keep
                        if None, don't filter
        A: feature matrix
        b: target vector (nominal)

        Returns
        -------
        A_filt : filtered A
        a_filt : filtered a
        '''
        if A is None:
            #print("TEST ------ default")
            A = self.X
            a = self.y_names

        if y_namesSet is None:  # don't filter
            return( A, a)
        else:
            selector = np.in1d(a, y_namesSet)
            return( A[selector, :], a[selector])


class dataXy(dataXy_names):
    """
    features object -- annotated data
    Parameters:
    ------------
        X : the data matrix  ( # instnces X #features )
            or a tuple of data matrixes (X1, X2, ...)
        y_names : array with the names of the names ( # instances (m))
        attrNames : names array ( # features X 1 )
    """
    
    def __init__(self, X=None, y_names=None, attrNames=None, datStr=''):
        dataXy_names.__init__(self, X=X, y_names=y_names, attrNames=None, datStr='')
                
    def fitLabelEncoder(self, y_names):
        self.le = preprocessing.LabelEncoder()
        self.y = np.array(self.le.fit_transform(y_names))
        return self.le
            
    def num2nom(self, num):
        '''nominal --> numeric'''
        if np.ndim(num) == 0: num = [num]
        try:
            return self.le.inverse_transform(num)
        except AttributeError:
            print('ERROR! call first self.fitLabelEncoder()')
        except ValueError:
            print('ERROR! y contains new labels')
    
    def nom2num(self, nom):
        '''target name (nominal) --> target (numeric)'''
        try:
            return self.le.transform(nom)
        except AttributeError:
            print('ERROR! call first self.fitLabelEncoder()')
        except ValueError:
            print('ERROR! y contains new labels')
        
    def targetNumNomDict(self):
        '''> target dict { num : label }'''
        y_namesSet = list(set(self.y_names))
        return dict(zip(self.nom2num(y_namesSet), y_namesSet))
                
class _LabelEncoder(preprocessing.LabelEncoder):
    def __init__(self):
        preprocessing.LabelEncoder.__init__(self)                


class labelTransformer():
    """
    label encoder for transforming nominal labels into numeric values
    """

    def __init__(self, y_names):
        self.le = _LabelEncoder()
        self.le.fit_transform(y_names)
        self.classes_ = self.le.classes_

    def _num2nom(self, num):
        try:
            return self.le.inverse_transform(num)
        except ValueError:
            print('ERROR! y contains new labels')     
            return np.array([None])


    def num2nom(self, num):
        '''nominal --> numeric'''
        if np.ndim(num) == 0:
            num = np.array([num])
            return self._num2nom(num).item()
        else:
            return self._num2nom(num)

    def _nom2num(self, nom):
        '''target name (nominal) --> target (numeric)'''
        try:
            return self.le.transform(nom)
        except ValueError:
            print('ERROR! y contains new labels')
            return np.array([None])

    def nom2num(self, nom):
        '''target name (nominal) --> target (numeric)'''
        if np.ndim(nom) == 0: 
            nom = np.array([nom])
            return self._nom2num(nom).item()
        else:
            return self._nom2num(nom)


    def targetNumNomDict(self):
        '''> target dict { num : label }'''
        y_namesSet = list(self.le.classes_)
        return dict(zip(self.nom2num(y_namesSet), y_namesSet))


###### save model

def saveModel(clf, outModelName, fileModelName_fN=None, feExSettings=None):
    '''saves clf model and appends model_name in a register file
    Parameters:
    -----------
    clf : estimator
        classifier we want to save as pkl for later predictions
    outModelName : string
        settings of the model- features, clf, etc
    fileModelName_f : string
        name of the file where the clf-models are being filed'''
    try:
        shutil.rmtree(outModelName)
    except OSError:
        pass
    os.mkdir(outModelName)
    outModelName += '/model.pkl'
    joblib.dump(clf, outModelName)
    if fileModelName_fN:
        with open(fileModelName_fN, 'a') as out_file:
            out_file.write("{}\n".format(outModelName))

    if feExSettings:
        ### TO BE IMPLEMENTED save feExSettings
        pass
    return outModelName
