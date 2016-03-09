#!/usr/mprg/bin/python

from __future__ import print_function
import numpy as np
import scipy.io.arff as arff
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from collections import Counter
#from mpl_toolkits.axes_grid1 import ImageGrid
from subprocess import call
import pandas as pd

import os
import sys

import pylotwhale.signalProcessing.signalTools_beta as sT
import featureExtraction as fex


#from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.learning_curve import learning_curve
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix


#import time 
#import itertools as it
#import os
#import ast

#import sequencesO_beta as seqs
sys.path.append('/home/florencia/whales/scripts/')
#import matrixTools as mt 

"""
    Preprocessing function of machine learning
    florencia @ 16.05.15

"""

#################################################################################
##############################    FEATURES    ##################################
#################################################################################


####    preprocessing    ##################################


def rescale(M):
    '''
    rescales the columns of a matrix
    so that each feature varies between [0,1]
    '''    
    return np.divide((1.0*M - np.min(M, axis=0)), np.max(M, axis=0) - np.min(M, axis=0) )

         
def standarize(M):
    '''
    standarized the columns of a matrix
    ''' 
    return 1.0*(M-np.mean(M, axis=0))/np.std(M-np.mean(M, axis=0), axis=0) 
    
    
def scale(M, normfun=rescale, axis=1):
    '''
    rescales a matrix
    axis = 0 (columns)
         = 1 (rows)
         
    normfun 'normalization function'
                {rescale, standarize}
                
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
    if y == None: y=np.zeros(np.shape(M)[0])
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

         
def arffData2Xy(arffFile):
    '''
    reads an arff file ( output of bextract ) and returns:
    > the data matrix  ( #instnces X #features )
    > labels array ( #instances X 1 )
    '''    
    dat, metDat = arff.loadarff(open(arffFile))
    attrNames = metDat.names()
    N_attr = len(attrNames)
    M = np.array([dat[attrNames[i]] for i in range(len(attrNames[:-1])) ])
    
    labels = dat[attrNames[-1]]
    labSet = list(set(labels))
    N_labels = len(labels)
    print(N_attr, N_labels, labSet)
    
    return M.T, labels
    
    

    
def vis_dataXy(X, y, outPl='', plTitle=''):
    '''
    plots features (X) and labels y
    X, features matrix transposed (n x m)
    y, target vector (1 x m)
    m, # instaces
    n, # features
    '''    
    fig, ax = plt.subplots()#1,1,figsize=(5,len(y)/10))
    y[y==1]=np.max(X)
    ax.imshow(np.vstack((X, y,y,y,y)), aspect='auto', interpolation='nearest')#, cmap=plt.cm.Accent)
    ax.set_ylabel('features')
    ax.set_xlabel('insatances')
    if plTitle: ax.set_title(plTitle)
    if outPl: fig.savefig(outPl)


def plXy(X, y, y_ix_names=None, figsize=None, outFig='', plTitle='',
        cmapName_L = 'gray_r', cmapName_Fig = 'gray_r'):
    '''
    y_names : dictionary with the mapping between the target index and the names
             { ix : target } 
    imshows the:
        X matrix n x m
        y vector 1 x m
    '''    
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width, height = 0.1, 0.9, 0.9
    width_y, bottom_y = 0.05, 0.1
    bottom = bottom_y + width_y 

    rect_X = [left, bottom, width, height]  # [x0, y0, xf, yf ] - big  plot
    rect_Y = [left, bottom_y, width, width_y] # small plot

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=figsize)
    
    # plt.subplots()
    axX = fig.add_axes(rect_X) # big plot
    axY = fig.add_axes(rect_Y) # small plot
    
    # no tick labels
    axX.xaxis.set_major_formatter(nullfmt) 
    axY.yaxis.set_major_formatter(nullfmt)
    
    # axis labels
    axX.set_ylabel('features')
    axY.set_xlabel('instances')
    
    vmax=None
    #### target legends
    if type(y_ix_names)==dict:
        N=len(y_ix_names)
        vmax=N-1
        cmap = plt.cm.get_cmap(cmapName_L, N) 
        clrs = [cmap(i) for i in range(cmap.N)]
        artistLi=[]
        txtLi=[]
        for i in np.arange(N):
            #print(i, N, len(clrs))
            artistLi.append(plt.Line2D((0,1),(0,0), color=clrs[i], linewidth=14))
            #print(y_ix_names.keys(), y_ix_names.values())            
            #print(y_ix_names[i])            
            txtLi.append(y_ix_names[i])
        axX.legend(artistLi, txtLi)
        
    #### PLOTS
    ### figure    
    axX.imshow(X, aspect='auto', interpolation='nearest', 
               cmap=plt.cm.get_cmap(cmapName_Fig))
    ### annotations (labels)   
    if y.ndim == 1: y = y.reshape((1, len(y)) )
    ## check dimensions
    m_instances_X = np.shape(X)[1]
    m_instances_y = np.shape(y)[1]
    
    if m_instances_y != m_instances_X : 
        print('WARNING! X and y have different sizes (%d =/= %d)'%(m_instances_X, m_instances_y) )
    axY.imshow(y, aspect='auto', cmap=plt.cm.get_cmap(cmapName_L), vmax=vmax, 
               interpolation='nearest')#, bins=bins)
    
    if plTitle: axX.set_title(plTitle)#, bbox_inches='tight')
    # save
    if outFig: 
        #fig.tight_layout()        
        fig.savefig(outFig, bbox_inches='tight')


    
def vis_arff(arffFile, preproFun='standardize', outPl='', outDir='', plTitle=''):
    
    matrixTransf = {'rescale': rescale, 
                   'standardize': standarize, 
                   'unit': lambda x: x}
    
    if not outPl: # no out-file name
        outPl = arffFile.replace('.arff', '-%s.png'%preproFun) 
        if not outDir: # no out-dir
            outPl = outPl.replace('/arff/', '/images/') 
        else: # out Dir is given
            outPl = os.path.join(outDir, os.path.basename(outPl))
                
    X, y = arffData2Xy(arffFile)
    if not plTitle: plTitle = "(%d, %d) "%np.shape(X)

    ### labels: nom --> num
    le = preprocessing.LabelEncoder()
    y_num = le.fit_transform(y)
    
    M = scale(X.T, normfun = matrixTransf[preproFun])
    print(np.shape(M), np.shape(y_num), outPl)
    vis_dataXy(M, y_num, outPl, plTitle)
        
    
def wav2mf(wavFi, mfDir):
    '''
    creates mf colections with the given wav file
    the collection is stored in the mfDir
    '''
    bN = os.path.basename(wavFi)
    mfFi = os.path.join(mfDir, bN.replace('.wav','.mf'))
    with open(mfFi, 'w') as f:
        f.write(wavFi)
    return mfFi
    
def mf2arff(mfFi, arffDir, baseN=None, moreOptions='Default'):
    '''
    uses bextract to extract the features of a collection file (.mf)
    and stores the output in arffDir
    < mfFi, list of collections [col1.mf, ..., col2.mf]    
    < moreOptions = ['-m', '30', '-n'] (default)
        -m, size of the texture window
        -n, normalize audio file
        see bextract for more options
    < baseN == None, then this is automatically assigned as the base name
            of the firs file in the mf list
    '''
    ## bextract options
    if moreOptions == 'Default': moreOptions = ['-m', '30', '-n']
    ## check we have a list of collection files
    if (not type(mfFi) == list and type(mfFi)) == str: mfFi = [mfFi]
    assert type(mfFi) == list, 'mfFi, !list with <col>.mf'
    ### file handling for arff
    ## add paths
    mfFi = [os.path.abspath(os.path.expanduser(Fi)) for Fi in mfFi ]
    ## base name for arff file
    if baseN == None : baseN = os.path.basename(mfFi[0]).replace('.mf', '')
    ## add options to the base name    
    optStr = "_".join(moreOptions)
    arffFi = os.path.join(arffDir, baseN + '%s.arff'%optStr)
    
    command = ['bextract', '-fe'] + mfFi + ['-w', arffFi] + moreOptions 
    call(command)
    print(" ".join(command))

    return arffFi


def correctBextractLabelMess(X, y, m_cut=None, N_sections=1):
    """
    bextract seem to mess up with the labels when working in timeline mode
    but this seems to occurre only in the beginnig. This function filters out
    the first 2 (N_sections) sections
    X, feature matrix m x n
    y, m x 1
    """
    '''
    if not m_cut:
        le = preprocessing.LabelEncoder()
        y_num = le.fit_transform(y)
        m_cut = np.where(np.diff(y_num) !=0 )[N_sections]
    '''
    print(np.shape(X), m_cut)
    return X[m_cut:, :], y[m_cut:]
    


### X, y data objects


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
        m, n = np.shape(new_X) # m (instances) x n (features)
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
        
    def filterInstances(self, y_namesSet, A=None, a=None ):
        '''
        returns the instances with y_names in the y_namesSet
        Parameters
        ----------        
        y_namesSet : list of y_names to keep
        A : feature matrix
        b : target vector (nominal)
        
        Returns
        -------
        A_filt : filtered A
        a_filt : filtered a
        '''
        if A is None : 
            #print("TEST ------ default")
            A = self.X
            a = self.y_names
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
              
    def num2nom(self, num):
        '''nominal --> numeric'''
        if np.ndim(num) == 0: num = [num]
        try:
            return self.le.inverse_transform(num)
        except ValueError:
            print('ERROR! y contains new labels')
    
    def nom2num(self, nom):
        '''target name (nominal) --> target (numeric)'''
        try:
            return self.le.transform(nom)
        except ValueError:
            print('ERROR! y contains new labels')
        
    def targetNumNomDict(self):
        '''> target dict { num : label }'''
        y_namesSet = list(self.le.classes_)
        return dict(zip(self.nom2num(y_namesSet), y_namesSet))



#################################################################################
########################   CLASSIFIER EVALUATION    #############################
#################################################################################    

### Evaluate a list of classifiers over a collection   


def bestCVScoresfromGridSearch(gs):
    '''retieve CV scores of the best model from a gridsearch object
    Params:
    -------
        gs : gridsearch object
    Retunrs: (mu, std) of the bets scores
    --------
    '''
    mu, std = _bestCVScoresfromGridSearch(gs)
    assert mu - gs.best_score_ < 0.01, "retrived value doesn't match best score {} =/={}".format(mu, gs.best_score_)
    return mu, std
    
  
    
def _bestCVScoresfromGridSearch(gs):
    mu_max=0
    for pars, mu, cv_scrs in gs.grid_scores_[:]:
        if mu > mu_max:
            mu_max = mu
            cv_max = cv_scrs
 
    return np.mean(cv_max ), np.std(cv_max )


def printScoresFromCollection(feExFun, clf, lt, collFi, of):
    """
    clf : classifier
    le : label encoder (object)
    collfi : annotated wav collection (*.txt)
    of : out file (*.txt)
    """
    coll = fex.readCols(collFi, colIndexes =(0,1)) #np.loadtxt(collFi, delimiter='\t', dtype='|S')
    for wavF, annF in coll[:]:
        waveForm, fs = sT.wav2waveform(wavF)

        annF_bN = os.path.basename(annF)
        annotLi_t = sT.aupTxt2annTu(annF) ## in sample units

        M0, y0_names, featN, fExStr =  feExFun(waveForm, fs, annotations=annotLi_t)
        datO = dataXy_names(M0, y0_names)
        A, a_names = datO.filterInstances(lt.classes_)
        a = lt.nom2num(a_names)
        
        scsO = clfScoresO(clf, A, a)
        of.write("{}\t{}\n".format(scsO.scores2str(), annF_bN))
        

def clfGeneralizability(clf_list, wavAnnCollection, featExtFun, labelEncoder, labelSet=None):
    '''estimates the score of a list of classifiers, one score for each wav file'''
    clf_scores = [] #np.zeros(len(clf_list))
    for clf in clf_list: 
        acc, pre, rec, f1, size = coll_clf_scores(clf, wavAnnCollection, featExtFun, labelEncoder=labelEncoder, labelSet=labelSet)
        clf_scores.append( {"acc" : acc, "pre" : pre, "rec" : rec, "f1" : f1, "size" : size} )
    return clf_scores
            
def coll_clf_scores(clf, wavAnnCollection, featExtFun, labelTransformer, labelSet=None):
    '''estimates the score of a classifiers, for each wav file in the collection
    Parameters:
    < clf : classifier
    < wavAnnCollection : collection of wavs and annotations
                    list of tuples (<wav>, <txt>)
    < featExtFun : function(wavs, annotLi) --> A0, a_names, _, _
            import functools; 
            featExtFun = functools.partial(sT.waveform2featMatrix, 
            textWS=textWS, **featConstD)
    < labelEncoder : label encoder of the features
    < labelSet : list of labels to consider, if None => all labels are kept
    '''
    n = len(wavAnnCollection)
    acc = np.zeros(n)
    pre = np.zeros(n)
    rec = np.zeros(n)
    f1 = np.zeros(n)
    sizes = np.zeros(n)
    if labelSet is None: labelSet = labelTransformer.num2nom(clf.classes_ )
    i=0
    for wavF, annF in wavAnnCollection[:]:
        waveForm, fs = sT.wav2waveform(wavF) # read wavs
        annotLi_t = sT.aupTxt2annTu(annF) # read annotations
        A0, a_names, _, _ = featExtFun(waveForm, fs, annotations=annotLi_t)
        mask = np.in1d(a_names, labelSet) # filer unwanted labels
        a = labelTransformer.nom2num( a_names[mask]) #conver labels to numeric
        A = A0[mask]
        y_pred = clf.predict(A)
        ## scores
        acc[i] = accuracy_score(a, y_pred)
        pre[i]=precision_score(a, y_pred)
        rec[i]=recall_score(a, y_pred)
        f1[i] = f1_score(a, y_pred)
        sizes[i] = len(a)    
        i+=1
    return acc, pre, rec, f1, sizes	
				
   
def clfScores(clf, X, y):
    '''
    calculets the scores for the predictability of clf over the set X y
    Parameters
    -----------
    clf :  classifier
    X : feature matrix
    y : targets numpy array of integers
    Returns
    --------
    s
    R : recall [array]
    P : presicion [array]
    F1 : [array]
    '''
    y_pred = clf.predict(X)
    s = np.sum(y == y_pred)/(1.*len(y)) #clf.score(X, y)
    cM = confusion_matrix(y, y_pred, labels=clf.classes_)
    P = cM.diagonal()/(np.sum(cM, axis=0)*1.)
    R = cM.diagonal()/(np.sum(cM, axis=1)*1.)
    F1 = [2.*R[i]*P[i]/(P[i]+R[i]) for i in range(len(P))]	
    return(s, P, R, F1)
    
def printClfScores( fileN, clf, X, y, l0):
    '''
    prints the scores of the classifier (clf) over the set X y
    Parameters:
    -----------
    fileN : file to wich we are going to append the socres
    Clf :  classifier (object)
    X : feature matrix ( m_instances x n_features)
    y : groud thruth (in the bases of the classifier)
    l0 :  first line, identifier of the set
    Return:
    -------
    S : accuracy
    '''
    S, P, R, F1 = clfScores(clf, X, y)
    if fileN:
        with open(fileN, 'a') as f:
            f.write('%s\n'%l0)
            f.write('S = %f\n'%S)
            f.write('P = %s\n'%', '.join(str(item) for item in P))
            f.write('R = %s\n'%', '.join(str(item) for item in R))
            f.write('F1 = %s\n'%', '.join(str(item) for item in F1))
            return S
    else:
        return S
			
def printIterClfScores( fileN, clf, X, y, c0, comments=None, commtLi='#'):
    '''
    csv file with the scores of the clf
    Parameters
    -------------
    < c0 :  first column
    < comments :  comment string
    < coomtLi :  symbol at the start of a comment line
    '''
    ## write comments
    if isinstance(comments, str):
        comments = '#' + comments.replace('\n', '\n' + commtLi) # add '#' at the biging of the lines
        if not os.path.isfile(fileN): open(fileN, 'w').close() # touch
        if os.stat(fileN).st_size == 0:  # write comments if empty 
            with open(fileN, 'w') as f:
                f.write(comments+'\n')
                
    ## write scores
    S, P, R, F1 = clfScores( clf, X, y)			
    with open(fileN, 'a') as g:
		g.write("%s, %s, %s, %s, %s\n"%(c0, S, 
				', '.join(str(item) for item in P), 
				', '.join(str(item) for item in R), 
				', '.join(str(item) for item in F1) ) )

### confusion matrix

def plConfusionMatrix(cM, labels, outFig='', figsize=None):
    '''
    plots confusion matrix
    cM : confusion matrix
    labels : class labels 
            le.inverse_transform(clf.classes_)
    outFig : name where to save fig
    '''
    # myML.plConfusionMatrix(cM, labels, outFig='', figsize=None)
        
    fig, ax = plt.subplots(figsize=figsize)#(5, 5))
    ax.imshow(cM, cmap=plt.cm.Blues, alpha=0.3, interpolation='nearest')
    
    r,c = np.shape(cM)
    
    ## display numbers in the matrix
    for i in range(r):
        for j in range(c):
            ax.text(x=j, y=i, s=cM[i, j], va='center', ha='center')
    
    ## ticks labels
    ax.set_xticks(range(c))        
    ax.set_xticklabels(labels,rotation=90)
    ax.set_yticks(range(r))        
    ax.set_yticklabels(labels)#,rotation=90)
    ## axis labels
    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')
    
    if outFig: fig.savefig(outFig)    
    
### learnig curve

def plLearningCurve(clf, X, y, samples_arr=None, cv=10, n_jobs=1, outFig='',
                    y_min = 0.8, y_max = 1.1, figsize=None):
                        
    '''plots the learning curve using learning_curve
    Retunrs:
    train_sizes, train_scores, test_scores
    '''
    
    if samples_arr is None: samples_arr=np.linspace(0.1, 1.0, 10)
    
    train_sizes, train_scores, test_scores =\
                learning_curve(estimator=clf, 
                X=X, 
                y=y, 
                train_sizes=samples_arr, 
                cv=cv,
                n_jobs=n_jobs)

    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots( figsize=figsize)

    ax.plot(train_sizes, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')
         
    ax.fill_between(train_sizes, 
                 train_mean + train_std,
                 train_mean - train_std, 
                 alpha=0.15, color='blue')

    ax.plot(train_sizes, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

    ax.fill_between(train_sizes, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')   
    
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([y_min, 1.0])
    plt.tight_layout()
    if outFig: fig.savefig(outFig, dpi=300)            
    return train_sizes, train_scores, test_scores


class clfScoresO():
    def __init__(self, clf, X, y):
        self.clf = clf
        self.X = X
        self.y = y
        self.classes = self.clf.classes_
        self.y_pred = clf.predict(X)
        self.cM = confusion_matrix(self.y, self.y_pred, labels=self.classes)
    
        self.accuracy = np.sum(self.y == self.y_pred)/(1.*len(self.y))
        self.pre = self.cM.diagonal()/(np.sum(self.cM, axis=0)*1.) #for each class
        self.recc = self.cM.diagonal()/(np.sum(self.cM, axis=1)*1.)#for each class
        P = self.pre
        R = self.recc
        self.f1 = np.array([2.*R[i]*P[i]/(P[i]+R[i]) for i in range(len(P))])
        
    def scores2str(self, fln=3, dgt=5):
        '''
        returns a string with the score of a classifier
        Parameters:
        -----------
        fln : float point precission
        dgt : space digits for printing format
        Return:
        -------
        outS : a string with the acc, pre, recc and f1 scores for each class
                <acc>   <pre_1> <recc_1> <f1_1>   <pre_2> <recc_2> <f1_2> ...
        '''
        outS='{1:.{0}}'.format(fln, self.accuracy)
        for i in range(len(self.classes)):
            outS +=  "\t{2:{1}.{0}} {3:{1}.{0}} {4:{1}.{0}}".format(fln, dgt, self.pre[i], self.recc[i], self.f1[i])

        return outS    
        
    def plConfusionMatrix(self, labels, outFig='', figsize=None):
        
        plConfusionMatrix(self.cM, labels, outFig=outFig, figsize=figsize)


               
#########################################################

                
   


class dataObj:
    """
    ### DON'T USE ME
    THIS CLASS WILL BE REPLAZED BY 
    ---------------> datX
    ---------------> datXy
    use this two instead
    bounds feature matrix, targets with other common preprocessing 
    functionalities
    < X : the data matrix  ( # instnces X #features )
    < attrNames : labels array ( # instances X 1 )
    annotated, tells if the data set should be trated as an annotated (True)
        containing the ground truth or not (False)
    """

    def __init__( self, X, attrNames, target=None, std_features=False, datStr=''):       
        #print("TEST", arffFile)
        
        self.attrNames = attrNames # features + output
        self.X = X
        self.datStr=datStr
        self.m_instances, self.n_attr = np.shape(self.X)
        #print( len(self.attrNames), self.n_attr)# '# targets must match n'
        assert len(self.attrNames) == self.n_attr, "# targets must match n"
    
        ##### Inicilizes the basic features
        if target is None: # un-annotated data
            self.y_names = np.zeros(self.m_instances)
            #np.empty(self.m_instances)*np.nan # target
            print("Unlabeled data set")                   
        else: # annotated data set  
            assert len(target) == self.m_instances, '# targets must match m'
            self.y_names = target # target
            print("Annotated data set:", set(self.y_names))

        ### ALL data
        # data matrix
        self.X = standarize(self.X) if std_features else self.X
               
        ### y-labels: nom --> num
        self.__le = preprocessing.LabelEncoder()
        self.y = np.array(self.__le.fit_transform(self.y_names))
        self.y_set = list(set(self.y_names))
        self.y_target_dict = dict(zip(self.tNames2target(self.y_set), self.y_set))
                  
        
    def new_feature(self, newFeatureName, arr):
        'updates the dataframe with a new feature column'
        assert len(arr) == self.m_instances, "bad dimensions"
        self.feature_names.append(newFeatureName)
        arr = arr.reshape( len(arr), 1)
        self.X = np.hstack(self.X, arr)
        
    def new_instances(self, newInstances):
        '''updates the dataframe with a new insatnces
        newInstances : array with the new instances
        '''
        assert np.shape(newInstances)[1] == self.n_attr, "bad dimensions"
        self.X = np.vstack(self.X, newInstances)    
        
    def visualize(self, outPl='', plTitle=''):
        #print("TEST", outPl )
        plXy(self.X.T, self.y, outFig=outPl, plTitle=plTitle)
        
    def nom2num(self, labEncoder, nom):
        return labEncoder.inverse_transform(nom)
    
    def target2tNames(self, y_num):
        '''target (num) --> target name'''
        return self.nom2num(self.__le, y_num)
    
    def tNames2target(self, y_name):
        '''target name (nominal) --> target (numeric)'''
        return self.__le.transform(y_name)
        
    def targetFrequencies(self):
        fr = np.bincount(self.y)
        lnum = np.arange(len(fr))
        return( zip( self.target2tNames(lnum), fr) )
        #return self.y_names.value_counts()
        
    def targetNumDict(self):
        '''> target dict { num : label }'''
        y_numSet = list(set(self.y))
        return dict(zip(y_numSet, self.target2tNames(y_numSet)))

    
class arff2dataFrame(dataObj):
    """
    loads an arff file into a pandasDataFrame
    > the data matrix  ( #instnces X #features )
    > labels array ( #instances X 1 )
    annotated, tells if the data set should be trated as an annotated (True)
        containing the ground truth or not (False)
    """

    def __init__(self, arffFile, annotated=True, std_features=False):
        
        self.__dat, self.__metDat = arff.loadarff(open(arffFile))
        attrNames = self.__metDat.names() # features + output
        self.df = self.get_df()
        X = self.get_X()
        y_names = None
        
        ##### Inicilizes the basic features
        if annotated: # annotated data set            
            y_names = self.df[self.attrNames[-1]] # target
        

        dataObj.__init__(self, X, attrNames, target=y_names,
                         std_features=std_features)

                  
    def get_df(self):
        return pd.DataFrame(data=self.__dat, columns=self.attrNames)
    
    def get_X(self):
        '''
        defines the feature data matrix from the dataFrame
        '''
        return np.array(self.df[self.feature_names])
            
        
class readAudioArff(arff2dataFrame):
    
    def __init__(self, arffFile, annotated=True, std_features=False):
        self.arffFi = arffFile
        arff2dataFrame.__init__(self, self.arffFi, annotated=annotated, std_features=std_features)
    
    def readCommentLines(self):
        commentsDict = {}
        with open(self.arffFi, 'r') as f:
            for line in f:
                if line[0] == '%':
                    #print(line)
                    liLi = line.strip().split(' ')
                    #print(liLi)
                    if not liLi[1] in commentsDict.keys(): 
                        print("creating key ", liLi[1])                        
                        commentsDict[liLi[1]]=[]
                    commentsDict[liLi[1]].append( liLi[2])
        return commentsDict 
        
    def readFilenames(self, filename = 'filename'):
        return self.readCommentLines()[filename]
                
            
class whaleSoundDetector():
    """
    bounds feature matrix, targets with other common preprocessing 
    functionalities
    < X : the data matrix  ( # instances X # features )
    < attrNames : labels array ( # instances X 1 )
    annotated, tells if the data set should be treated as an annotated (True)
        containing the ground truth or not (False)
        
    """

    def __init__(self, genre_list=None, model_path=''):       
        '''
        genre classification object
        Parameters
        ----------
        genre_list : list with music genres
        model_path : 'saved_model/model_ceps.pkl'
        '''
        print("genre classification model created")        
        self.model_path = model_path if os.path.exists(model_path) else None
        self.genre_list = genre_list if isinstance(genre_list, list) else None
        self.genre_list_ix = range(len(self.genre_list)) if isinstance(genre_list, list) else None
        self.clf = joblib.load(self.model_path) if os.path.exists(model_path) else None
        self.outFile=None
        self.cm=None
        

    
    ### DATA
    
    def readCeps(self, genre_list, base_dir):
        '''
        Reads ceps and labels into numpy arrays
        Parameters
        ----------
        base_dir : dir with the ceps files (*.npy)
        '''
        X, y = read_ceps(genre_list, base_dir = base_dir)
        return X, y
    
    ### MODEL    
        
    def trainModel(self, genre_list, train_dir, outModel, ROCplsName='ceps', plIO=True):
        '''
        trains new model form data in "train_dir"
        Parameters
        ----------
        genre_list : new genre list
        train_dir :
        outModel : path to the new model
        ROCplsName : 
        plIO :  ROC curves switch
        '''        
        X, y = read_ceps(genre_list, base_dir = train_dir)
        modelPath, train_avg, test_avg, self.cms = train_model(X, y, ROCplsName, plot=plIO, outModelName=outModel)
        ### confusion matrix
        cm_avg = np.mean(self.cms, axis=0)
        cm_norm = cm_avg / np.sum(cm_avg, axis=0)
        self.cm = cm_norm
        return modelPath
        
    def updateModel(self, path2model, genre_list):
        '''
        extract labels form self.model
        '''
        self.model_path = path2model # path to new model
        self.clf = joblib.load(self.model_path)   
        self.genre_list = genre_list
        self.genre_list_ix = range(len(self.genre_list))

    
    def trainAndUpdateModel(self, genre_list, train_dir, outModel):
        '''
        trains new model and updates parameters
        Parameters
        ----------
        genre_list : list of music genres
        train_dir : dir with the training ceps (*ceps.npi)
        outModel :  path where to save the trained model (*.npy)
        ''' 
        self.updateModel(self.trainModel(genre_list, train_dir, outModel), genre_list)
        
    def clfMetrics(self, X_test, y_test):
        '''
        in case we loaded the clf, we can estimate the confusion matrix parameters giving a 
        test data X, y
        Parameters
        ----------
        X :  ceps matrix
        y :  labels
        Return
        ------
        self.cm : confusion matrix
        '''
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        return cm
        
    
    def plConfusionMatrix(self, outFig, cm, genreListLabels=None):
        '''
        Plot confusion matrix
        PARAMETERS
        ----------
        cm : confusion matrix
        '''
    
        if genreListLabels == None : genreListLabels = self.genre_list
                
        assert cm is not None, 'no cms, use clfMetrics to define one'        
        
        plot_confusion_matrix(cm, genre_list_labels = genreListLabels,
                              name="ceps",title="CEPS classifier - Confusion matrix",
                              outFig=outFig)
    

        
    ### PREDICT    
    
    def genreProbs(self, wav_file):
        X, y = read_ceps_test( create_ceps_test(wav_file)+".npy")
        probs = self.clf.predict_proba(X)[0]
        return probs  
    
    def predictGenre(self, wav_file):
        probs = self.genreProbs(wav_file)
        max_prob = max(probs)
        for i,j in enumerate(probs):
            if probs[i] == max_prob:
                max_prob_index=i
        return self.genre_list[ max_prob_index]
    
    def musicGenrePredictions(self, wav_file):
        probs = self.genreProbs(wav_file)
        max_prob = max(probs)
        for i,j in enumerate(probs):
            if probs[i] == max_prob:
                max_prob_index=i
        return probs, max_prob_index     
  
    
    def printHeadderComment(self, outFile):
        self.outFile = outFile
        with open(outFile, 'a') as f:
            f.write("#%s\n#%s\n"%(time.strftime("%c"), "\t".join(self.genre_list)))
    
    def printGenrePredictions(self, wav_file, outFile=None):
        '''
        for one file
        '''
        probs, max_prob_index = self.musicGenrePredictions(wav_file)
        if outFile == None: outFile=self.outFile
        with open(outFile, 'a') as f:
            f.write("#%s\n%s\t%s\n"%(wav_file,
                                    "\t".join([str('%.2f'%p) for p in probs]), 
                                    self.genre_list[max_prob_index] ))
            
    def printGenrePredictionsDir(self, wavDir, outFile=None):
        '''
        for all the wavs in a dir
        '''
        for path, dirs, files in os.walk(wavDir):
            for fi in files:
                if fi.endswith('wav'):
                    test_file = os.path.join(path, fi)
                    print( "\n###", test_file)
                    self.printGenrePredictions(test_file)
        

def gridSearch_clfStr(best_params):
    s=''
    for a, b in best_params.items():
        s += str(a)+str(b)+'-'
    return a
