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
##############################    FUNCTOINS    ##################################
#################################################################################


####    preprocessing    ##################################




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



X = np.arange()

datO = myML.dataXy_names_test(X, y_labs)



class dataXy_names_test(dataX):
    """
    features object -- annotated data
    Parameters:
    ------------
        X : the data matrix  ( # instnces X #features )
            or a tuple of data matrixes (X1, X2, ...)
        y_labels : array with the names of the labels ( # instances (m))
        attrNames : labels array ( # features X 1 )
    """
    
    def __init__(self, X=None, y_labels=None, attrNames=None, datStr=''):
        self.X = self.y = None
        dataX.__init__(self, X=X, attrNames=None, datStr='')
        self.load_y(y_labels) # self.y_labels
        
    def load_y(self, labels):
        '''loads and updates y_labels and the label encoder
        > self.y_labels
        > self.y
        '''
        if isinstance(labels, tuple) : # load more data
            self.y_labels = np.concatenate(labels)

        elif labels is None: # nothing to load
            self.y_labels = None
        else: # load for the first time
            self.y_labels = labels
    
    def addInstances(self, new_X, new_y_labels):
        '''adds annotated instances
        new_X : feature matrix
                    np.array (m, n)
        new_y_labels : targets
                    np.array (m,)
        '''
        m, n = np.shape(new_X) # m (instances) x n (features)
        ## check dimensions
        if self.X is None: # first time we load data
            self.load_X(new_X)
            self.load_y(new_y_labels)
        else: # stack data
            assert(self.n_attr is None or n == self.n_attr), "different feature spaces {} =/= {}".format(n, self.n_attr)
            self.load_X((self.X, new_X))
            assert(m == len(new_y_labels)),  "inconsistent labeling {} =/= {}".format(m, len(new_y_labels))
            self.load_y((self.y_labels, new_y_labels))
        
    def checkDimesions(X, y_labels):
        if self.m_instances == len(y_labels):
            return True
        else:
            print("invalid data {} =/= {}".format(self.m_instances, len(y)))
            self.X = np.nan
            self.y = np.nan
            return False

        
    def targetFrequencies(self):
        return( dict(Counter(self.y_labels) ))
        
    def filterInstances(self, labelSet, A=None, a=None, nominal=True ):
        '''
        returns the instances with labels in the labelSet
        Parameters
        ----------        
        labelSet : list of labels
        A : feature matrix
        b : target vector (nominal)
        nominal : if True - nominal targets will be retunrned
                    otherwise numeric labels are returned
        Returns
        -------
        A_filt :  filtered A
        a_filt : filtered a
        '''
        if A is None : 
            #print("TEST ------ default")
            A = self.X
            a = self.y_labels
        selector = np.in1d(a, labelSet)
        if nominal is True:
            return( A[selector, :], a[selector])
        else: # return numeric targets
            return( A[selector, :], self.nom2num(a[selector]) )      