#!/usr/mprg/bin/python

from __future__ import print_function, division

import numpy as np
import scipy as sp
import scipy.stats

#import pylab as pl
#import sys
#import time 
#import itertools as it
import scipy.stats as st

#import sequencesO_beta as seqs
#sys.path.append('/home/florencia/whales/scripts/')
#import matrixTools as mt 

"""
    Module for doing statistics
    florencia @ 16.05.14

"""

#################################################################################
##############################    FUNCTIONS    ##################################
#################################################################################

def mean_confidence_interval(data, confidence=0.95):
    """
    Confidence intervals
    returns:
    > mu, the mean 
    > h, the distance to the of the ci to mu
    Assuming a students t-distribution
    """
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def inORout(x, mu, h):
    if (x < mu-h or x > mu+h):
        print( "is out")
        return 1
    else:
        return 0

def tabularChiSquare(p, df):
    """
    returns the tabular value of a chi-square distribution with
    df - degrees of freedom
    p - p-value (\alpha = p/100)
    
    by default the st package measures the proportion under a curve (like chi-sq)
    form left to right which is the most natural. However, the common practice
    when doing a chi-square statistics is to take the proportion of the area
    from right to left, tht'a why we difine this function.
    """
    return st.chi2._ppf(1 - p, df)
    
def chiSquare(O, E):
    '''
    chisquare test of an observed frquencies (O)
    against the expected E assuming H0 is true. 
    the same as st.shisquare()
    > chi-square
    > p-alue corresponding to the computed chi-square
    '''
    assert(len(O) == len(E))
    assert np.all(O > 4), "less than five occurences in this class %s"%np.min(O)
    assert np.all(E > 4), "less than five occurences in this class %s"%np.min(E)
    df = len(O)-1
    Xsquare = np.sum((O-E)**2/E)
    return Xsquare, st.chisqprob(Xsquare, df)
    
def Gstatistics(O, E):
    '''
    chisquare test of an observed frquencies (O)
    against the expected E assuming H0 is true. 
    > chi-square
    > p-alue corresponding to the computed chi-square
    '''
    assert(len(O) == len(E))
    assert np.all(O > 4), "less than five occurences in this class %s"%np.min(O)
    assert np.all(E > 4), "less than five occurences in this class %s"%np.min(E)
    df = len(O)-1
    #print(O*np.log(O/E))
    G = 2*np.sum(O*np.log(O/E))
    return G, st.chisqprob(G, df)
    
    
    
    
    
def testDiffProportions(p1, p2, n1, n2, pcValue=0.9, test='two'):
    """
    z-test for the difference of proportions
    tests the null hypothesis
    'two tailed' H0: p1 = p2, H1: 
    'right tail' H0: p1 - p2 , H1: p1 > p2
    'left tail' H0: p1 < p2
    where p1 and p2 are two proportions, random variables normaly distruibuted.

    Returns
    --------
    y[0]: 1 reject H0, -1 cannot reject H0
    y[1]: z- value
    y[2]: z-critical value correspondent to the given pc-avlue
    """
    assert(test=='two') # TODO: EXTEND TO RIGHT AND LEFT TESTS!
    assert(np.logical_and( p1 <= 1, p2 <= 1))
    
    if(not np.logical_and( n1 >= 30, n2 >= 30)):
         #print("sample size too small n1 = %d, n2 = %d"%(n1, n2))
         return(np.nan, np.nan, np.nan)
   
    else:
        #print(n1, n2)
        zc = {"two": 0.5 + 1.0*(1.0 - pcValue)/2.0, "right": pcValue, "left" : 1-pcValue}
        p = (1.0 * p1 * n1 + 1.0 * p2 * n2) / (n1 + n2)
        s = np.sqrt(p * (1 - p)*( 1.0 / n1 + 1.0 / n2))
        z = (p1 - p2) / s
        zc = st.norm.ppf(pcValue)
        if( np.abs(z) > np.abs(zc)):
            #print("p1 and p2 are different\nREJECT H0!")
            return(1, z, zc)
        else:
            #print("H0 cannot be rejected!")
            return(-1, z, zc)
            
            
            
def elementwiseDiffPropTestXY(X, Y, min_counts=5, pcValue=0.9999):
    """elementwise diff of proportions test between X and Y, H0: X=Y
    Parameters
    ----------
    X: observed frequencies [2-dim-numpy array]
    Y: expected frequencies [2-dim-numpy array]
    min_counts: min number of counts used to compute the 
        proportion of the class in matter
    pcValue: critical p-value
    Returns
    -------
        {-1, 0, 1} - numpy array with the outcome of H0
        1 reject, -1 cannot reject H0, 0 cannot apply test
    """
    XYtest = np.full(np.shape(X), 0)    
    nr, nc = np.shape(X)
    nx = X.sum(axis = 1)*1.
    ny = Y.sum(axis = 1)*1.
    
    for r in range(nr)[:]:
        for c in range(nc)[:]:
            if X[r,c] >= min_counts:
                px = 1.0*X[r,c]/nx[r]
                py = 1.0*Y[r,c]/ny[r]
                XYtest[r,c] = testDiffProportions(px, py, nx[r], ny[r], 
                                                    pcValue=pcValue)[0]
            else:
                XYtest[r,c] = np.NaN
                
    return XYtest
                


# continuous distributions

def KSsimilarity(feature_arr):
    """~similarity between sets of continuous distributions given as rows
    of feature_arr (2darray)
    Similarity is measured as the p-values of the KS-test
    p=1 distributions were drawn from the same pdf,
    the closes p is to zero the more different are the distributions"""
    
    dist = np.zeros((len(feature_arr), len(feature_arr)))
    for i in np.arange(len(feature_arr)):
        for j in np.arange(i+1, len(feature_arr)): # np.arange(len(feature_arr)):
            #print(i, j)
            dist[i,j] = st.ks_2samp(feature_arr[i], feature_arr[j])[1]
    return dist