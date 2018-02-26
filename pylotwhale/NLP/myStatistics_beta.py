from __future__ import print_function, division

import numpy as np
#import scipy as sp
#import scipy.stats as stats
import nltk
import random
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

#import pylab as pl
#import sys
#import time
#import itertools as it
import scipy.stats as st
import statsmodels as sm

import pylotwhale.NLP.annotations_analyser as aa
import pylotwhale.NLP.ngramO_beta as ngr
from seaborn import xkcd_palette


"""
    Module for doing statistics
    florencia @ 16.05.14

"""


##### RANDOMISATIONS TEST FOR BIGRAMS ######

def teStat_proportions_diff(p1):
    """test statistic for differnce of proportions p1-p2"""
    return p1  # 2*p1-1


def plDist_with_obsValue(i, j, shuffledDistsM, obsM, ax=None, plTitle=None,
                         kwargs_obs=None, **kwargs):
    """plot randomised distribution (distsM) with observable
        kwargs_obs {color: 'r', lw=2.5}
        i=conds
        j=samps"""
    if kwargs_obs is None:
        kwargs_obs = {'color': 'r', 'lw': 2.5}
    #fig, ax =  plt.subplots(figsize=None)
    if plTitle:
        ax.set_title(plTitle)
    ax.hist(shuffledDistsM[:, i, j], **kwargs)
    ax.axvline(obsM[i, j], **kwargs_obs)
    return ax


def repsProportion_from_bigramMtx(M):
    """proportion of repetitions"""
    return np.sum(np.diag(M))/M.sum()


def repsProportion(seq, deg=1):
    Nreps += len(seq[seq[deg:] == seq[:-deg]])


def repsProportion_in_listOfSeqs(liOfSeqs, deg=1):
    """proportion of repetitions in a list of sequeces
    liOfSeqs: list of lists
    """
    Nbigrams = 0
    Nreps = 0
    for seql in liOfSeqs:
        seq = np.array(seql)
        Nreps += np.count_nonzero([seq[deg:] == seq[:-deg]])
        Nbigrams += len(seq)-1
    return Nreps, Nbigrams


#### USING pandas DataFrame, to define the sequences


def shuffleSeries(dataFr, shuffleCol='timeSs'):
    """
    shuffles a series (shuffleCol) from a data frame:
    > data frame
    > name a of the column ti shuffle
    """
    x = dataFr[shuffleCol].values.copy()  # select series
    random.shuffle(x)  # shuffle
    shuffledRecs = dataFr.copy()  # new series
    shuffledRecs[shuffleCol] = x  # set shuffled series
    return shuffledRecs  # data frames and labels


def shuffled_cfd(df, Dtint, label='call', time_param='ici'):
    """returns the conditional frequencies
    of the bigrams in a df after shuffling <label>
    Parameters
    ----------
    df : Pandas.DataFrame
    Dtint : size two list-like
    label : string
        name of the label to randomise
    time_param : string
        name of the time param (Dtint) to define the sequences
    Returns
    -------
    cfd_ns h: nltk.ConditionalFrequencyDist
        counts of the randomised sequences
    """
    sh_df = shuffleSeries(df, shuffleCol=label)  # shuffle the calls
    # define the sequences
    sequences = aa.seqsLi2iniEndSeq(aa.df2listOfSeqs(sh_df, Dt=Dtint, l=label,
                                                     time_param=time_param))
    my_bigrams = nltk.bigrams(sequences)  # detect bigrams
    cfd_nsh = ngr.bigrams2Dict(my_bigrams)  # count bigrams
    return cfd_nsh


def randomisation_test4bigrmas(df_dict, Dtint, obsTest, Nsh, condsLi, sampsLi,
                               label='call', time_param='ici',
                               testStat=teStat_proportions_diff):
    """one sided randomisation test for each bigram conditional probability
        under the null hypothesis H0: testStat_observed < testStat_shuffled
        returns the p-values
    Parameters
    ----------
    df_dict : dict
        dictionary of dataframes (tapes)
    Dt : tuple
        (None, Dt)
    obsTest : ndarray
        observed stat for each bigram
    Nsh : int
    condLi, sampLi : list
        list of conditions and samples
    testStat : callable
    Returns
    -------
    p_values : ndarray
    shuffle_test : ndarray
        shuffled test distributions
    """
    nr, nc = np.shape(obsTest)
    shuffle_tests = np.zeros((Nsh, nr, nc))
    N_values_r = np.zeros_like(obsTest)
    for i in range(Nsh):  # shuffle ith-loop
        cfd_sh = nltk.ConditionalFreqDist()  # initialise cond freq dist.
        for t in df_dict.keys():  # for each tape
            thisdf = df_dict[t]
            cfd_sh += shuffled_cfd(thisdf, Dtint, label=label,
                                   time_param=time_param)  # counts
        Mp_sh, samps, conds = ngr.condFreqDict2condProbMatrix(cfd_sh,
                                                              condsLi, sampsLi)  # normalised matrix
        shTest_i = testStat(Mp_sh)  # compute satat variable
        shuffle_tests[i] = shTest_i  # save distribution for later
        N_values_r[shTest_i >= obsTest] += 1  # test right
        #N_values_l[shTest_i < obsTest] += 1 # test left
    p_r = 1.0*N_values_r/Nsh
    return p_r, shuffle_tests


def randomisation_test_repetitions(df_dict, Dtint, obsTest, Nsh, callsLi,
                                   label='call', time_param='ici',
                                   testStat=repsProportion_from_bigramMtx):
    """randomisation test for repetitions within the interval Dtint
    shuffle tapes within a tape, define Dtint-seqeunces and count repetitions
    Parameters
    ----------
    df_dict: dict
        dictionary of dataframes (tapes)
    Dtint: tuple
        interval for defining sequences, eg. (None, Dt)
    obsTest: float
        observed stat for each bigram
    Nsh: int
    callsLi: list
        list of conditions and samples
    testStat: callable
    Returns
    -------
    p_values: ndarray
    shuffle_test: ndarray
        shuffled test distributions
    """
    N_values = 0
    shuff_dist = np.zeros(Nsh)
    for i in range(Nsh):  # shuffle ith-loop
        cfd_sh = nltk.ConditionalFreqDist()  # initialise cond freq dist
        for t in df_dict.keys():  # for each tape
            thisdf = df_dict[t]
            cfd_sh += shuffled_cfd(thisdf, Dtint, label=label,
                                   time_param=time_param)  # counts in current tape
        Mp_sh, samps, conds = ngr.bigramsDict2countsMatrix(cfd_sh, callsLi, callsLi)
        shTest_i = testStat(Mp_sh)  # compute satat variable
        shuff_dist[i] = shTest_i
        if shTest_i >= obsTest:
            N_values += 1

    p = 1.0*N_values/Nsh
    return p, shuff_dist


#### USING PRE-DEFINED SEQUENCES

def shuffleSequence(seq):
    '''randomises the elements of a sequence'''
    return np.random.shuffle(seq)


def shuffleSeqOfSeqs(seqOfSeqs):
    '''shuffles the elements in each sequence,
    keeping the subsequence structure
    number of sequences and sequence size'''
    shuffled_seqOfSeqs = []
    for s in seqOfSeqs:
        shuffled_seqOfSeqs.append(shuffleSequence(s))
    return shuffled_seqOfSeqs


#### plotting

def pValue_colour(p, pc=0.05):
    '''returns predefined colours depending on the p-value'''
    if p < pc:
        return xkcd_palette(['blue'])[0]
    if (1 - p) < pc:
        return xkcd_palette(['red'])[0]
    else:
        return xkcd_palette(['light grey'])[0]


##### other older functions #####

def mean_confidence_interval(data, confidence=0.95):
    """
    Confidence intervals
    Returns
    > mu, the mean
    > h, the distance to the of the ci to mu
    Assuming a students t-distribution
    """
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t._ppf((1+confidence)/2., n-1)
    return m, h


def inORout(x, mu, h):
    if (x < mu-h or x > mu+h):
        print("is out")
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
    
    
    
### DISTANCES AND COMPARISONS
    
def stat_testDiffProportions(p1, p2, n1, n2, pcValue=0.9, test='two'):
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
    #assert(test == 'two')  # TODO: EXTEND TO RIGHT AND LEFT TESTS!
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

def KSsimilarity(feature_arr, i_diag=1):
    """~similarity matrix between sets of continuous distributions given as rows
    of feature_arr (2darray)
    Similarity is measured as the p-values of the KS-test
    p=1 distributions were drawn from the same pdf,
    the closer p is to zero the more different are the distributions
    returns only the lower traingle, upper traingle is set up to nan
    Parameters
    ----------
    feature_arr: list
    i_diag: int
         deviation from diagonal, = 0 include diagonal
    """

    p = np.zeros((len(feature_arr), len(feature_arr))) + np.nan
    for i in np.arange(len(feature_arr)):
        for j in np.arange(i+i_diag, len(feature_arr)): # np.arange(len(feature_arr)):
            #print(i, j)
            p[i,j] = st.ks_2samp(feature_arr[i], feature_arr[j])[1]
    return p


def KL_div_symm(x, y):
    """returns the symmetric KL-divergence"""
    return st.entropy(x, y) + st.entropy(y, x)

def pairwise_probDists_distance(feature_arr, i_diag=1, dist_fun=KL_div_symm):
    """computes the distance between probability distributions
    Parameters
    ----------
    feature_arr: list
        each element of the list has a probability distribution
    i_diag: int
         deviation from diagonal, = 0 include diagonal
    dist_fun: callable
        distance function
    """

    p = np.zeros((len(feature_arr), len(feature_arr))) + np.nan
    for i in np.arange(len(feature_arr)):
        for j in np.arange(i+i_diag, len(feature_arr)): # np.arange(len(feature_arr)):
            #print(i, j)
            p[i,j] = dist_fun(feature_arr[i], feature_arr[j])
    return p

    
def pairwise_probDists_distance_df(di, dist_fun=KL_div_symm):
    """
    di: dict of nd arrays
        pdfs
    """
    keys = di.keys()
    df = pd.DataFrame(index=keys, columns=keys, dtype=float)

    for i in np.arange(len(keys)):
        for j in np.arange(len(keys)): # np.arange(len(feature_arr)):
            #print(i, j)
            #df[i,j] = dist_fun(feature_arr[i], feature_arr[j])
            df.at[keys[i], keys[j]] = dist_fun(di[keys[i]], di[keys[j]])
            df.at[keys[j], keys[i]] = df.at[keys[i], keys[j]]
            
    return df


def KL_div_joint(x, y, Nsh=10):
    """KL-divergence for assessing the significance of the correlation
    between two variables x and y
    computes the KL-div between sh(xy) and  obs(xy)
    Returns
    -------
        the KL divergence between the observed and a shuffled distribution
        the KL divergence between the two shuffled distributions
    """
    #assert(len(x) == len(y))
    y_sh = np.array(y)
    KL_dist_obs_sh = np.zeros((Nsh))
    KL_dist_sh_sh = np.zeros((Nsh))
    for i in np.arange(Nsh):
        _, _, pdf = joint_pdf(x, y)
        np.random.shuffle(y_sh)
        _, _, pdf_sh1 = joint_pdf(x, y_sh)
        np.random.shuffle(y_sh)
        _, _, pdf_sh2 = joint_pdf(x, y_sh)
        KL_dist_obs_sh[i] = st.entropy(pdf_sh1.ravel(), pdf.ravel())
        KL_dist_sh_sh[i] = st.entropy(pdf_sh1.ravel(), pdf_sh2.ravel())
    return KL_dist_obs_sh, KL_dist_sh_sh

def KL_div_joint_inv(x, y, Nsh=10):
    """KL-divergence for assessing the significance of the correlation
    between two variables x and y
    computes the KL-div between obs(xy) and sh(xy)

    Returns
    -------
        the KL divergence between the observed and a shuffled distribution
        the KL divergence between the two shuffled distributions
    """
    #assert(len(x) == len(y))
    y_sh = np.array(y)
    KL_dist_obs_sh = np.zeros((Nsh))
    KL_dist_sh_sh = np.zeros((Nsh))
    for i in np.arange(Nsh):
        _, _, pdf = joint_pdf(x, y)
        np.random.shuffle(y_sh)
        _, _, pdf_sh1 = joint_pdf(x, y_sh)
        np.random.shuffle(y_sh)
        _, _, pdf_sh2 = joint_pdf(x, y_sh)
        KL_dist_obs_sh[i] = st.entropy(pdf.ravel(), pdf_sh1.ravel())
        KL_dist_sh_sh[i] = st.entropy(pdf_sh1.ravel(), pdf_sh2.ravel())
    return KL_dist_obs_sh, KL_dist_sh_sh



#####

def joint_pdf(x, y, grid_size=100j):
    """joint probability between x and y estimated with a Gaussian kernel
    Parameters
    ----------
    x, y: 1d arrays
    Returns
    -------
    X, Y, KDE(X, Y) : coordinates X, Y, Z"""

    assert len(x) == len(y), 'must be same length'
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    X, Y = np.mgrid[xmin: xmax: grid_size, ymin: ymax: grid_size]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    return X, Y, np.reshape(kernel(positions).T, X.shape)


def CVbw_KDE(x, kde_param_grid=None, **CV_kwargs):
    '''KDE estimating the bw via cross validation
    Parameters
    ----------
    x: 1d numpy array
        data to fit KDE
    kde_param_grid: dict
        GridSearchCV params for KernelDensity()
        eg. kde_param_grid = {'bandwidth': np.logspace(-2, 0, 10)}
    Returns
    -------
    kde: estimator
    '''

    if kde_param_grid is None:
        kde_param_grid = {'bandwidth': np.logspace(-2, 0, 10)}

    grid = GridSearchCV(estimator=KernelDensity(),
                        param_grid=kde_param_grid, **CV_kwargs)
    data = x[:, np.newaxis]
    grid.fit(data)
    return grid.best_estimator_

def get_KDE_CVbw(x, kde_param_grid=None, **CV_kwargs):
    '''get optimal bw'''
    kde = CVbw_KDE(x, kde_param_grid=kde_param_grid, **CV_kwargs)
    return(kde.bandwidth)


def fit_KDE_CVbw(x, supp_range, num=1000, kde_param_grid=None, 
                 **CV_kwargs):  #, bw_range=None):
    '''fit KDE estimating using cross validation to estimate the bandwith'''

    # get model
    kde = CVbw_KDE(x, kde_param_grid=kde_param_grid, **CV_kwargs)

    # generate density from sample
    supp = np.linspace(*supp_range, num=num)[:, np.newaxis]
    y = kde.score_samples(supp)
    return np.exp(y)


def fit_KDE(x, supp_range, num=100, bw='normal_reference', **kwargs):
    """
    Fits KDE to a sample x in the range x_0 to x_f, using a bw selection rule
    Parameters
    ----------
    x: 1 dim numpy array
        sample
    supp: 2-dim tuple
        range of the KDE
    num: number of points
    **kwargs: dict
        eg. gridsize
        see sm.nonparametric.kde.KDEUnivariate.fit()
    addFloat: float
        to avoid the distribution to have zeros, eg. np.nextafter(0,1)
    Return
    ------
    y: 1 dim numpy array
        KDE

    See:
    ----
    fit_KDE_CVbw
    """
    assert(len(supp_range) == 2)
    ## fit model
    kde = sm.nonparametric.kde.KDEUnivariate(x)
    kde.fit(bw=bw, **kwargs)
    ## evaluate
    supp = np.linspace(*supp_range, num=num)
    y0 = kde.evaluate(supp)

    return y0


def deZero(x, n=0, epsilon0=0):
    """shifts an array by the smallest non zero value"""
    epsilon =np.sort(np.array(list(set(x))))[n] # list(set(np.sort(x[x > 0]))[n]
    x += epsilon + epsilon0
    return x


