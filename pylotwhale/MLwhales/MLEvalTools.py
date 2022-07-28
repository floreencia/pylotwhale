from __future__ import print_function, division
import os
import functools
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn import metrics as mt

import pylotwhale.signalProcessing.signalTools_beta as sT
import pylotwhale.signalProcessing.audioFeatures as auf
import pylotwhale.MLwhales.predictionTools as pT

import pylotwhale.MLwhales.MLtools_beta as myML
import pylotwhale.MLwhales.featureExtraction as fex

"""
Tools for evaluating classifiers
- over collections and wavefiles
- print scores to file
- plot confusion matrix
- plot learning curve
"""

############################################################################
#####################   CLASSIFIER EVALUATION    ###########################
############################################################################

#### scoring functions

## f1 score for one class: calls


def classIndex_f1_score(y_true, y_pred, classIndex=1,
                        scoringFunction=mt.f1_score):
    """defines a scoringFunction that is maximal for classIndex
    returns a scoring function from one component (classIndex)
    from a vectorial scoring"""
    return scoringFunction(y_true, y_pred, average=None)[classIndex]


def make_class_f1score_fun(classTag=1, lt=None,
                           scoringFunction=classIndex_f1_score):
    """defines a scoringFuncion that is maximal at className
    if a label transformer is given, className is transformed from numeric
    with nom2num"""
    if isinstance(lt, myML.labelTransformer):  # check whether a labelTransfomer is given
        classTag = lt.nom2num(classTag)
    return functools.partial(scoringFunction, classIndex=classTag)


def getCallScorer(classTag=1, lt=None):
    """returns a scoring function that maximises
    the f1 score for classTag
    Parameters
    ----------
    classTag: str, int
        class id
    lt: myML.labelTransformer
        ehrn lt is given the classTag is transformed with lt.nom2num
    """
    return mt.make_scorer(make_class_f1score_fun(classTag=classTag, lt=lt))


### Evaluate a list of classifiers over a collection


def get_gridSearchresults_str(gs, max_std=0.02):
    gsResults_str = ''
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']
    for i, (mean, std, params) in enumerate(zip(means, stds, gs.cv_results_['params'])):
        if std < max_std:
            pstr = str(params).replace('\n', '')
            gsResults_str += "%d %2.1f (+/-%2.01f) \n# %r\n"% (i, mean*100, std * 2*100, pstr)
    return gsResults_str


def bestCVScoresfromGridSearch(gs):
    '''retrieve CV scores of the best model from a gridsearch object
    Parameters
    ----------
        gs : gridsearch object
    Returns (mu, std) of the bets scores
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


def printScoresFromCollectionFile(feExFun, clf, lt, collFi, out_file, labelsHierarchy):
    """
    clf : classifier
    le : label encoder (object)
    collfi : annotated wav collection (*.txt)
    out_file : out file (*.txt)
    """
    coll = fex.readCols(collFi, colIndexes =(0,1)) #np.loadtxt(collFi, delimiter='\t', dtype='|S')
    printScoresFromCollection(feExFun, clf, lt, coll, out_file, labelsHierarchy)


def getScoresFromWav(wavF, annF, feExFun, clf, lt, labelsHierarchy):
    """
    extracts features and its labels (ground truth) from wavF and annF files
    and compares the clf predictions with the ground truth
    Parameters
    ----------
    wavF: str
    annF: str
    feExFun: callable
    clf : classifier
    le : label encoder (object)
    labelsHierarchy: list
    """

    A, a_names = fex.getXy_fromWavFAnnF(wavF, annF, feExFun, labelsHierarchy, 
                                        filter_classes=lt.classes_)
    a = lt.nom2num(a_names)
    return clfScoresO(clf, A, a)

def printScoresFromCollection(feExFun, clf, lt, coll, out_file, labelsHierarchy):
    """
    clf : classifier
    le : label encoder (object)
    coll: list,
        wav ann collection [(wav_file, ann_file), (wav_file, ann_file), ...]
    out_file: str,
        file where scores will be printed
    """
     #np.loadtxt(collFi, delimiter='\t', dtype='|S')

    for wavF, annF in coll[:]:
        scsO = getScoresFromWav(wavF, annF, feExFun, clf, lt, labelsHierarchy)
        annF_bN = os.path.basename(annF)
        with open(out_file, 'a') as f:
            f.write("{}\t{}\n".format(scsO.scores2str(), annF_bN))


def clfGeneralizability(clf_list, wavAnnCollection, featExtFun,
                        labelEncoder, labelSet=None):
    '''estimates the score of a list of classifiers,
    one score for each wav file'''
    clf_scores = []  # np.zeros(len(clf_list))
    for clf in clf_list:
        acc, pre, rec, f1, size = coll_clf_scores(clf, wavAnnCollection,
                                                  featExtFun,
                                                  labelEncoder=labelEncoder,
                                                  labelSet=labelSet)
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
        a = labelTransformer.nom2num( a_names[mask]) # convert labels to numeric
        A = A0[mask]
        y_pred = clf.predict(A)
        ## scores
        acc[i] = mt.accuracy_score(a, y_pred)
        pre[i]= mt.precision_score(a, y_pred)
        rec[i] = mt.recall_score(a, y_pred)
        f1[i] = mt.f1_score(a, y_pred)
        sizes[i] = len(a)    
        i+=1
    return acc, pre, rec, f1, sizes	
				
   
def clfScores(clf, X, y):
    '''
    calculates the scores for the predictability of clf over the set X y
    Parameters
    -----------
    clf :  classifier
    X : feature matrix
    y : targets numpy array of integers
    Returns
    --------
    s
    R : recall [array]
    P : precision [array]
    F1 : [array]
    '''
    y_pred = clf.predict(X)
    s = np.sum(y == y_pred)/(1.*len(y)) #clf.score(X, y)
    cM = mt.confusion_matrix(y, y_pred, labels=clf.classes_)
    P = cM.diagonal()/(np.sum(cM, axis=0)*1.)
    R = cM.diagonal()/(np.sum(cM, axis=1)*1.)
    F1 = [2.*R[i]*P[i]/(P[i]+R[i]) for i in range(len(P))]	
    return(s, P, R, F1)
    
    
### print scores in file

def print_precision_recall_fscore_support(y_true, y_pred, ofile, labels=None,
                                          sep=", ", strini="", strend=""):
    scores = mt.precision_recall_fscore_support(y_true, y_pred, labels=labels)
    with open(ofile, 'a') as f:
        f.write(strini)
        for sc in scores[:-1]:
            f.write("{}{}".format(sep.join(["{:.2f}".format(item*100) for item in sc]), sep))
        sc = scores[-1]
        f.write(sep.join(["{}".format(item) for item in sc]))
        f.write(strend)

def printWSD2_scores(wavF, true_annF, template_annF, WSD2_clf, WSD2_feExFun, lt,
                    scoreClassLabels, outF, strini="", strend="", m='auto',
                    readSectionsWSD2='default', # for WSD2
                    labelsHierarchy='default'): # for instantiating truth labels

    if labelsHierarchy == 'default':
        labelsHierarchy = ['c']
    # load waveform
    waveForm, fs = sT.wav2waveform(wavF)
    tf = len(waveForm)/fs
    if m == 'auto':
        m = int(len(waveForm)/512)

    # ground thruth
    y_true_names = auf.annotationsFi2instances(true_annF, m, tf,
                                                labelsHierarchy=labelsHierarchy)

    # predictions
    y_pred_names = pT.WSD2predict(wavF, template_annF, WSD2_feExFun, lt, WSD2_clf, m, tf,
                                  readSections=readSectionsWSD2)

    # print P, R, f1, support
    print_precision_recall_fscore_support(y_true_names, y_pred_names, outF,
                                               strini=strini, strend=strend,
                                               labels=scoreClassLabels)
    return outF



def printClfScores( fileN, clf, X, y, l0):
    '''
    prints the scores of the classifier (clf) over the set X y
    Parameters:
    -----------
    fileN : file to which we are going to append the scores
    Clf :  classifier (object)
    X : feature matrix ( m_instances x n_features)
    y : ground truth (in the bases of the classifier)
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
        g.write("%s, %s, %s, %s, %s\n" % (c0, S, ', '.join(str(item) for item in P),
                                          ', '.join(str(item) for item in R),
                                            ', '.join(str(item) for item in F1)))

### confusion matrix

def plConfusionMatrix(cM, labels, outFig='', fontSz=20, figsize=None, 
                      display_nums=True, alpha=0.3, title=None):
    '''
    plots confusion matrix
    cM : confusion matrix
    labels : class labels 
            le.inverse_transform(clf.classes_)
    outFig : name where to save fig
    '''
        
    fig, ax = plt.subplots(figsize=figsize)#(5, 5))
    ax.imshow(cM, cmap=plt.cm.Blues, alpha=alpha, interpolation='nearest')
    
    r,c = np.shape(cM)
    
    ## display numbers in the matrix
    if display_nums:
        for i in range(r):
            for j in range(c):
                ax.text(x=j, y=i, s="{:.3f}".format(cM[i, j]), va='center', ha='center') 
                        #fontsize=fontSz )
    
    ## ticks labels
    ax.set_xticks(range(c))        
    ax.set_xticklabels(labels,rotation=90)
    ax.set_yticks(range(r))        
    ax.set_yticklabels(labels)#,rotation=90)
    ## axis labels
    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')
    if title: ax.set_title(title)
    
    if outFig: fig.savefig(outFig)  

    return fig , ax 
    
### learning curve

def plLearningCurve(clf, X, y, samples_arr=None, cv=10, n_jobs=1, 
                    scoring=None,
                    outFig='',
                    y_min = 0.8, y_max = 1.1, figsize=None):
                        
    '''plots the learning curve using sklearn's learning_curve
    Returns:
    train_sizes, train_scores, test_scores
    '''
    
    if samples_arr is None: samples_arr=np.linspace(0.1, 1.0, 5)
    
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
    plt.legend(loc='center left')
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
        self.cM = mt.confusion_matrix(self.y, self.y_pred, labels=self.classes)
    
        self.accuracy = np.sum(self.y == self.y_pred)/(1.*len(self.y))
        self.pre = self.cM.diagonal()/(np.sum(self.cM, axis=0)*1.) #for each class
        self.recc = self.cM.diagonal()/(np.sum(self.cM, axis=1)*1.)#for each class
        P = self.pre
        R = self.recc
        self.f1 = np.array([2.*R[i]*P[i]/(P[i]+R[i]) for i in range(len(P))])
        
        self.get_clf_report = self.clf_report(self.y, self.y_pred)#, target_names=self.classes)
        
    def scores2str(self, fln=3, dgt=5):
        '''
        returns a string with the score of a classifier
        Parameters:
        -----------
        fln : float point precision
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
        
    def clf_report(self, y_true, y_pred, target_names=None, **kwargs):
        return mt.classification_report(y_true, y_pred, target_names=target_names)

               
#########################################################

      

def gridSearch_clfStr(best_params):
    s=''
    for a, b in best_params.items():
        s += str(a)+str(b)+'-'
    return a
