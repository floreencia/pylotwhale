# -*- coding: utf-8 -*-
"""
Common parameters for sklearn classifiers
@author: florencia
"""

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import numpy as np


class clfSettings():
    '''class for binding sklearn classifiers with common hyperparameter values'''
    def __init__(self, clf_name, fun, grid_params_di, pipeStep_name='clf'):
        self.clf_name = clf_name
        self.fun = fun
        self.grid_params_di = grid_params_di
        self.pipeStep_name = pipeStep_name


#### SVC linear
pen_range = [ 0.1, 1.0, 10.0, 100]
param_grid_di = {'clf__kernel': ['linear'],
                 'clf__C': pen_range}
                
svc_l =  clfSettings('svc_linear', 
                     SVC(), 
                    param_grid_di)


#### SVC rbf
gamma_range = [ 0.1, 1.0, 10.]
pen_range = [ 0.1, 1.0, 10.0, 100]
param_grid_di = {'clf__C': pen_range,
                'clf__gamma': gamma_range}

svc_rbf =  clfSettings('svc_rbf', SVC(), param_grid_di)


#### SVC rbf w/ probs
gamma_range = [ 0.1, 1.0, 10.]
pen_range = [ 0.1, 1.0, 10.0, 100]
param_grid_di = {'clf__C': pen_range,
                'clf__gamma': gamma_range}

svc_rbf_p =  clfSettings('svc_rbf_p',
                         SVC(probability=True), 
                        param_grid_di)

#### RF
ests_range = [50, 100]
param_grid_di = {"clf__max_depth": [3, None],
                 "clf__bootstrap": [True, False],
                 "clf__n_estimators": ests_range}

random_forest = clfSettings('rf', 
                            RandomForestClassifier(max_depth=None, bootstrap=True), 
                            param_grid_di)
