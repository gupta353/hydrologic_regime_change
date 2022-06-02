"""
Random classification parameter tuning

Author: Abhinav Gupta (Created: 10 Feb 2022)

"""

import numpy as np
import sklearn.ensemble
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

def RFClassParamTuning(x,y):

    # random forest parameter tuning using randomized search cross-validation
    n_estimators = [int(x1) for x1 in np.linspace(start=200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt', 0.33]
    max_depth = [int(x) for x in np.linspace(start=20, stop = 200, num = 9)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [4,8,12]

    random_grid = {'n_estimators' : n_estimators,
                    'max_features' : max_features,
                    'max_depth' : max_depth,
                    'min_samples_split' : min_samples_split,
                    'min_samples_leaf' : min_samples_leaf}

    RF = sklearn.ensemble.RandomForestClassifier(oob_score = True)

    RF_rand = RandomizedSearchCV(estimator = RF, param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = -1)
    RF_rand.fit(x, y)
    best_params = RF_rand.best_params_

    # random forest parameter tuning using grid search
    n_estimators = [best_params['n_estimators'], best_params['n_estimators']+100, best_params['n_estimators']+200]
    max_features = ['auto', 'sqrt', 0.33]
    if best_params['max_depth'] != None:
        max_depth = [best_params['max_depth']-10, best_params['max_depth']-5, best_params['max_depth'], best_params['max_depth']+5, best_params['max_depth']+10]
    else:
        max_depth = [None]
    min_samples_split = [best_params['min_samples_split'], best_params['min_samples_split']+1]
    min_samples_leaf = [best_params['min_samples_leaf']-2, best_params['min_samples_leaf'], best_params['min_samples_leaf']+2]

    random_grid = {'n_estimators' : n_estimators,
                    'max_features' : max_features,
                    'max_depth' : max_depth,
                    'min_samples_split' : min_samples_split,
                    'min_samples_leaf' : min_samples_leaf}

    RF_grid = GridSearchCV(estimator = RF, param_grid = random_grid, cv = 3, n_jobs = -1)
    RF_grid.fit(x, y)
    best_params = RF_rand.best_params_

    # RF classification using best parameters
    n_estimators = best_params['n_estimators']
    max_features = best_params['max_features']
    max_depth = best_params['max_depth']
    min_samples_split = best_params['min_samples_split']
    min_samples_leaf = best_params['min_samples_leaf']

    return n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf