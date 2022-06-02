"""
Random classification parameter tuning

Author: Abhinav Gupta (Created: 10 Feb 2022)

"""

import numpy as np
import sklearn.ensemble
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterSampler

# RF parameter tuning using cross validation
def RFRegParamTuningCV(x,y):

    # random forest parameter tuning using randomized search cross-validation
    n_estimators = [50]
    max_features = ['auto', 'sqrt', 0.30, 0.35, 0.40]
    max_depth = [int(x) for x in np.linspace(start=100, stop = 250, num = 50)]
    #max_depth.append(None)
    min_samples_leaf = [2, 4, 6, 8, 10]

    random_grid = {'n_estimators' : n_estimators,
                    'max_features' : max_features,
                    'max_depth' : max_depth,
                    'min_samples_leaf' : min_samples_leaf}

    RF = sklearn.ensemble.RandomForestRegressor(oob_score = True)

    RF_rand = RandomizedSearchCV(estimator = RF, param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = -1)
    RF_rand.fit(x, y)
    best_params = RF_rand.best_params_
    """
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
    """

    # RF using best parameters
    n_estimators = best_params['n_estimators']
    max_features = best_params['max_features']
    max_depth = best_params['max_depth']
    min_samples_leaf = best_params['min_samples_leaf']

    return n_estimators, max_features, max_depth, min_samples_leaf


# RF parameter tuning with a separate validation set 
def RFRegParamTuningV(xtrain,ytrain, xval, yval):

    # random forest parameter tuning using randomized search cross-validation
    n_estimators = [50]
    max_features = ['auto', 'sqrt', 0.30, 0.35, 0.40]
    max_depth = [int(x) for x in np.linspace(start=100, stop = 250, num = 7)]
    #max_depth.append(None)
    min_samples_leaf = [2, 4, 6, 8, 10]

    random_grid = {'n_estimators' : n_estimators,
                    'max_features' : max_features,
                    'max_depth' : max_depth,
                    'min_samples_leaf' : min_samples_leaf}

    param_list = list(ParameterSampler(random_grid, n_iter = 50))
    mse = []
    for param_set in param_list:
        RF = sklearn.ensemble.RandomForestRegressor(n_estimators = param_set['n_estimators'], min_samples_leaf  = param_set['min_samples_leaf'], max_features = param_set['max_features'], max_depth = param_set['max_depth'], oob_score = True, n_jobs = -1, min_impurity_decrease = 0, ccp_alpha = 0)
        RF.fit(xtrain, ytrain)
        yest = RF.predict(xval)      
        mse.append(np.sum((yest - yval)**2))
    mse = np.array(mse)
    ind = np.nonzero(mse == np.min(mse))
    ind = ind[0][0]

    best_params = param_list[ind]

    n_estimators = best_params['n_estimators']
    max_features = best_params['max_features']
    max_depth = best_params['max_depth']
    min_samples_leaf = best_params['min_samples_leaf']

    return n_estimators, max_features, max_depth, min_samples_leaf