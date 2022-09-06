"""
This module contains functions to compute hydrological goodness-of-fit metrices

Author: Abhinav Gupta (Created: 6 Jul 2022)
"""


import numpy as np
import random

# function to compute NSE
def computeNSE(obs, pred):
    sse = np.sum((obs - pred)**2)
    sst = np.sum((obs - np.mean(obs))**2)
    nse = 1 - sse/sst
    return nse

# function to compute NSE uisng bootstap sampling
def computeNSEBS(obs, pred):
    counter = list(np.arange(0,len(obs)))
    nse = []
    for bs in range(1000):
        random.seed(bs + 100)
        inds = random.choices(counter, k = len(obs))
        obs_tmp = obs[np.array(inds)]
        pred_tmp = pred[np.array(inds)]
        sse = np.sum((obs_tmp - pred_tmp)**2)
        sst = np.sum((obs_tmp - np.mean(obs_tmp))**2)
        nse.append(1 - sse/sst)
    return nse