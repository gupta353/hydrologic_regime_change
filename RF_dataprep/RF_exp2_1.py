"""
This script carries out experiment B: Using 1990-2000 water years data from both A and B for training 

Author: Abhinav Gupta (Created: 27 May 2022)

"""

import numpy as np
import os
import datetime
import sklearn.ensemble
import matplotlib.pyplot as plt
import RF_ParamTuning
import time
import pickle
from random import choices

# function to read dynamic data
def readDynamic(filename):
    fid = open(filename)
    data = fid.readlines()
    fid.close()
    dyn = []
    for rind in range(1,len(data)):
        data_tmp = data[rind].split()
        yy = int(data_tmp[0])
        mm = int(data_tmp[1])
        dd = int(data_tmp[2])
        datenum = datetime.date(yy, mm, dd).toordinal()
        dyn.append([datenum] + [float(x) for x in data_tmp[3:]])

    return dyn

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
        inds = choices(counter, k = len(obs))
        obs_tmp = obs[np.array(inds)]
        pred_tmp = pred[np.array(inds)]
        sse = np.sum((obs_tmp - pred_tmp)**2)
        sst = np.sum((obs_tmp - np.mean(obs_tmp))**2)
        nse.append(1 - sse/sst)
    return nse

# function to remove rows containing NaNs
def removeNaN(Y):
    X = Y.copy()
    del_ind = []
    for ind in range(0,len(X)):
        X_tmp = X[ind,:]
        if np.all(np.isnan(X_tmp) == False):
            a = 1
        else:
            del_ind.append(ind)
    if len(del_ind) != 0:
        Xp = np.delete(X, np.array(del_ind), axis = 0)
    else:
        Xp = X

    return Xp

####################################################################################################

# read list of the basins which have undergone a regime change
direc = 'D:/Research/non_staitionarity/codes/results/data_split'
fname = 'set_A.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data  = fid.readlines()
fid.close()
basins = []
psd = []
ptrend = []
pdiff = []
for rind in range(1, len(data)):
    data_tmp = data[rind].split()
    basins.append(data_tmp[0])
    psd.append([float(x) for x in data_tmp[1:8]])
    ptrend.append([float(x) for x in data_tmp[8:15]])
    pdiff.append([float(x) for x in data_tmp[15:-2]])
psd = np.array(psd)
ptrend = np.array(ptrend)
pdiff = np.array(pdiff)

# keep the basins corresponding to statistically significant change in F4 (less than 2 weeks timescales component)
inds = np.nonzero((ptrend[:,4]<=0.05) & (pdiff[:,4]<=0.05))
inds = inds[0]
A = [basins[i] for i in inds]

# read the list of 531 basins
direc = 'D:/Research/non_staitionarity/data/531_basins'
fname = 'basin_list_zero_appended.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
basins_531 = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    basins_531.append(data_tmp[0])

# remove the data from basins that are not contained in basins_531
A_final = []
for ind in range(0,len(A)):
    basin_tmp = A[ind]
    ii = [i for i in range(0, len(basins_531)) if basins_531[i] == basin_tmp]
    if len(ii) != 0:
        A_final.append(basin_tmp)
##################################################################################################
## read the list of basisn which have not undergone a regime change
# basins that are contained in 531 basin but not in set A_final or 'basins'
B = []
for basin_tmp in basins_531:
    if basin_tmp not in basins:
        B.append(basin_tmp)
##################################################################################################

# read static data for all basins
direc_static = 'D:/Research/non_staitionarity/data/RF_static_data'
fname = 'RF_static.txt'
filename = direc_static + '/' + fname
fid = open(filename)
data = fid.readlines()
fid.close()
static_data = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    static_data.append([data_tmp[0]] + [float(x) for x in data_tmp[1:]])
##################################################################################################

# prepare data for RF regression
direc_dynamic = 'D:/Research/non_staitionarity/data/RF_dynamic_data_2'
train_datenum_1 = datetime.date(1991,10,1).toordinal()
train_datenum_2 = datetime.date(2001,9,30).toordinal()
val_datenum_1 = datetime.date(1985,11,30).toordinal()
val_datenum_2 = datetime.date(1991,9,30).toordinal()
test_datenum_1 = datetime.date(2001,10,1).toordinal()

eps = 0.1 # to normalize MSE

train_data = []
val_data = []
test_data=[]
sd_vals = []
basins_used = []
basins_used_A = []
test_data_A = []
for basin in A_final + B:
    # read dynamic data
    fname = basin + '_met_dynamic_2.txt'
    filename = direc_dynamic + '/' + fname
    if os.path.exists(filename):
        
        basins_used.append(basin)
        met = readDynamic(filename)
    
        # join static and met data
        ind = [i for i in range(0, len(static_data)) if static_data[i][0] == basin]
        ind = ind[0]
        for wind in range(0,len(met)):
            met[wind] = met[wind] + static_data[ind][1:]
        met = np.array(met)

        # divide the data into training, validation, and test data
        ind_train_1 = np.nonzero(met[:,0] == train_datenum_1)
        ind_train_1 = ind_train_1[0][0]
        ind_train_2 = np.nonzero(met[:,0] == train_datenum_2)
        ind_train_2 = ind_train_2[0][0]
        ind_val_1 = np.nonzero(met[:,0] == val_datenum_1)
        ind_val_1 = ind_val_1[0][0]
        ind_val_2 = np.nonzero(met[:,0] == val_datenum_2)
        ind_val_2 = ind_val_2[0][0]
        ind_test_1 = np.nonzero(met[:,0] == test_datenum_1)
        ind_test_1 = ind_test_1[0][0]

        train_tmp = met[ind_train_1:ind_train_2+1,:]
        val_tmp = met[ind_val_1:ind_val_2+1,:]
        test_tmp = met[ind_test_1:,:]

        # compute standard deviation of streamflow in training set
        sd_data = np.array(train_tmp)[:,1]
        sd = np.nanstd(sd_data) + eps
        sd_vals.append(sd)

        # add standard deviation as the last predictor (it should be removed before model training)
        train_tmp = np.concatenate((train_tmp, sd*np.ones((train_tmp.shape[0], 1))), axis = 1)
        val_tmp = np.concatenate((val_tmp, sd*np.ones((val_tmp.shape[0], 1))), axis = 1)
        test_tmp = np.concatenate((test_tmp, sd*np.ones((test_tmp.shape[0], 1))), axis = 1)

        train_data.append(train_tmp)
        val_data.append(val_tmp)
        test_data.append(test_tmp)

        if basin in A_final:
            test_data_A.append(test_tmp)
            basins_used_A.append(basin)

train_data_final = np.concatenate(train_data)
val_data_final = np.concatenate(val_data)
test_data_final = np.concatenate(test_data)

# remove NaNs
train_data_final = removeNaN(train_data_final)
val_data_final = removeNaN(val_data_final)
test_data_final = removeNaN(test_data_final)

# response and predictor variables with streamflow nomalization 
ytrain = train_data_final[:,1]/train_data_final[:,-1]
xtrain = train_data_final[:,2:-1]

yval = val_data_final[:,1]/val_data_final[:,-1]
xval = val_data_final[:,2:-1]

ytest = test_data_final[:,1]/test_data_final[:,-1]
xtest = test_data_final[:,2:-1]

# prepare test data for each watershed separately
test_data_prep = []
sd_test = []
for tind in range(len(test_data_A)):
    test_data_tmp = test_data_A[tind]
    test_data_tmp = removeNaN(test_data_tmp)
    ytest_tmp = test_data_tmp[:,1]          # in cfs
    xtest_tmp = test_data_tmp[:,2:-1]       # sd values are removed from the set of predictor variables
    datenums_tmp = test_data_tmp[:,0]
    sd_test.append(test_data_tmp[0,-1])
    test_data_prep.append([datenums_tmp, ytest_tmp, xtest_tmp])

# RF parameter tuning, model trainign and testing
"""
n_estimators, max_features, max_depth, min_samples_leaf = RF_ParamTuning.RFRegParamTuningV(xtrain, ytrain, xval, yval)
rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = 200, min_samples_leaf  = min_samples_leaf, max_features = max_features, max_depth = max_depth, oob_score = True, n_jobs = -1, min_impurity_decrease = 0, ccp_alpha = 0)
rgr.fit(xtrain, ytrain)

# save model
fname = 'model_saved_exp2_1'
direc = 'D:/Research/non_staitionarity/codes/results/RF_global'
filename = direc + '/' + fname
pickle.dump(rgr, open(filename, 'wb'))
"""

# load model
sname = 'model_saved_exp2_1'
direc = 'D:/Research/non_staitionarity/codes/results/RF_global'
filename = direc + '/' + sname
rgr = pickle.load(open(filename, 'rb'))

# predict on test set
for tind in range(len(test_data_prep)):
    xtest_tmp = test_data_prep[tind][2]
    ypred = rgr.predict(xtest_tmp)*sd_test[tind]
    test_data_prep[tind].append(ypred)

# compute NSE
"""
NSE = []
for ind in range(0,len(test_data_prep)): 
    NSE.append(computeNSE(test_data_prep[ind][1],test_data_prep[ind][3]))

# save NSE values to a textfile
sname = 'basin_NSE_exp2_2.txt'
save_direc = 'D:/Research/non_staitionarity/codes/results/RF_global'
filename  = save_direc + '/' + sname
fid = open(filename, 'w')
fid.write('BASIN\tNSE\tsd\n')
for wind in range(0,len(A_final)):
    fid.write('%s\t%f\t%f\n'%(basins_used[wind], NSE[wind], sd_test[wind]))
fid.close()
"""

# compute bootstrap NSE
NSEBS = []
for ind in range(0,len(test_data_prep)):
    nse = computeNSEBS(test_data_prep[ind][1],test_data_prep[ind][3])
    NSEBS.append([basins_used_A[ind], nse])

# save NSE BS values
fname = 'nse_uncertainty_exp2_1'
direc = 'D:/Research/non_staitionarity/codes/results/RF_global'
filename = direc + '/' + fname
pickle.dump(NSEBS, open(filename, 'wb'))

a = 1