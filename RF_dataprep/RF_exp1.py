"""
This script carries out RF experiment 1 (using data from set A which contains data from 
watersheds that have undergone streamflow regime change)

Author: Abhinav Gupta (Created: 25 May 2022)

"""

import numpy as np
import os
import datetime
import sklearn.ensemble
import matplotlib.pyplot as plt
import RF_ParamTuning
import time
import pickle

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
basins = [basins[i] for i in inds]

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
basins_final = []
for ind in range(0,len(basins)):
    basin_tmp = basins[ind]
    ii = [i for i in range(0, len(basins_531)) if basins_531[i] == basin_tmp]
    if len(ii) != 0:
        basins_final.append(basin_tmp)
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
train_datenum_1 = datetime.date(1980,10,1).toordinal()
train_datenum_2 = datetime.date(1989,9,30).toordinal()
val_datenum_1 = train_datenum_2+1
val_datenum_2 = datetime.date(1994,9,30).toordinal()
test_datenum_1 = datetime.date(2001,10,1).toordinal()

eps = 0.1 # to normalize MSE

train_data = []
val_data = []
test_data=[]
sd_vals = []
basins_used = []
for basin in basins_final:
    
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
        if len(ind_train_1[0]) == 0:
            ind_train_1 = 0
        else:
            ind_train_1 = ind_train_1[0]

        ind_train_2 = np.nonzero(met[:,0] == train_datenum_2)
        ind_train_2 = ind_train_2[0][0]
        ind_val_2 = np.nonzero(met[:,0] == val_datenum_2)
        ind_val_2 = ind_val_2[0][0]
        ind_test_1 = np.nonzero(met[:,0] == test_datenum_1)
        ind_test_1 = ind_test_1[0][0]

        train_tmp = met[ind_train_1:ind_train_2+1,:]
        val_tmp = met[ind_train_2+1:ind_val_2+1,:]
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

# prapare test data for each watershed separately
test_data_prep = []
for tind in range(len(test_data)):
    test_data_tmp = test_data[tind]
    test_data_tmp = removeNaN(test_data_tmp)
    ytest_tmp = test_data_tmp[:,1]          # in cfs
    xtest_tmp = test_data_tmp[:,2:-1]       # sd values are removed from the set of predictor variables
    datenums_tmp = test_data_tmp[:,0]
    test_data_prep.append([datenums_tmp, ytest_tmp, xtest_tmp])

# RF parameter tuning, model trainign and testing
n_estimators, max_features, max_depth, min_samples_leaf = RF_ParamTuning.RFRegParamTuningV(xtrain, ytrain, xval, yval)
rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = 200, min_samples_leaf  = min_samples_leaf, max_features = max_features, max_depth = max_depth, oob_score = True, n_jobs = -1, min_impurity_decrease = 0, ccp_alpha = 0)
rgr.fit(xtrain, ytrain)

# save model
fname = 'model_saved_exp1'
direc = 'D:/Research/non_staitionarity/codes/results/RF_global'
filename = direc + '/' + fname
pickle.dump(rgr, open(filename, 'wb'))

# predict on test set
for tind in range(len(test_data_prep)):
    xtest_tmp = test_data_prep[tind][2]
    ypred = rgr.predict(xtest_tmp)*sd_vals[tind]
    test_data_prep[tind].append(ypred)

NSE = []
for ind in range(0,len(test_data_prep)): 
    NSE.append(computeNSE(test_data_prep[ind][1],test_data_prep[ind][3]))

# save NSE values to a textfile
sname = 'basin_NSE_exp1.txt'
save_direc = 'D:/Research/non_staitionarity/codes/results/RF_global'
filename  = save_direc + '/' + sname
fid = open(filename, 'w')
fid.write('BASIN\tNSE\tsd\n')
for wind in range(0,len(basins_used)):
    fid.write('%s\t%f\t%f\n'%(basins_used[wind], NSE[wind], sd_vals[wind]))
fid.close()

a = 1