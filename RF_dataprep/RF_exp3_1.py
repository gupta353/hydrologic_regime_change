"""

This script implements RF method by first identifying calibration watersheds that are
similar to test waterhed in terms of climatic statistics (precipitation, rainfall and solar radiation) 

"""

import numpy as np
import os
import datetime
import sklearn.ensemble
import matplotlib.pyplot as plt
import RF_ParamTuning
import time
import pickle
import computeClimate

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

        if basin in A_final:
            test_data_A.append(test_tmp)

# choose a watershed
test_basin = test_data_A[1]
rain_data = test_basin[:,(0,2)]
tmax_data = test_basin[:,(0,35)]
tmin_data = test_basin[:,(0,68)]
sol_data = test_basin[:,(0,134)]
lambdas, alphas = computeClimate.PrcpStat(rain_data)
tmax_mean = computeClimate.TempSolStat(tmax_data)
tmin_mean = computeClimate.TempSolStat(tmin_data)
sol_mean = computeClimate.TempSolStat(sol_data)
tst_st = list(lambdas) + list(alphas) + list(tmax_mean) + list(tmin_mean) + list(sol_mean)
tst_st = np.array(tst_st)

# compute climate statistics for each watershed in training and test set
trn_st = []
for train_basin in train_data:
    rain_data = train_basin[:,(0,2)]
    tmax_data = train_basin[:,(0,35)]
    tmin_data = train_basin[:,(0,68)]
    sol_data = train_basin[:,(0,134)]

    lambdas, alphas = computeClimate.PrcpStat(rain_data)
    tmax_mean = computeClimate.TempSolStat(tmax_data)
    tmin_mean = computeClimate.TempSolStat(tmin_data)
    sol_mean = computeClimate.TempSolStat(sol_data)

    trn_st.append(list(lambdas) + list(alphas) + list(tmax_mean) + list(tmin_mean) + list(sol_mean))
trn_st = np.array(trn_st)

# normalize the training and test climate statistics based upon the traning mean and std
trn_st_m = np.nanmean(trn_st, axis = 0)
trn_st_sd = np.nanstd(trn_st, axis = 0)
for cind in range(trn_st.shape[1]):
    trn_st[:,cind] = (trn_st[:,cind] - trn_st_m[cind])/trn_st_sd[cind]

tst_st = (tst_st - trn_st_m)/trn_st_sd

# compute the euclidean distance between test and training climatic statistics
D = []
for bind in range(trn_st.shape[0]):
    dist = (np.sum((tst_st - trn_st[bind,:])**2))**0.5
    D.append(dist)
D = np.array(D)
D = np.concatenate(( D.reshape(1,-1).T, np.arange(0,len(train_data)).reshape(1,-1).T ), axis = 1)
D = D[np.argsort(D[:,0]), :]

## create RF models using different number of watersheds in the training set
# test data
ytest = test_basin[:,1]
xtest = test_basin[:,2:-1]
sd_test = test_basin[0,-1]

num_basins = range(20,200, 20)
NSE = []
for num in num_basins:
    trn_tmp = []
    for ind in D[0:num,1].astype('int'):
        trn_tmp.append(train_data[ind])

    train_data_final = np.concatenate(trn_tmp)
    ytrain = train_data_final[:,1]/train_data_final[:,-1]
    xtrain = train_data_final[:,2:-1]

    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = 200, min_samples_leaf  = 4, max_features = 0.33, max_depth = 200, oob_score = True, n_jobs = -1, min_impurity_decrease = 0, ccp_alpha = 0)
    rgr.fit(xtrain, ytrain)
    ypred = rgr.predict(xtest)*sd_test
    NSE.append(computeNSE(ytest, ypred))

a = 1
