"""

Benchmark RF model for prediction in time using data across several watersheds 

Author: Abhinav Gupta (Created: 2 May 2022)

"""
import datetime
import os
import numpy as np
import sklearn.ensemble
import matplotlib.pyplot as plt
import RF_ParamTuning
from sklearn.tree import DecisionTreeRegressor
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

######################################################################################################
# read all the rain-dominated watersheds data
direc = 'D:/Research/non_staitionarity/data/RF_dynamic_data_2'
listFiles = os.listdir(direc)
basins = []

for fname in listFiles:
    basins.append(fname.split('_')[0])

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
 
########################################################################################
# read the list of 531 basins
direc = 'D:/Research/non_staitionarity/data/531_basins'
fname = 'basin_list_zero_appended.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
basin_list = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    basin_list.append(data_tmp[0])

########################################################################################

# prepare data for RF regression
direc_dynamic = 'D:/Research/non_staitionarity/data/RF_dynamic_data_2'
ind_train_1 = 0
ind_train_2 = 1825
ind_val_1 = 3650
ind_val_2 = 5445
ind_test_1 = 1825
ind_test_2 = 3650

eps = 0.1 # to normalize MSE

train_data = []
val_data = []
test_data=[]
sd_vals = []
basins_used = []
for basin in basin_list:
    
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
        
        train_tmp = met[ind_train_1:ind_train_2,:]
        val_tmp = met[ind_val_1:ind_val_2,:]
        test_tmp = met[ind_test_1:ind_test_2,:]

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


# normalize the streamflow data of training and validation sets
"""
for ind in len(train_data):
    train_data_tmp = train_data[ind]
    m = np.mean(train_data_tmp)
"""

train_data_final = np.concatenate(train_data)
val_data_final = np.concatenate(val_data)
test_data_final = np.concatenate(test_data)

# remove NaNs
train_data_final = removeNaN(train_data_final)
val_data_final = removeNaN(val_data_final)
test_data_final = removeNaN(test_data_final)

# response and predictor variables without streamflow nomalization (Here streamflow is converted to mm day)
"""
ytrain = 24*3.6*0.02832*train_data_final[:,1]/train_data_final[:,126]
xtrain = train_data_final[:,2:]

yval = 24*3.6*0.02832*val_data_final[:,1]/val_data_final[:,126]
xval = val_data_final[:,2:]

ytest = 24*3.6*0.02832*test_data_final[:,1]/test_data_final[:,126]
xtest = test_data_final[:,2:]
"""

# response and predictor variables with streamflow nomalization 
ytrain = train_data_final[:,1]/train_data_final[:,-1]
xtrain = train_data_final[:,2:-1]

yval = val_data_final[:,1]/val_data_final[:,-1]
xval = val_data_final[:,2:-1]

ytest = test_data_final[:,1]/test_data_final[:,-1]
xtest = test_data_final[:,2:-1]

###########################################################################################
# testing of RF speed with progresively large number of rows
#nrows = [5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 2500000]
"""
nrows = [5000, 10000, 20000, 50000, 100000, 200000]
dt = []
NSE = []
for rows in nrows:
    t = time.time()
    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = 100, min_samples_leaf  = 1, max_features = 0.33, min_samples_split = 2, max_depth = 100, oob_score = True, n_jobs = -1, min_impurity_decrease = 0, ccp_alpha = 0)
    rgr.fit(xtrain[0:rows],ytrain[0:rows])
    # ypred = rgr.predict(xtest[1:5000,:])
    #NSE.append(computeNSE(ytest[1:5000], ypred)) 
    dt.append(time.time() - t)
"""
#############################################################################
#### testing the model for tree depth (tree depth does not seem to have much impact; may be 100-500 is good enough)
"""
# choose an xtest
xtest = xtest[slice(0, xtest.shape[0], 10),:]
ytest = ytest[slice(0, ytest.shape[0], 10)]
del_ind = []
for ind in range(0,len(xtest)):
    xtest_tmp = xtest[ind,:]
    if np.all(np.isnan(xtest_tmp) == False):
        a = 1
    else:
        del_ind.append(ind)
xtest = np.delete(xtest, np.array(del_ind), axis = 0)
ytest = np.delete(ytest, np.array(del_ind))

dt = []
NSE = []
depths = [20, 50, 100, 200, 400, 500, 750, 1000, 2000, 5000, 10000]
for depth in depths:
    t = time.time()
    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = 100, min_samples_leaf  = 1, max_features = 0.33, min_samples_split = 2, max_depth = depth, oob_score = True, n_jobs = -1, min_impurity_decrease = 0, ccp_alpha = 0)
    rgr.fit(xtrain,ytrain)
    ypred = rgr.predict(xtest)
    NSE.append(computeNSE(ytest, ypred)) 
    dt.append(time.time() - t)
"""

#############################################################################
#### testing the model for number of trees (200 trees seem to be enough)
"""
xtest = xtest[slice(0, xtest.shape[0], 10),:]
ytest = ytest[slice(0, ytest.shape[0], 10)]
del_ind = []
for ind in range(0,len(xtest)):
    xtest_tmp = xtest[ind,:]
    if np.all(np.isnan(xtest_tmp) == False):
        a = 1
    else:
        del_ind.append(ind)
xtest = np.delete(xtest, np.array(del_ind), axis = 0)
ytest = np.delete(ytest, np.array(del_ind))

dt = []
NSE = []
trees = [25, 50, 75, 100, 200, 500]
for nest in trees:
    t = time.time()
    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = nest, min_samples_leaf  = 1, max_features = 0.33, min_samples_split = 2, max_depth = 20, oob_score = True, n_jobs = -1, min_impurity_decrease = 0, ccp_alpha = 0)
    rgr.fit(xtrain,ytrain)
    ypred = rgr.predict(xtest)
    NSE.append(computeNSE(ytest, ypred)) 
    dt.append(time.time() - t)
"""

#############################################################################
#### testing the model for minimum samples in the leaf (optimal value seems to be 8)
"""
xtest = xtest[slice(0, xtest.shape[0], 10),:]
ytest = ytest[slice(0, ytest.shape[0], 10)]
del_ind = []
for ind in range(0,len(xtest)):
    xtest_tmp = xtest[ind,:]
    if np.all(np.isnan(xtest_tmp) == False):
        a = 1
    else:
        del_ind.append(ind)
xtest = np.delete(xtest, np.array(del_ind), axis = 0)
ytest = np.delete(ytest, np.array(del_ind))

dt = []
NSE = []
leaf_samps = [1, 2, 4, 6, 8, 10, 15, 20]
for samps in leaf_samps:
    t = time.time()
    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = 50, min_samples_leaf  = samps, max_features = 0.33, min_samples_split = 2, max_depth = 100, oob_score = True, n_jobs = -1, min_impurity_decrease = 0, ccp_alpha = 0)
    rgr.fit(xtrain,ytrain)
    ypred = rgr.predict(xtest)
    NSE.append(computeNSE(ytest, ypred)) 
    dt.append(time.time() - t)
"""

#############################################################################
#### testing the model for max feature value required for splitting (0.3 to 0.70 seems to be optimal value)
"""
xtest = xtest[slice(0, xtest.shape[0], 10),:]
ytest = ytest[slice(0, ytest.shape[0], 10)]
del_ind = []
for ind in range(0,len(xtest)):
    xtest_tmp = xtest[ind,:]
    if np.all(np.isnan(xtest_tmp) == False):
        a = 1
    else:
        del_ind.append(ind)
xtest = np.delete(xtest, np.array(del_ind), axis = 0)
ytest = np.delete(ytest, np.array(del_ind))

dt = []
NSE = []
max_featu = [0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]
for maxf in max_featu:
    t = time.time()
    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = 50, min_samples_leaf  = 8, max_features = maxf, min_samples_split = 2, max_depth = 100, oob_score = True, n_jobs = -1, min_impurity_decrease = 0, ccp_alpha = 0)
    rgr.fit(xtrain,ytrain)
    ypred = rgr.predict(xtest)
    NSE.append(computeNSE(ytest, ypred)) 
    dt.append(time.time() - t)
"""

#### testing the model for max feature value required for splitting (0 is optimal value)
"""
xtest = xtest[slice(0, xtest.shape[0], 10),:]
ytest = ytest[slice(0, ytest.shape[0], 10)]
del_ind = []
for ind in range(0,len(xtest)):
    xtest_tmp = xtest[ind,:]
    if np.all(np.isnan(xtest_tmp) == False):
        a = 1
    else:
        del_ind.append(ind)
xtest = np.delete(xtest, np.array(del_ind), axis = 0)
ytest = np.delete(ytest, np.array(del_ind))

dt = []
NSE = []
impurity = [0, 10**(-10), 10**(-9), 10**(-8), 10**(-7)]
for imp in impurity:
    t = time.time()
    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = 50, min_samples_leaf  = 8, max_features = 0.35, min_samples_split = 2, max_depth = 100, oob_score = True, n_jobs = -1, min_impurity_decrease = imp, ccp_alpha = 0)
    rgr.fit(xtrain,ytrain)
    ypred = rgr.predict(xtest)
    NSE.append(computeNSE(ytest, ypred)) 
    dt.append(time.time() - t)
"""
##########################################################################################################
# prapare test data for each watershed separately
test_data_prep = []
for tind in range(len(test_data)):
    test_data_tmp = test_data[tind]
    test_data_tmp = removeNaN(test_data_tmp)
    ytest_tmp = test_data_tmp[:,1]          # in cfs
    xtest_tmp = test_data_tmp[:,2:-1]       # sd values are removed from the set of predictor variables
    datenums_tmp = test_data_tmp[:,0]
    test_data_prep.append([datenums_tmp, ytest_tmp, xtest_tmp])

# RF parameter tuning, model training and testing
# n_estimators, max_features, max_depth, min_samples_leaf = RF_ParamTuning.RFRegParamTuningV(xtrain, ytrain, xval, yval)
# max_features = 0.35
# max_depth = 197
# min_samples_leaf = 4
rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = 200, min_samples_leaf  = 4, max_features = 0.35, max_depth = 197, oob_score = True, n_jobs = -1, min_impurity_decrease = 0, ccp_alpha = 0)
rgr.fit(xtrain, ytrain)

# save model
fname = 'model_saved__normalized_diff_timeperiod_1'
direc = 'D:/Research/non_staitionarity/codes/results/RF_global'
filename = direc + '/' + fname
pickle.dump(rgr, open(filename, 'wb'))


# predict on test set
for tind in range(len(test_data)):
    xtest_tmp = test_data_prep[tind][2]
    ypred = rgr.predict(xtest_tmp)*sd_vals[tind]
    test_data_prep[tind].append(ypred)

NSE = []
for ind in range(0,len(test_data_prep)): 
    NSE.append(computeNSE(test_data_prep[ind][1],test_data_prep[ind][3]))

# save NSE values to a textfile
sname = 'basin_NSE_sd_normalized_diff_timeperiod_1.txt'
save_direc = 'D:/Research/non_staitionarity/codes/results/RF_global'
filename  = save_direc + '/' + sname
fid = open(filename, 'w')
fid.write('BASIN\tNSE\tsd\n')
for wind in range(0,len(basins_used)):
    fid.write('%s\t%f\t%f\n'%(basins_used[wind], NSE[wind], sd_vals[wind]))
fid.close()

a = 1


