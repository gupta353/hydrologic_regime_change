"""
RF regression model for one watershed

Author: Abhinav Gupta (Created: 19 Apr 2022)

"""
import datetime
import os
import numpy as np
import sklearn.ensemble
import matplotlib.pyplot as plt
import RF_ParamTuning
from sklearn.tree import DecisionTreeRegressor

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
# prepare data for RF regression
direc_dynamic = 'D:/Research/non_staitionarity/data/RF_dynamic_data_2'
direc_save = 'D:/Research/non_staitionarity/codes/results/RF_predictions_in_time_2'

# prepare training data
train_data = []
for basin in basins[0:]:

    # read dynamic data
    fname = basin + '_met_dynamic_2.txt'
    filename = direc_dynamic + '/' + fname
    met_data = readDynamic(filename)

    # read static data
    ind = [i for i in range(0,len(static_data)) if static_data[i][0]==basin]
    ind = ind[0]
    static_tmp = static_data[ind][1:]

    # combine met and static data
    for ind in range(0,len(met_data)):
        met_data[ind] = met_data[ind] + static_tmp    
    
    train = np.array(met_data)

    # train set
    xtrain = train[0:5445,2:110]
    ytrain = 24*3.6*0.02832*train[0:5445,1]/train[0:5445,120]

    # test set
    xtest = train[5445:,2:110]
    ytest = 24*3.6*0.02832*train[5445:,1]/train[5445:,120]

    # remove rows containing NaNs
    nanind = (np.isnan(xtest).any(axis=1))
    nanind = np.nonzero(nanind==True)
    nanind = nanind[0]
    xtest = np.delete(xtest, tuple(nanind), axis=0)
    ytest = np.delete(ytest, tuple(nanind), axis=0)

    # fit RF model
    n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf = RF_ParamTuning.RFRegParamTuning(xtrain, ytrain)
    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators = n_estimators, min_samples_leaf  = min_samples_leaf, max_features = max_features, min_samples_split = min_samples_split, max_depth = max_depth, oob_score = True, n_jobs = 5, min_impurity_decrease = 0, ccp_alpha = 0)
    rgr.fit(xtrain,ytrain)
    ypred = rgr.predict(xtest)

    # compute nse
    sse = np.sum((ytest-ypred)**2)
    sst = np.sum((ytest-np.mean(ytest))**2)
    nse = 1 - sse/sst

    # save test streamflow data to a textfile along with nse of prediction on test set
    sname = basin + '_test_obs_pred.txt'
    filename = direc_save + '/' + sname
    fid = open(filename, 'w')
    fid.write('NSE = ' + str(nse) + '\n')
    fid.write('Number of trees = ' + str(n_estimators) + '\n')
    fid.write('Minimum numbers of samples in a leaf node = ' + str(min_samples_leaf) + '\n')
    fid.write('Maximum nuber of features at a split = ' + str(max_features) + '\n')
    fid.write('Minimum number of samples for splitting = ' + str(min_samples_split) + '\n')
    fid.write('Maximum tree depth = ' + str(max_depth) + '\n')
    fid.write('Observed\tPredicted\n')
    for wind in range(0,len(ytest)):
        fid.write('%f\t%f\n'%(ytest[wind], ypred[wind]))
    fid.close()

    plt.scatter(ytest, ypred, s = 5)
    lim = np.max((np.max(ytest), np.max(ypred)))
    plt.ylim(0, lim)
    plt.xlim(0, lim)
    plt.plot([0, lim],[0, lim], color = 'black')
    plt.title('NSE = ' + str(round(nse*100)/100))
    plt.xlabel('Observed Streamflow (mm/day)')
    plt.ylabel('Estimated Streamflow (mm/day)')
    
    # save plot
    sname = basin + '_obs_vs_pred_plot.png'
    filename = direc_save + '/obs_vs_pred_plots/' + sname
    plt.savefig(filename, dpi = 300)
    plt.close()
