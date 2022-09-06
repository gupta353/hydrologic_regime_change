"""

For 'prediction in ungauged basin' 

This script creates several ML model for different clusters. For a given cluster, the watersheds closest
to cluster mean are identified using Mahalanobis distance and an LSTM is trained using the idnetified
watersheds. The number of similar watersheds used for calibration are varied succesively : {1, 2, 4, 8,
16, 32, 64, 90, 128}.

Similirity between watersheds is identified using the climatic statistics only: rainfall related and temperature related

Author: Abhinav Gupta (Created: 1 Sep 2022)

"""
import datetime
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import pickle
from random import choices
import copy
import random
import rainStats
import tempStats
import LSTMModule
import cProfile, pstats

import torch
from torch import classes, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

save_subdir = 'model_exp3_3'

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

# function to remove rows containing NaNs
def removeNaN(Y):
    X = Y.copy()
    del_ind = []
    for ind in range(0,len(X)):
        X_tmp = X[ind,:]
        if np.any(np.isnan(X_tmp)):
            del_ind.append(ind)

    if len(del_ind) != 0:
        Xp = np.delete(X, np.array(del_ind), axis = 0)
    else:
        Xp = X
    return Xp

# Euclidean distance between two vectors (equivalent to Mahalanobis distance for properly standardized vectors)
def Euclid(x1, x2):
    d = (np.sum((x1 - x2)**2))**0.5
    return d

####################################################################################################
####################################################################################################
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

# remove the basins in 'A' that are not contained in 'basins_531'
A_final = [a for a in A if a in basins_531]

## read the list of basisn which have not undergone a regime change
# basins that are contained in 531 basin but not in set A_final or 'basins'
B = [basin_tmp for basin_tmp in basins_531 if basin_tmp not in basins]

# read the cluster labels of the watersheds contained in set A_final
direc = 'D:/Research/non_staitionarity/codes/results/data_split'
fname = 'A_cluster_label.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
clus =[]
for ind in range(1, len(data)):
    data_tmp = data[ind].split()
    clus.append([data_tmp[0], float(data_tmp[1])])

# assign cluster labels to basins in A_final
A_final_label=  []
for ind in range(len(A_final)):
    basin = A_final[ind]
    label = [clus[i][1] for i in range(len(clus)) if clus[i][0]==basin]
    A_final_label.append(int(label[0]))

##################################################################################################
##################################################################################################
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
# prepare data for LSTM regression
direc_dynamic = 'D:/Research/non_staitionarity/data/LSTM_dynamic_data'

train_datenum_1 = datetime.date(1980,10,1).toordinal()
train_datenum_2 = datetime.date(1989,9,30).toordinal()
val_datenum_1 = datetime.date(1989,10,1).toordinal()
val_datenum_2 = datetime.date(1994,9,30).toordinal()
test_datenum_1 = datetime.date(2001,10,1).toordinal()
#test_datenum_2 = datetime.date(1999,9,30).toordinal()

"""
ind_train_1 = 0
ind_train_2 = 3650
ind_val_1 = 3650
ind_val_2 = 5445
ind_test_1 = 5445
ind_test_2 = ''
"""

eps = 0.1 # to normalize MSE

train_data = []
val_data = []
test_data=[]
SD_y = []
SD_y_A = []
test_data_A = []
basins_used = []
basins_used_A = []
for basin in A_final + B:          ###########

    # read dynamic data
    fname = basin + '_met_dynamic.txt'
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
        
        # if dates are specified for training, validation, and testing period
        ind_train_1 = np.nonzero(met[:,0] == train_datenum_1)
        if len(ind_train_1[0]) != 0:
            ind_train_1 = ind_train_1[0][0]
        else:
            ind_train_1 = 0
        ind_train_2 = np.nonzero(met[:,0] == train_datenum_2)
        ind_train_2 = ind_train_2[0][0]
        ind_val_1 = np.nonzero(met[:,0] == val_datenum_1)
        ind_val_1 = ind_val_1[0][0]
        ind_val_2 = np.nonzero(met[:,0] == val_datenum_2)
        ind_val_2 = ind_val_2[0][0]
        ind_test_1 = np.nonzero(met[:,0] == test_datenum_1)
        ind_test_1 = ind_test_1[0][0]
        # ind_test_2 = np.nonzero(met[:,0] == test_datenum_2)
        # ind_test_2 = ind_test_2[0][0]

        train_tmp = met[ind_train_1:ind_train_2+1,:]
        val_tmp = met[ind_val_1:ind_val_2+1,:]
        test_tmp = met[ind_test_1:,:]

        # compute standard deviation of streamflow in training set
        sd_data = np.array(train_tmp)[:,1]
        sd = np.nanstd(sd_data) + eps
        SD_y.append(sd)

        # add standard deviation as the last predictor (it should be removed before model training)
        train_tmp = np.concatenate((train_tmp, sd*np.ones((train_tmp.shape[0], 1))), axis = 1)
        val_tmp = np.concatenate((val_tmp, sd*np.ones((val_tmp.shape[0], 1))), axis = 1)
        test_tmp = np.concatenate((test_tmp, sd*np.ones((test_tmp.shape[0], 1))), axis = 1)

        # remove NaNs
        train_tmp = removeNaN(train_tmp)
        val_tmp = removeNaN(val_tmp)
        test_tmp = removeNaN(test_tmp)
        
        train_data.append(train_tmp)
        val_data.append(val_tmp)
        test_data.append(test_tmp)

        if basin in A_final:
            test_data_A.append(test_tmp)
            basins_used_A.append(basin)
            SD_y_A.append(sd)

##########################################################################################################
# compute rainfall and temperature statsitics for all watersheds for training period
cl_stat_tr = []
for train in train_data:
    prcp = train[:,2]
    tmax = train[:,3]
    tmin = train[:,4]
    datenums = train[:,0]
    
    # rainfall stats
    ns_tr = rainStats.numStorms(prcp)                      # number of storms per day
    msd_tr = rainStats.meanStormDepth(prcp)                # mean storm depth
    nrd_tr = rainStats.FracRainDays(prcp)                  # frcation of prcp days
    hf_tr = rainStats.high_prcp_freq(prcp)                 # fraction of high precipitation days
    lf_tr = rainStats.low_prcp_freq(prcp)                  # fraction of low precipitation days
    hd_tr = rainStats.avg_high_prcp_duration(prcp)         # average high prcp duration
    ld_tr = rainStats.avg_low_prcp_duration(prcp)          # average low prcp duration
    hp_tr = rainStats.high_prcp_avg(prcp)                  # average high prcp average value
    lp_tr = rainStats.low_prcp_avg(prcp)                   # average low prcp average value
    sr_tr = rainStats.seasonalRain(prcp, datenums)         # compute seasonal average prcipitation
    pstat_tr = [ns_tr]+[msd_tr]+[nrd_tr]+[hf_tr]+[lf_tr]+[hd_tr]+[ld_tr]+[hp_tr]+[lp_tr] + sr_tr

    # temperature stats
    stmax_tr = tempStats.seasonalTemp(tmax, datenums)      # mean seasonal maximum-temperature
    stmin_tr = tempStats.seasonalTemp(tmin, datenums)      # mean seasonal minimum-temperature
    tstat_tr = stmax_tr + stmin_tr

    cl_stat_tr.append(pstat_tr + tstat_tr)
cl_stat_tr = np.array(cl_stat_tr)

# compute rainfall and temperature statsitics for all watersheds for testing period
cl_stat_ts = []
for test in test_data_A:
    prcp = test[:,2]
    tmax = test[:,3]
    tmin = test[:,4]
    datenums = test[:,0]
    
    # rainfall stats
    ns_ts = rainStats.numStorms(prcp)                      # number of storms per day
    msd_ts = rainStats.meanStormDepth(prcp)                # mean storm depth
    nrd_ts = rainStats.FracRainDays(prcp)                  # frcation of prcp days
    hf_ts = rainStats.high_prcp_freq(prcp)                 # fraction of high precipitation days
    lf_ts = rainStats.low_prcp_freq(prcp)                  # fraction of low precipitation days
    hd_ts = rainStats.avg_high_prcp_duration(prcp)         # average high prcp duration
    ld_ts = rainStats.avg_low_prcp_duration(prcp)          # average low prcp duration
    hp_ts = rainStats.high_prcp_avg(prcp)                  # average high prcp average value
    lp_ts = rainStats.low_prcp_avg(prcp)                   # average low prcp average value
    sr_ts = rainStats.seasonalRain(prcp, datenums)         # compute seasonal average prcipitation
    pstat_ts = [ns_ts]+[msd_ts]+[nrd_ts]+[hf_ts]+[lf_ts]+[hd_ts]+[ld_ts]+[hp_ts]+[lp_ts] + sr_ts

    # temperature stats
    stmax_ts = tempStats.seasonalTemp(tmax, datenums)      # mean seasonal maximum-temperature
    stmin_ts = tempStats.seasonalTemp(tmin, datenums)      # mean seasonal minimum-temperature
    tstat_ts = stmax_ts + stmin_ts

    cl_stat_ts.append(pstat_ts + tstat_ts)
cl_stat_ts = np.array(cl_stat_ts)

# normalize the climatic statistics                                     ### check
cl_m = np.mean(np.concatenate((cl_stat_tr, cl_stat_ts)), axis = 0)
cl_std = np.std(np.concatenate((cl_stat_tr, cl_stat_ts)), axis = 0)
for col in range(cl_stat_tr.shape[1]):
    cl_stat_tr[:,col] = (cl_stat_tr[:,col] - cl_m[col])/cl_std[col]
    cl_stat_ts[:,col] = (cl_stat_ts[:,col] - cl_m[col])/cl_std[col]

############################################################################################################
############################################################################################################
############################################################################################################

# choose cluster label for which models are to be created
#label = 0
for label in range(1,10):      ################
    label_ind = [i for i in range(len(A_final_label)) if A_final_label[i]==label]           # index of watersheds with 'label' in 'A_final' 
    label_ind = np.array(label_ind)
    clus_basins = [A_final[i] for i in label_ind]                                           # basins contained in cluster

    # indices of basins conatined in 'basin_used_A' (basins that are contained in cluster)
    basin_used_label_ind = [i for i in range(len(basins_used_A)) if basins_used_A[i] in clus_basins]       
    basin_label = [basins_used_A[i] for i in basin_used_label_ind]
    test_rel = [test_data_A[i] for i in basin_used_label_ind]
    SD_y_A_label = [SD_y_A[i] for i in basin_used_label_ind]

    # choose cluster center at test basins in climatic statistics space 
    clus_mean = np.mean(cl_stat_ts[basin_used_label_ind,:], axis = 0)

    # find the distance between test cluster mean and training data points 
    dist = []
    for ind in range(cl_stat_tr.shape[0]):
        dist.append(Euclid(cl_stat_tr[ind,:], clus_mean))
    dist = np.array(dist)

    # replace the distance corresponding to basins contained in clusters by 'inf' (training data from these basins will be used for each model)
    inf_ind = []
    for ind in range(len(basins_used)):
        if basins_used[ind] in clus_basins:
            dist[ind] = np.inf
            inf_ind.append(ind)

    # choose nearest basins in traning set and create LSTM models (the basins contained in the cluster are always chosen)

    num_basins = [1, 2, 4, 8, 16, 32, 64, 90, 128]
    #num_basins = [90,128]
    batch_size = 2**8
    input_dim = len(test_rel[0][0,2:-1])        ###################
    hidden_dim = 256
    n_layers = 1
    output_dim = 1
    epochs = 50
    seeds = [i for i in range(8)]
    for num in num_basins:

        save_direc = 'D:/Research/non_staitionarity/codes/results/LSTM_global/' + save_subdir + '/' + str(label)
        if os.path.exists(save_direc) == False:
            os.mkdir(save_direc)

        # indices of basins in train_data to be used for training
        dist_x = dist.copy()
        idx = np.argsort(dist_x)[0:num]
        #idx = np.concatenate((idx, inf_ind), axis = 0)

        # extract relevant traning and validation data 
        train_rel = [train_data[i].copy() for i in idx]
        val_rel = [val_data[i].copy() for i in idx]
        test_rel_X = [test_data_tmp.copy() for test_data_tmp in test_rel]

        NSE_avg, NSEBS = LSTMModule.LSTMImplement(train_rel, val_rel, test_rel_X, batch_size, input_dim, hidden_dim, n_layers, output_dim, epochs, seeds, device, save_direc, num)
        #mean_X, std_X = LSTMModule.LSTMImplement(train_rel, val_rel, test_rel_X, batch_size, input_dim, hidden_dim, n_layers, output_dim, epochs, seeds, device, save_direc, num)

        # save mean_X and std_X
        """
        normalization_stat = [mean_X,std_X]
        filename  = save_direc + '/' + 'norm_stat_clus_' + str(label) + '_numbasins_' + str(num) + '.txt'
        fid = open(filename, 'w')
        fid.write('mean_X\tstd_X\n')
        for wind in range(0,len(mean_X)):
            fid.write('%s\t%f\n'%(mean_X[wind], std_X[wind]))
        fid.close()
        """
        # save NSE values to a textfile
        filename  = save_direc + '/' + 'nse_avg_clus_' + str(label) + '_numbasins_' + str(num) + '.txt'
        fid = open(filename, 'w')
        fid.write('BASIN\tNSE\tsd\n')
        for wind in range(0,len(NSE_avg)):
            fid.write('%s\t%f\t%f\n'%(basin_label[wind], NSE_avg[wind], SD_y_A_label[wind]))
        fid.close()

        # save NSE BS values
        fname = 'nse_uncertainty_clus_' + str(label) + '_numbasins_' + str(num)
        filename = save_direc + '/' + fname
        pickle.dump(NSEBS, open(filename, 'wb'))

        gc.collect()
        torch.cuda.empty_cache()

###########################################################################################################################################################
"""
if __name__ == '__main__':
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    stats = pstats.Stats(pr).strip_dirs().sort_stats('tottime')
    stats.dump_stats('D:/Research/non_staitionarity/codes/results/LSTM_global/model_exp3_0/profile_data.prof')

    # Save as textfile
    result = io.StringIO()
    stats = pstats.Stats(pr, stream = result).strip_dirs().sort_stats('tottime')
    stats.print_stats()
    with open('D:/Research/non_staitionarity/codes/results/LSTM_global/model_exp3_0/profile_data.txt', 'w+') as f:
        f.write(result.getvalue())
"""