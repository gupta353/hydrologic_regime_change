"""
This script detects identifies the changes in the FRAIMA model parameters by computing slopes of the parameter time-series

Author: Abhinav Gupta (Created: 08 Nov 2021)

"""

import numpy as np
import os
import statsmodels.api as sm

# read gauage information
direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata'
fname = 'gauge_information.txt'
filename = direc + '/' + fname

fid = open(filename,'r')
gauge_info = fid.readlines()
fid.close()

gauge_info_data=  []
for gind in range(1,len(gauge_info)):
    data_tmp = gauge_info[gind].split('\t')
    gauge_info_data.append([data_tmp[1],float(data_tmp[3]),float(data_tmp[4])])
gauge_info_data = np.array(gauge_info_data)

###########################################################################################
wlen = 3650
mstep = 365*3

direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results'
listFile = os.listdir(direc)

slope_data = []
for local_dir in listFile:

    if os.path.isdir(direc + '/' + local_dir):

        fname = local_dir + '_coefficient_estimates_mean.txt'
        filename = direc + '/' + local_dir + '/' + fname

        fid = open(filename,'r')
        data = fid.readlines()
        fid.close()

        params_name = data[0].split()
        # identify AR and MA parameters
        boolean = []
        for pind in range(0,len(params_name)-2):
            boolean.append(any(string in params_name[pind] for string in ['AR']))
        boolean = np.array(boolean)
        ar_locs = np.nonzero(boolean == True)
        ma_locs = np.nonzero(boolean == False)
        p = ar_locs[0].shape[0]
        q = ma_locs[0].shape[0]

        coeffs = []
        for ind in range(1,len(data)):
            data_tmp = data[ind].split()
            coeffs.append(data_tmp)
        coeffs = np.array(coeffs)
        coeffs = coeffs.astype(np.float)

        numWindows = coeffs.shape[0]

        # compute slope for each parameter in for loop
        x = np.array(range(1,numWindows+1)).reshape(1,-1).T
        cons_term = np.ones((numWindows,1))
        x = np.concatenate((cons_term,x), axis = 1)
        slope = []
        for pind in range(0,coeffs.shape[1]):
            
            parData = coeffs[:,pind]
            model = sm.OLS(parData, x)
            results = model.fit()
            slope.append(results.params[1])

        # read deseasonalized streamflow data
        fname = 'strm.txt'
        filename = direc + '/' + local_dir + '/' + fname

        fid = open(filename,'r')
        strm_data = fid.readlines()
        fid.close()
        strm = []
        for ind in range(1,len(strm_data)):
            strm_tmp = strm_data[ind].split()[2]
            strm.append(float(strm_tmp))
        strm = np.array(strm)

        # compute average deseasonalized streamflows over different time-windows
        N = strm.shape[0]
        avg_strm = []
        for ind in range(0,N-wlen+1,mstep):
            avg_strm.append(np.mean(strm[ind:ind+wlen]))
        avg_strm = np.array(avg_strm)

        # compute average slope for AR and MA parameters
        ar_slope_avg = np.mean(slope[0:p])
        ma_slope_avg = np.mean(slope[p:p+q])
        d_slope = slope[p+q]

        # compute an aggregate index of change
        w1 = np.mean(avg_strm)
        w2 = np.mean(coeffs[:,-1])
        w1_plus_w2 = w1 + w2
        w1 = w1/w1_plus_w2
        w2 = w2/w1_plus_w2 
        aggregate_index = w1*ar_slope_avg + w2*ma_slope_avg

        # identify index of gauge in gauge_info_data
        gind = np.nonzero(gauge_info_data[:,0] == local_dir)
        slope_data.append([local_dir,ar_slope_avg,ma_slope_avg,d_slope,aggregate_index,float(gauge_info_data[gind,1]),float(gauge_info_data[gind,2])])

# write data to a textfile
sname = 'parameter_change_data.txt'
filename = direc + '/' + sname
fid = open(filename,'w')
fid.write('gauge_id\tar_slope_avg\tma_slope_avg\td_slope_avg\taggregate_index\tlat\tlong\n')
for write_data in slope_data:
    fid.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\n'%tuple(write_data))
fid.close()

a = 1