"""
This script reads autocorrelation data for each time-window and identifies the watersheds for which
the absolute value of autocorrelation is greater than a threshold at any time-step

Author: Abhinav Gupta (Created on: 22 Nov 2021)

"""
import os
import numpy as np
import matplotlib.pyplot as plt

direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final'
data_direc = 'D:/Research/non_staitionarity/data'

fname = 'autocorrelations_means.txt'

corr_thresh = 0.15  # threshold below which correlation is assumed negligible
numTimeLags = 30    # number of time lags upto which auto-correlations are considered
zero_thresh = 1     # number of time-windows where authocorrelation was greater than the threshold for atleast one time lag  

listFile = os.listdir(direc)

basin_list = []
for local_dir in listFile:

    if os.path.isdir(direc + '/' + local_dir) and (local_dir != 'FARIMA_ML_Models'):
        
        filename = direc + '/' + local_dir + '/' + fname

        fid = open(filename)
        data = fid.readlines()
        fid.close()

        corrs = []

        for corr_ind in range(1,len(data)):
            corrs.append(data[corr_ind].split())
        corrs = np.array(corrs)
        corrs = corrs.astype('float')
        corrs  = np.delete(corrs,0,0)
        corrs = corrs[0:numTimeLags,:]

        thresh_inds = []
        for col_ind in range(0,corrs.shape[1]):
            thresh_inds.append(list(np.nonzero(np.absolute(corrs[:,col_ind])>corr_thresh)))
        thresh_inds = np.array(thresh_inds)
        thresh_ind_sizes = [thresh_inds[i].size for i in range(0,len(thresh_inds))]
        thresh_ind_sizes = np.array(thresh_ind_sizes)
        
        numZeros = len(np.nonzero(thresh_ind_sizes)[0])
        if numZeros > zero_thresh:
            basin_list.append(local_dir)
        
print(len(basin_list))

# write basin list to a textfile
sname = 'basin_list_significant_autocorr.txt'
filename = direc + '/' + sname
fid = open(filename,'w')
for wind in range(0,len(basin_list)):
    fid.write('%s\n'%basin_list[wind])
fid.close()

