"""

This script compares the PSD area obtained by using optimal FARIAM model parameter set and the mean psd obtained by using the
computing the average over different PSDs

Author: Abhinav Gupta (Created: 25 Mar 2022)

"""

import numpy as np
import os
import FarimaModule
import matplotlib.pyplot as plt
import scipy.stats

# function to create an illustration of plot
def readMeanArea(filename):
    fid = open(filename, 'r')
    data = fid.readlines()
    fid.close()
    tmp=[]
    for rind in range(1,len(data)):
        data_tmp = data[rind].split()
        tmp.append([float(val) for val in data_tmp])
    tmp = np.array(tmp)
    m = np.mean(tmp, axis = 0)
    return m

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

####################################################################################################
direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final'
window = 7          # window number for which comparison is to be made
# read the list of basins which need to be removed
fname = 'basin_list_significant_autocorr.txt'
filename = direc + '/' + fname
fid = open(filename,'r')
data = fid.readlines()
fid.close()
basin_list = []
for basin in data:
    basin_list.append(basin.split()[0])

# read the list of basins which need to be removed for other reasons
fname = 'basin_list_to_be_removed.txt'
filename = direc + '/' + fname
fid = open(filename,'r')
data = fid.readlines()
fid.close()
for basin in data:
    basin_list.append(basin.split()[0])  
#######################################################################################################
wlen = 3650
mstep = 365*3
N = 3650            # number of time steps at which streamflow is available in each window
fs = 1              # sampling frequency (number of samples per day)
fmax = fs/2         # maximum frequncy according to aliasing effect
#fregions = [[0, 2*10**(-3)], [2*10**(-3), 10**(-2)], [10**(-2), 2*10**(-2)],[2*10**(-2), 10**(-1)],[10**(-1), 2.5*10**(-1)],[2.5*10**(-1), 5*10**(-1)]]
fregions = [[0, 1/365], [1/365, 1/120], [1/120, 1/30], [1/30, 1/15], [1/15, 1/2], [1/365, 1/30],[1/30, 1/2]]
fregions_label = ['Greater than 1-year timescales', '4-months to 1-year timescales', '1-month to 4-months timescales', '2-weeks to 1-month timescales', 'Less than 2-weeks timescales', '1-month to 1-year timescales', 'Less than 1-month timescales']

# frequency values at which PSD is to be computed 
fint = 2*fmax/N
f = np.arange(0, fmax, fint)
f = f[1:]
one_sided = True

listFile = os.listdir(direc)

local_dir_ind = -1
write_data = []
area_opt = []
area_dist = []
write_data = []
for local_dir in listFile:
    if os.path.isdir(direc + '/' + local_dir) and (local_dir not in basin_list) and (local_dir != 'FARIMA_ML_Models'):
        
        bind  = [i for i in range(0,len(gauge_info_data)) if gauge_info_data[i][0] == local_dir]
        lat = gauge_info_data[bind[0]][1]
        long = gauge_info_data[bind[0]][2]
        
        # read mean area of the statistical distribution
        area_dist = []
        for win in range(0,9):
            fname = 'areas_distribution_' + str(win) + '.txt'
            filename = direc + '/' + local_dir + '/' + fname
            if os.path.isfile(filename):
                m = readMeanArea(filename)
                area_dist.append(m)
        area_dist = np.array(area_dist)

        # compute trend and p value of trend
        x = np.arange(1,area_dist.shape[0]+1)
        write_sl = []       # list to hold trend value for each frequency region
        write_pval = []     # list to hold p value of trend for each frequency region
        for find in range(0,area_dist.shape[1]):
            sl, inter, rval, pval, stdder = scipy.stats.linregress(x, area_dist[:,find])
            write_sl.append(sl)
            write_pval.append(pval)
        
        # read p-value of difference between areas
        fname = 'areas_p_values.txt'
        filename = direc + '/' + local_dir + '/' + fname
        fid = open(filename, 'r')
        data = fid.readlines()
        fid.close()
        pvals_diff = []
        for rind in range(1,len(data)):
            data_tmp = data[rind].split()
            pvals_diff.append([float(x) for x in data_tmp])
        pvals_diff = pvals_diff[-1]
        write_data.append([local_dir] + write_sl + write_pval + pvals_diff + [lat, long])

"""
x = []
y = []
z = []
for ii in range(0,len(write_data)):
    x.append(write_data[ii][7])
    y.append(write_data[ii][21])
    z.append(write_data[ii][14])
plt.scatter(x, y)
plt.scatter(x, z, s = 3)
plt.plot([np.min(x), np.max(x)], [0.05, 0.05])
plt.show()
"""

# write data to a textfile
sname = 'changePSD_statSgfcnc.txt'
filename = direc + '/' + sname
fid = open(filename, 'w')
fid.write('Gauge_id\tGreater_than_1_year\t4_months_to_1_year_timescales\t1_month_to_4_months\t2_weeks_to_1_month_timescales\tLess_than_2_weeks\t1_month_to_1_year\tLess_than_1_month\ttrend_p_val_1\ttrend_p_val_2\ttrend_p_val_3\ttrend_p_val_4\ttrend_p_val_5\ttrend_p_val_6\ttrend_p_val_7\tdiff_p_val_1\tdiff_p_val_2\tdiff_p_val_3\tdiff_p_val_4\tdiff_p_val_5\tdiff_p_val_6\tdiff_p_val_7\tLat\tLong\n')
formatspec = '%s\t' + '%f\t'*22 + '%f\n'
for wind in range(0,len(write_data)):
    fid.write(formatspec%tuple(write_data[wind]))
fid.close()
