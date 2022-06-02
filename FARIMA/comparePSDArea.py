"""

This script compares the PSD area obtained by using optimal FARIAM model parameter set and the mean psd obtained by using the
computing the average over different PSDs

Author: Abhinav Gupta (Created: 25 Mar 2022)

"""

import numpy as np
import os
import statsmodels.api as sm
import FarimaModule
import matplotlib.pyplot as plt
from scipy.integrate import simpson

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
gauge_info_data = np.array(gauge_info_data)
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
for local_dir in listFile:
    if os.path.isdir(direc + '/' + local_dir) and (local_dir not in basin_list) and (local_dir != 'FARIMA_ML_Models'):
        
        # read area for optimal FARIMA parameter
        fname = 'area_optimal_param_set.txt'
        filename = direc + '/' + local_dir + '/' + fname
        fid = open(filename, 'r')
        data = fid.readlines()
        fid.close()
        area_opt_tmp = []
        for rind in range(1,len(data)):
            data_tmp = data[rind].split()
            area_opt_tmp.append([float(val) for val in data_tmp])
        area_opt.append(area_opt_tmp)

        # read mean area of the statistical distribution
        area_dist_tmp = []
        for win in range(0,len(area_opt_tmp)):
            fname = 'areas_distribution_' + str(win) + '.txt'
            filename = direc + '/' + local_dir + '/' + fname
            m = readMeanArea(filename)
            area_dist_tmp.append(m)
        area_dist.append(area_dist_tmp)

## create scatter plot that compare the results obtained by the two methods
x = []
y = []
for ind in range(0,len(area_opt)):
    x.append(area_opt[ind][window])
    y.append(area_dist[ind][window])
x = np.array(x)
y = np.array(y)

# plots
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
min_x = np.min([np.min(x), np.min(y)])
max_x = np.min([np.max(x), np.max(y)])
for pind in range(0,x.shape[1]):
    plt.scatter(x[:,pind], y[:,pind], s = 5)
plt.legend(fregions_label, frameon = False)
plt.plot([min_x, max_x],[min_x, max_x], color = 'black')
plt.xlim([min_x, max_x])
plt.ylim([min_x, max_x])
plt.xlabel('Obtained by optimal parameter set')
plt.ylabel('Obtained by taking mean of the PSDs, computed\n using parameter posterior distribution')

sname = 'comaprison_mean_opt_PSD_window_' + str(window) + '.png'
filename = direc + '/' + sname
plt.savefig(filename, dpi = 300)
        
