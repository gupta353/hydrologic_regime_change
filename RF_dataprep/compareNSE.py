"""
This script compares NSE obatined by prediction in time model and global Model

Author: Abhinav Gupta (Created: 11 May 2022)

"""
import numpy as np
import matplotlib.pyplot as plt

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
##############################################################################################    
# read NSEs for global model
"""
direc_1 = 'D:/Research/non_staitionarity/codes/results/RF_global'
fname = 'basin_nse_lat_long.txt'
filename = direc_1 + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
basin_nse_global = []
for rind in range(1, len(data)):
    data_tmp = data[rind].split()
    basin_nse_global.append([data_tmp[0], float(data_tmp[1])])
"""

# read NSEs for global model 1
direc_1 = 'D:/Research/non_staitionarity/codes/results/RF_global'
fname = 'basin_NSE_sd_normalized.txt'
filename = direc_1 + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
basin_nse_global_0 = []
for rind in range(1, len(data)):
    data_tmp = data[rind].split()
    basin_nse_global_0.append([data_tmp[0], float(data_tmp[1])])

# read NSEs for global model 2
direc_1 = 'D:/Research/non_staitionarity/codes/results/RF_global'
fname = 'basin_NSE_sd_normalized_diff_timeperiod_flip.txt'
filename = direc_1 + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
basin_nse_global_1 = []
for rind in range(1, len(data)):
    data_tmp = data[rind].split()
    basin_nse_global_1.append([data_tmp[0], float(data_tmp[1])])

# read NSEs for local model
direc_1 = 'D:/Research/non_staitionarity/codes/results/prediction_in_time_gis_files'
fname = 'RF_prediction_in_time_nses_3.txt'
filename = direc_1 + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
basin_nse_local = []
for rind in range(1, len(data)):
    data_tmp = data[rind].split()
    basin_nse_local.append([data_tmp[0], float(data_tmp[2])])

# match nses of the global and local model
"""
final_data= []
basins_final = []
for ind in range(0,len(basin_nse_global)):
    
    basin = basin_nse_global[ind][0]

    # lat long
    lind = [i for i in range(0,len(gauge_info_data)) if gauge_info_data[i][0]==basin]
    lat = gauge_info_data[lind[0]][1]
    long = gauge_info_data[lind[0]][2]

    nse_global = basin_nse_global[ind][1]

    ind1 = [i for i in range(0, len(basin_nse_local)) if basin_nse_local[i][0] == basin]
    nse_local = basin_nse_local[ind1[0]][1]

    basins_final.append(basin)
    final_data.append([nse_local, nse_global, lat, long])
"""

# match NSE of the two global models
final_data= []
basins_final = []
for ind in range(0,len(basin_nse_global_1)):
    
    basin = basin_nse_global_1[ind][0]

    # lat long
    lind = [i for i in range(0,len(gauge_info_data)) if gauge_info_data[i][0]==basin]
    lat = gauge_info_data[lind[0]][1]
    long = gauge_info_data[lind[0]][2]

    nse_global_1 = basin_nse_global_1[ind][1]

    ind1 = [i for i in range(0, len(basin_nse_global_0)) if basin_nse_global_0[i][0] == basin]
    nse_global_0 = basin_nse_global_0[ind1[0]][1]

    basins_final.append(basin)
    final_data.append([nse_global_0, nse_global_1, lat, long])

final_data  = np.array(final_data)
plt.scatter(final_data[:,0], final_data[:,1])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([-2, 1], [-2, 1], color = 'black')
plt.grid(linestyle = '--')
plt.xlabel('NSE local model')
plt.ylabel('NSE global model')
plt.show()

# save plot
"""
sname = 'compare_RF_global_local_model_2.png'
save_direc = 'D:/Research/non_staitionarity/codes/results/ML_models_plots'
filename = save_direc + '/' + sname
plt.savefig(filename, dpi = 300)
plt.close()

# save the difference between global and local model in a textfile
nse_diff = final_data[:,1] - final_data[:,0]
sname = 'nse_diff_global_minus_local.txt'
save_direc = 'D:/Research/non_staitionarity/codes/results/RF_global'
filename = save_direc + '/' + sname
fid = open(filename, 'w')
fid.write('Basin\tnse_diff\tlat\tlong\n')
for wind in range(0, len(basins_final)):
    fid.write('%s\t%f\t%f\t%f\n'%(basins_final[wind], nse_diff[wind], final_data[wind,2], final_data[wind,3]))
fid.close()
"""
plt.hist(final_data[:,0], bins = 10000, cumulative=True, label='CDF', histtype='step', density = True)
plt.hist(final_data[:,1], bins = 10000, cumulative=True, label='CDF', histtype='step', density = True)
plt.xlim([0, 1])
plt.grid(linestyle = '--')
plt.show()