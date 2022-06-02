"""
This script collects area under NPSD of contributions from different frequwncy regions in the firs time-window

Author: Abhinav Gupta (Created: 30 March 2022)

"""
import os
import numpy as np

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
###############################################
direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final'

listFile = os.listdir(direc)
fname1 = 'area_optimal_param_set.txt'

write_data = []
for folder in listFile:
    dir_tmp = direc + '/' + folder
    filename = dir_tmp  + '/' + fname1
    if os.path.isdir(dir_tmp) and os.path.isfile(filename) and  folder != 'FARIMA_ML_Models':

        ind = [i for i in range(0,len(gauge_info_data)) if gauge_info_data[i][0] == folder]
        lat = gauge_info_data[ind[0]][1]
        long = gauge_info_data[ind[0]][2]

        fid = open(filename)
        data = fid.readlines()
        fid.close()
        data_tmp = data[1].split()

        # read d vlaues for watersheds

    
        fname2 = folder + '_coefficient_estimates_mean.txt'
        filename = direc + '/' + folder + '/' + fname2
        fid = open(filename)
        data = fid.readlines()
        fid.close()
        d_tmp = data[1].split()
        
        write_data.append([folder, float(data_tmp[0]), float(data_tmp[3]), float(data_tmp[5]), float(data_tmp[6]), float(d_tmp[-2]), lat, long])

# write data to a textfile
sname = 'area_PSD_first_time_window.txt'
filename = direc + '/' + sname
fid = open(filename, 'w')
fid.write('Basin\tGreater_than_1_year\t2_weeks_to_1_month\t1_month_to_1_year\tless_than_1_month\td\tlat\tlong\n')
for wind in range(0, len(write_data)):
    fid.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n'%tuple(write_data[wind]))
fid.close()