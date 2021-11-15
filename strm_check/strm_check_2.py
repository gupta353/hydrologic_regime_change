"""
This script read streamflow data from CAMELS watershed and saves data corresponding to watersheds without any missing values to the folder

Note: Only the watersheds for which streamflow data is avaialble between 1981 and 2014 is copied

Author: Abhinav Gupta (Created: 5 Oct 2021)

"""
import os
import datetime
import numpy as np
import shutil

direc = 'D:/Research/non_staitionarity/data'

# read the data basins contained in complete_watersheds_0
direc_tmp = direc + '/CAMELS_GLEAMS_combined_data/complete_watersheds_0'
list_complete_watersheds_0 = os.listdir(direc_tmp)

# read the data basins contained in complete_watersheds_1
direc_tmp = direc + '/CAMELS_GLEAMS_combined_data/complete_watersheds_1'
list_complete_watersheds_1 = os.listdir(direc_tmp)

list_complete_watersheds = list_complete_watersheds_0 + list_complete_watersheds_1

# read all the basins
listFile = os.listdir(direc + '/CAMELS_GLEAMS_combined_data/all_watersheds')

# difference between the list of two basins
remaining_basins = []
for basin in listFile:
    if (basin not in list_complete_watersheds):
        remaining_basins.append(basin)

#read streamflow data
for fname in remaining_basins:
   #   fname = '01047000_GLEAMS_CAMELS_data.txt'
    filename = direc + '/CAMELS_GLEAMS_combined_data/all_watersheds/' + fname
    fid = open(filename,'r')
    data = fid.readlines()
    fid.close()
    strm = []
    for ind in range(1,len(data)):
        data_tmp = data[ind].split()
        date_tmp = datetime.date(int(data_tmp[0]),int(data_tmp[1]),int(data_tmp[2]))
        strm.append([date_tmp.toordinal(),float(data_tmp[9])*0.028])  # multiplicative factor converting cfs to cms

    # select data to use
    begin_date = datetime.date(1981,10,1)
    end_date = datetime.date(2014,9,30)
    begin_datenum = begin_date.toordinal()
    end_datenum = end_date.toordinal()

    ind1 = [i for i in range(0,len(strm)) if (strm[i][0] == begin_datenum)]
    ind2 = [i for i in range(0,len(strm)) if (strm[i][0] == end_datenum)]

    strm = np.array(strm)
    strm_data = strm[ind1[0]:ind2[0],1]

    # find indices of negative streamflow values
    missing_ind = np.nonzero(strm_data<0)
    nans = np.isnan(strm_data)
    nan_ind = np.nonzero(nans) 

    len_missing_ind = missing_ind[0].shape[0]
    len_nan_ind = nan_ind[0].shape[0]

    if (len_missing_ind == 0 and len_nan_ind == 0):
        shutil.copy(filename,direc + '/CAMELS_GLEAMS_combined_data/complete_watersheds_2')