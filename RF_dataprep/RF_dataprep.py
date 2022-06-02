"""
This script prepares rainfall, temperature, swe, and streamflow data for RF regression

Author: Abhinav Gupta (Created: 18 Apr 2022)

"""
import os
import datetime
import numpy as np

# function to read rainfall and streamflow data
def readData(filename, begin_year, end_year):

    begin_datenum = datetime.date(begin_year, 10, 1).toordinal()
    end_datenum = datetime.date(end_year, 9, 30).toordinal()
    fid = open(filename, 'r')
    data = fid.readlines()
    fid.close()
    final_data = []
    for rind in range(1, len(data)):
        data_tmp = data[rind].split()
        datenum = datetime.date(int(data_tmp[0]), int(data_tmp[1]), int(data_tmp[2])).toordinal()
        final_data.append([datenum, float(data_tmp[3]), float(data_tmp[6]), float(data_tmp[9])])
    ind1 = [i for i in range(0,len(final_data)) if final_data[i][0]==begin_datenum]
    ind2 = [i for i in range(0,len(final_data)) if final_data[i][0]==end_datenum]
    final_data = final_data[ind1[0]:ind2[0]+1]
    return final_data

# function to read SWE data
def readSWE(filename, begin_year):

    begin_datenum = datetime.date(begin_year, 10, 1).toordinal()
    end_datenum = datetime.date(2011, 9, 30).toordinal()
    fid = open(filename, 'r')
    data = fid.readlines()
    fid.close()
    swe_data = []
    for rind in range(1, len(data)):
        data_tmp = data[rind].split()
        datenum = datetime.date(int(data_tmp[0]), int(data_tmp[1]), int(data_tmp[2])).toordinal()
        swe_data.append([datenum, float(data_tmp[3])])
    ind1 = [i for i in range(0,len(swe_data)) if swe_data[i][0]==begin_datenum]
    ind2 = [i for i in range(0,len(swe_data)) if swe_data[i][0]==end_datenum]
    swe_data = swe_data[ind1[0]:ind2[0]+1]
    
    return swe_data

# function to read min-max temperatures, vaport pressure, and solar radiation data
def readEvap(filename, begin_year, end_year):

    begin_datenum = datetime.date(begin_year, 10, 1).toordinal()
    end_datenum = datetime.date(end_year, 9, 30).toordinal()
    fid = open(filename, 'r')
    data = fid.readlines()
    fid.close()
    evap_data = []
    for rind in range(4, len(data)):
        data_tmp = data[rind].split()
        datenum = datetime.date(int(data_tmp[0]), int(data_tmp[1]), int(data_tmp[2])).toordinal()
        evap_data.append([datenum, float(data_tmp[6]), float(data_tmp[8]), float(data_tmp[9]), float(data_tmp[10])])
    ind1 = [i for i in range(0,len(evap_data)) if evap_data[i][0]==begin_datenum]
    ind2 = [i for i in range(0,len(evap_data)) if evap_data[i][0]==end_datenum]
    evap_data = evap_data[ind1[0]:ind2[0]+1]
    
    return evap_data

# compute inverse cumulative sum (for rain and swe)
def cumRainSwe(met):
    x = met.copy()
    x1 = np.flipud(x)
    cum_sum = np.cumsum(x1)

    return cum_sum

# compute inverse cumulative average (for evap related variables)
def cumEvap(met):
    x = met.copy()
    x1 = np.flipud(x)
    cum_sum = np.cumsum(x1)
    for cind in range(0, len(cum_sum)):
        cum_sum[cind] = cum_sum[cind]/(cind+1)

    return cum_sum
#####################################################################################################

direc_rain = 'D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data/complete_watersheds_12'
direc_swe = 'D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data/swe'
direc_evap = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/common_directory'
direc_save = 'D:/Research/non_staitionarity/data/RF_dynamic_data_2'

begin_year = 1985
end_year = 2013

listFile = os.listdir(direc_rain)

for fname in listFile:
    
    basin = fname.split('_')[0]

    # read rainfall, and streamflow data
    filename = direc_rain + '/' + fname
    rn_swe_strm = readData(filename ,begin_year, end_year)
    rn_swe_strm = np.array(rn_swe_strm)

    # read swe data
    """
    fname_swe = basin + '_swe.txt'
    filename = direc_swe + '/' + fname_swe
    swe_data = readSWE(filename, begin_year)
    swe_data = np.array(swe_data)
    """

    # read min-max temperatures data, vapor pressure, and solar radiation data
    fname_evap = basin + '_lump_cida_forcing_leap.txt'
    filename = direc_evap + '/' + fname_evap
    evap_data = readEvap(filename, begin_year, end_year)
    evap_data = np.array(evap_data)

    # compute lagged rainfall, lagged SWE, lagged min-max temperatures, and vapor pressure as predictor variables
    rain = rn_swe_strm[:,1]
    strm = rn_swe_strm[:,3]
    datenums = rn_swe_strm[:,0]
    #swe = swe_data[:,1]
    sr = evap_data[:,1]
    tmax = evap_data[:,2]
    tmin = evap_data[:,3]
    vp = evap_data[:,4]
    write_data = []
    for ind in range(60,len(rain)):

        # lagged variables for rain 
        rain_temp = rain[ind-60:ind+1]
        cum_rain_temp = cumRainSwe(rain_temp)
        lagged_rain = cum_rain_temp[slice(0,31)]
        lagged_rain = np.concatenate((lagged_rain, [cum_rain_temp[45], cum_rain_temp[60]]), axis = 0)
        #lagged_rain = cum_rain_temp

        # lagged variables for swe
        #swe_temp = swe[ind-60:ind+1]
        #cum_swe_temp = cumRainSwe(swe_temp)
        #lagged_swe = cum_swe_temp[slice(0,31)]
        #lagged_swe = np.concatenate((lagged_swe, [cum_swe_temp[45], cum_swe_temp[60]]), axis = 0)
        #lagged_swe = cum_swe_temp

        # lagged temperature data
        tmax_temp = tmax[ind-60:ind+1]
        cum_tmax_temp = cumEvap(tmax_temp)
        lagged_tmax = cum_tmax_temp[slice(0,31)]
        lagged_tmax = np.concatenate((lagged_tmax, [cum_tmax_temp[45], cum_tmax_temp[60]]), axis = 0)
        # lagged_tmax = cum_tmax_temp

        tmin_temp = tmin[ind-60:ind+1]
        cum_tmin_temp = cumEvap(tmin_temp)
        lagged_tmin = cum_tmin_temp[slice(0,31)]
        lagged_tmin = np.concatenate((lagged_tmin, [cum_tmin_temp[45], cum_tmin_temp[60]]), axis = 0)
        #lagged_tmin = cum_tmin_temp

        # lagged vapor pressure data
        vp_temp = vp[ind-60:ind+1]
        cum_vp_temp = cumEvap(vp_temp)
        lagged_vp = cum_vp_temp[slice(0,31)]
        lagged_vp = np.concatenate((lagged_vp, [cum_vp_temp[45], cum_vp_temp[60]]), axis = 0)
        # lagged_vp = cum_vp_temp

        # lagged solar radiation data
        sr_temp = sr[ind-60:ind+1]
        cum_sr_temp = cumEvap(sr_temp)
        lagged_sr = cum_sr_temp[slice(0,31)]
        lagged_sr = np.concatenate((lagged_sr, [cum_sr_temp[45], cum_sr_temp[60]]), axis = 0)
        #lagged_sr = cum_sr_temp

        # streamflow at current time step
        strm_temp = strm[ind]

        # date
        datenum_temp = datenums[ind]
        date_temp = datetime.date.fromordinal(int(datenum_temp)) 
        date_temp = [date_temp.year, date_temp.month, date_temp.day]

        # compile all the data into one list
        write_data.append(date_temp + [strm_temp] + list(lagged_rain) + list(lagged_tmax) + list(lagged_tmin) + list(lagged_vp) + list(lagged_sr))
        
    header = ['Year','Month', 'Day'] + ['Streamflow(cfs)'] + ['rain_lag_' + str(i) for i in list(range(0,31)) + [45, 60]] + ['tmax_lag_' + str(i) for i in list(range(0,31)) + [45, 60]] + ['tmin_lag_' + str(i) for i in list(range(0,31)) + [45, 60]] + ['vp_lag_' + str(i) for i in list(range(0,31)) + [45, 60]] + ['sr_lag_' + str(i) for i in list(range(0,31)) + [45, 60]]
    
    # write data to textfile
    sname = basin + '_met_dynamic_2.txt'
    filename = direc_save + '/' + sname
    fid = open(filename, 'w')
    formatspec = '%s\t'*(len(header)-1) + '%s\n'
    fid.write(formatspec%tuple(header))
    formatspec = '%s\t'*3 + '%f\t'*(len(header)-4) + '%f\n'
    for wind in range(0,len(write_data)):
        fid.write(formatspec%tuple(write_data[wind]))
    fid.close()