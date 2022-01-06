"""
This script computes seasonal component of streamflow and temperature data, smooths the data using 30 
day moving average window, and plots streamflow vs temperature for smoothed data

Also, this script computes computes the follwoing 4 metrices:
(1) Maximum SWE (regime)
(2) Ratio of maximum SWE (regime) to total streamflow (regime)
(3) ratio of rainfall (regime) to streamflow (regime)   
(4) Correlation between rainfall regime and streamflow regimes

Author: Abhinav Gupta (Created: 01 Dec 2021)

"""
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os

# define a function for computing seasonal component
def seasonalCompAvg(x):
    # inputs: x = input time-series (daily scale temperature and streamflow)
    # outputs: y = seasonal component
    y = []
    for ind in range(0,365):
        y_tmp = x[range(ind,x.shape[0],365)]
        y.append(y_tmp.mean())
    return y

# define a function to compute running mean
def runningMean(x, wlen):
# inputs: x = time-series to be smoothed
# outputs: wlen = length of window 

    # appends values at the begining and end of the series in circular manner
    numappend = int((wlen/2))  # number of zeros to be apended to the left and right
    x = x[-numappend:] + x + x[0:numappend]

    y = []
    for ind in range(numappend,len(x)-numappend):
        xtmp = x[ind-numappend:ind+numappend]
        y.append(np.mean(xtmp))
    return y

####################################################################################################################################################################################
# read basin area and lat-long data
gauge_info_direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata'
fname = 'gauge_information.txt'
filename = gauge_info_direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
basin_lat_long_area = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split('\t')
    basin_lat_long_area.append([data_tmp[1], data_tmp[3], data_tmp[4], float(data_tmp[5])])

####################################################################################################################################################################################
direc = 'D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data/complete_watersheds_0'
direc_temp = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/common_directory'
swe_direc = 'D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data/swe'
save_direc = 'D:/Research/non_staitionarity/codes/results/rain_snow_dominated'

begin_year = 1980
end_year = 2014
end_year_swe = 2011

listFile = os.listdir(direc)

swe_regime_max_list = []
swe_strm_frac_list = []
corr_rain_strm_list = []
rain_strm_frac_list = []
write_data = []
for fname in listFile:

    basin = fname.split('_')[0]

    # read streamflow data
    filename = direc + '/' + fname
    fid = open(filename, 'r')
    data = fid.readlines()
    fid.close()
    date_rain_strm = []
    for rind in range(1,len(data)):
        data_tmp = data[rind].split()
        date_tmp = datetime.date(int(data_tmp[0]),int(data_tmp[1]),int(data_tmp[2]))
        date_rain_strm.append([date_tmp.toordinal(),float(data_tmp[3]),float(data_tmp[9])*0.02832])
    date_rain_strm = np.array(date_rain_strm)

    # read temperature data
    fname_temp = basin + '_lump_cida_forcing_leap.txt'
    filename = direc_temp + '/' + fname_temp
    fid = open(filename, 'r')
    data = fid.readlines()
    fid.close()
    date_tmax_tmin = []
    for rind in range(4,len(data)):
        data_tmp = data[rind].split()
        date_tmp = datetime.date(int(data_tmp[0]),int(data_tmp[1]),int(data_tmp[2]))
        date_tmax_tmin.append([date_tmp.toordinal(), float(data_tmp[8]), float(data_tmp[9])])
    date_tmax_tmin = np.array(date_tmax_tmin)

    # read swe data
    fname = basin + '_swe.txt'
    filename = swe_direc + '/' + fname
    fid = open(filename, 'r')
    data = fid.readlines()
    fid.close()
    date_swe = []
    for rind in range(1,len(data)):
        data_tmp = data[rind].split()
        date_tmp = datetime.date(int(data_tmp[0]),int(data_tmp[1]),int(data_tmp[2]))
        date_swe.append([date_tmp.toordinal(),float(data_tmp[3])])
    date_swe = np.array(date_swe)

    # extract watershed area and lat long
    basin_ind = [i for i in range(0,len(basin_lat_long_area)) if basin_lat_long_area[i][0] == basin]
    basin_area = basin_lat_long_area[basin_ind[0]][3]      # in km2
    lat = basin_lat_long_area[basin_ind[0]][1]      
    long = basin_lat_long_area[basin_ind[0]][2]     

    # extract data in the required period
    begin_date = datetime.date(begin_year,10,1)
    end_date = datetime.date(end_year,9,30)
    begin_datenum = begin_date.toordinal()
    end_datenum = end_date.toordinal()

    # extract temperature data
    ind1 = [i for i in range(0,len(date_tmax_tmin)) if date_tmax_tmin[i,0] == begin_datenum]
    ind2 = [i for i in range(0,len(date_tmax_tmin)) if date_tmax_tmin[i,0] == end_datenum]
    tmax = date_tmax_tmin[ind1[0]:ind2[0],1]
    tmin = date_tmax_tmin[ind1[0]:ind2[0],2]

    # extract rain and streamflow data
    ind1 = [i for i in range(0,len(date_rain_strm)) if date_rain_strm[i,0] == begin_datenum]
    ind2 = [i for i in range(0,len(date_rain_strm)) if date_rain_strm[i,0] == end_datenum]
    rain = date_rain_strm[ind1[0]:ind2[0],1]
    strm = date_rain_strm[ind1[0]:ind2[0],2]

    # convert streamflow into mm/day
    strm = strm*3600*24/basin_area/1000

    # extract swe data
    end_date_swe = datetime.date(end_year_swe,9,30)
    end_datenum_swe = end_date_swe.toordinal()
    ind1 = [i for i in range(0,len(date_swe)) if date_swe[i,0] == begin_datenum]
    ind2 = [i for i in range(0,len(date_swe)) if date_swe[i,0] == end_datenum_swe]
    swe = date_swe[ind1[0]:ind2[0],1]

    # identify seasonal components of streamflow and temperature
    seasonal_strm = seasonalCompAvg(strm)
    seasonal_rain = seasonalCompAvg(rain)
    seasonal_tmax = seasonalCompAvg(tmax)
    seasonal_tmin = seasonalCompAvg(tmin)
    seasonal_swe = seasonalCompAvg(swe)

    # compute 30 day running mean
    strm_regime = runningMean(seasonal_strm, 30)
    rain_regime = runningMean(seasonal_rain, 30)
    tmin_regime = runningMean(seasonal_tmin, 30)
    tmax_regime = runningMean(seasonal_tmax, 30)
    swe_regime = runningMean(seasonal_swe, 30)
    ##############################################################################################################
    # compute mean of swe over the period (some years are missing from the data)
    swe_regime_max = np.nanmax(swe_regime)
    
    # compute fraction of total streamflow contributed by snowmelt
    tot_strm_regime = np.sum(strm_regime)
    swe_strm_frac = np.nanmax(swe_regime)/tot_strm_regime

    # compute correlation b/w rain_regim and strm_regime
    corr_rain_strm = np.corrcoef(strm_regime, rain_regime)

    # compute fraction of total streamflow contributed by snowmelt
    rain_strm_frac = np.sum(rain_regime)/tot_strm_regime   

    swe_regime_max_list.append(swe_regime_max)
    swe_strm_frac_list.append(swe_strm_frac)
    corr_rain_strm_list.append(corr_rain_strm[0,1])
    rain_strm_frac_list.append(rain_strm_frac)

    # Prepare write data
    write_data.append([basin, float(lat), float(long), swe_regime_max, swe_strm_frac, rain_strm_frac, corr_rain_strm[0,1]])
    ###############################################################################################################################################
    
    # plot data
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = '10'

    fig, ax = plt.subplots(3,1)
    fig.set_size_inches((10,6))

    # streamflow and temperature data
    ax[0].plot(range(1,len(strm_regime)+1),strm_regime)
    ax[0].set_ylabel('Streamflow regime ' + r'$(mm-day^{-1}$)', fontsize = 8)
    ax[0].set_xlabel('Day of the water year', fontsize = 8)

    ax1 = ax[0].twinx()
    ax1.plot(range(1,len(tmax_regime)+1),tmax_regime, color = 'Tab:orange')
    ax1.plot(range(1,len(tmin_regime)+1),tmin_regime, linestyle = '--', color = 'Tab:orange')
    ax1.set_ylabel('Temperature regime ' + r'$(^\circ C)$', fontsize = 8)
    ax1.legend(['Maximum temperature', 'Minimum temperature'], frameon = False, loc = 'best')

    # rain and swe data
    ax[1].plot(range(1,len(rain_regime)+1),rain_regime)
    ax[1].set_ylabel('Rainfall regime ' + r'$(mm-day^{-1})$', fontsize = 8)
    ax[1].set_xlabel('Day of the water year', fontsize = 8)

    ax2 = ax[1].twinx()
    ax2.plot(range(1,len(swe_regime)+1),swe_regime, linestyle = '--')
    ax2.set_ylabel('SWE regime' + r'$(mm)$', fontsize = 8)

    ax[2].scatter(strm_regime, tmin_regime, s = 5)
    ax[2].scatter(strm_regime, tmax_regime, s = 5)
    ax[2].legend(['Minimum temperature', 'Maximum temperature'], frameon = False, loc = 'best')
    ax[2].set_ylabel('Temperature regime ' + r'$(^\circ C)$', fontsize = 8)
    ax[2].set_xlabel('Streamflow regime' + r'$(mm-day^{-1}$)', fontsize = 8)
    
    fig.suptitle('Maximum daily SWE: ' + str(round(swe_regime_max,2)) + ' mm\n' + 'Correlation between rainfall and streamflow regime = ' + str(round(corr_rain_strm[0,1],2)) + '\n Ratio of SWE to streamflow = ' + str(round(swe_strm_frac,2)) + '\n Rain to streamflow ratio = ' + str(round(rain_strm_frac,2)))

    fig.tight_layout()

    #plt.show()
    #plt.show(block = False)
    #plt.pause(3)

    # save plot
    sname =  basin + '_snow_signature.png'
    filename = save_direc + '/snow_signature_plots/' + sname
    plt.savefig(filename, dpi = 300) 
    plt.close()
    

# write data to a textfile
"""
sname = 'rain_snow_dominated_12.txt'
filename = save_direc + '/' + sname
fid = open(filename, 'w')
fid.write('Gauge\tLat\tLong\tMaximum_swe_regime\tRatio_of_maximum_swe_to_tot_strm\tRatio_of_total_rain_to_tot_strm\tCorr_rain_strm\n')
for wind in range(0,len(write_data)):
    fid.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\n'%tuple(write_data[wind]))
fid.close()
"""
