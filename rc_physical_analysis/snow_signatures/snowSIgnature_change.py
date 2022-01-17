"""
This script computes seasonal component of streamflow and temperature data, smooths the data using 30 
day moving average window, and plots streamflow vs temperature for smoothed data. Subsequently, this script
computes the slope of streamflow vs temperature curve

Author: Abhinav Gupta (Created: 12 Jan 2022)

"""
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import scipy.signal
import scipy.stats

# define a function for computing seasonal component
def seasonalCompAvg(x):
    # inputs: x = input time-series (daily scale temperature and streamflow)
    # outputs: y = seasonal component
    y = []
    for ind in range(0,365):
        y_tmp = x[range(ind,x.shape[0],365)]
        y.append(y_tmp.mean())
    return y

# define a function to compute circular running mean
def runningMean(x, wlen):
# inputs: x = time-series to be smoothed
# outputs: wlen = length of window 

    # appends values at the begining and end of the series in circular manner
    numappend = int((wlen/2))  # number of values to be apended to the left and right
    x = x[-numappend:] + x + x[0:numappend]

    y = []
    for ind in range(numappend,len(x)-numappend):
        xtmp = x[ind-numappend:ind+numappend]
        y.append(np.mean(xtmp))
    return y

# Fractional change between current and previous time-step
def time_series_change(x):
# input: x = time-series (array like)
# output: y = fractional change between previous and current time-step
    change = x[1:] - x[0:len(x)-1]
    y = change/x[1:len(x)]

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


# read the list of snow dominated watersheds
direc = 'D:/Research/non_staitionarity/codes/results/rain_snow_dominated'
fname = 'snow_dominated_watersheds_4.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
basin_list = []
for rind in range(0,len(data)):
    basin_list.append(data[rind].split()[0])

# read time-period bounds data
direc = 'D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data'
fname = 'time_period_basin_data.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
basin_time_period = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    basin_time_period.append([data_tmp[0], data_tmp[1], data_tmp[2]])

############################################################################################################

for basin in basin_list:

    # identify time-period of full data
    ind  = [i for i in range(0,len(basin_time_period)) if basin_time_period[i][0] == basin]
    begin_year = basin_time_period[ind[0]][1]
    end_year = basin_time_period[ind[0]][2]

    # identify drainage area of the basin
    ind  = [i for i in range(0,len(basin_lat_long_area)) if basin_lat_long_area[i][0] == basin]
    basin_area = basin_lat_long_area[ind[0]][3]     # in km2
    Lat = basin_lat_long_area[ind[0]][1]
    Long = basin_lat_long_area[ind[0]][2]

    # read streamflow data
    direc = 'D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data/all_watersheds'
    fname = basin + '_GLEAMS_CAMELS_data.txt'
    filename = direc + '/' + fname
    fid = open(filename, 'r')
    data = fid.readlines()
    fid.close()
    strm = []
    for wind in range(1,len(data)):
        data_tmp = data[wind].split()
        date = datetime.date(int(data_tmp[0]), int(data_tmp[1]), int(data_tmp[2]))
        strm.append([date.toordinal(), float(data_tmp[9])*0.0283])
    strm = np.array(strm)

    # extract streamflow data in the relevant time-period
    begin_date = datetime.date(int(begin_year), 10, 1)
    end_date = datetime.date(int(end_year), 9, 30)
    begin_datenum = begin_date.toordinal()
    end_datenum = end_date.toordinal()

    begin_ind = [i for i in range(0,len(strm)) if strm[i][0] == begin_datenum]
    end_ind = [i for i in range(0,len(strm)) if strm[i][0] == end_datenum]
    strm = strm[begin_ind[0]:end_ind[0]+1, 1]
    strm = strm*3600*24/basin_area/1000     # convert streamflow to mm/day

    # read temperature data
    direc_temp = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/common_directory'
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

    # extract temperature data
    ind1 = [i for i in range(0,len(date_tmax_tmin)) if date_tmax_tmin[i,0] == begin_datenum]
    ind2 = [i for i in range(0,len(date_tmax_tmin)) if date_tmax_tmin[i,0] == end_datenum]
    tmax = date_tmax_tmin[ind1[0]:ind2[0]+1,1]
    tmin = date_tmax_tmin[ind1[0]:ind2[0]+1,2]

    #  parameters of 10-year moving windows
    wlen = 3650     # length of the window (in days)
    mstep = 365*3    # time-step by which a window in moved (in days) 
    N = len(strm)
    
    window = []
    count = 0
    slope_rising_list = []
    intercept_rising_list = []
    rvalue_rising_list = []
    slope_falling_list = []
    intercept_falling_list = []
    rvalue_falling_list = []
    rising_limb_list = []
    falling_limb_list = []
    tpeak_list = []
    tavg_list = []
    strm_list = []
    for ind in range(0, N-wlen+1, mstep):
        count = count + 1
        window.append(count)

        strm_tmp = strm[ind:ind+wlen]
        tmax_tmp = tmax[ind:ind+wlen]
        tmin_tmp = tmin[ind:ind+wlen]

        # identify seasonal components of streamflow and temperature
        seasonal_strm = seasonalCompAvg(strm_tmp)
        seasonal_tmax = seasonalCompAvg(tmax_tmp)
        seasonal_tmin = seasonalCompAvg(tmin_tmp)

        # compute 30 day running mean
        strm_regime = runningMean(seasonal_strm, 30)
        tmin_regime = runningMean(seasonal_tmin, 30)
        tmax_regime = runningMean(seasonal_tmax, 30)
        tavg_regime = (np.array(tmin_regime) + np.array(tmax_regime))/2
        strm_regime = np.array(strm_regime)
        tmin_regime = np.array(tmin_regime)
        tmax_regime = np.array(tmax_regime)
        tavg_regime = np.array(tavg_regime)
        tavg_list.append(tavg_regime)
        strm_list.append(strm_regime)

        # identify day of peak
        ind_strm_max = np.nonzero(strm_regime == np.max(strm_regime))
        ind_strm_max = ind_strm_max[0][0]
        tpeak_list.append(ind_strm_max)

        # find fractional change between current and previous time-step
        frac_change = time_series_change(strm_regime)
        frac_change = runningMean(list(frac_change), 30)
        frac_change = np.array(frac_change)
        ind_peaks = scipy.signal.find_peaks(frac_change)
        ind_peaks = ind_peaks[0]

        # find nearest fractional-change-peak to the streamflow-peak 
        diff = ind_peaks - ind_strm_max
        diff[diff > -15] = -10000000000
        ind_diff = np.nonzero(diff == np.max(diff))
        ind_begin = ind_peaks[ind_diff[0][0]]

        # find negative of fractional change between current and previous time-step
        frac_change = -1*frac_change
        ind_peaks = scipy.signal.find_peaks(frac_change)
        ind_peaks = ind_peaks[0]

        # find nearest negative-fractional-change-peak to the streamflow-peak
        diff = ind_peaks - ind_strm_max
        diff[diff < 5] = 10000000000
        ind_diff = np.nonzero(diff == np.min(diff))
        ind_end = ind_peaks[ind_diff[0][0]]

        # extract temprature and streamflow regimes in the rising and falling limbs
        t_regime_1 = tavg_regime[ind_begin:ind_strm_max+1]
        t_regime_2 = tavg_regime[ind_strm_max+1:ind_end]
        strm_regime_1 = strm_regime[ind_begin:ind_strm_max+1]
        strm_regime_2 = strm_regime[ind_strm_max+1:ind_end]
        rising_limb_list.append([t_regime_1, strm_regime_1])
        falling_limb_list.append([t_regime_2, strm_regime_2])
        
        
        plt.plot(strm_regime)
        plt.plot(ind_begin, strm_regime[ind_begin], 'o')
        plt.plot(ind_end, strm_regime[ind_end],'o')
        plt.show()
        

        # compute slope of the rising limb
        """
        x = t_regime_1.reshape(1,-1)
        y = strm_regime_1.reshape(1,-1)
        slope_rising, intercept_rising, rvalue_rising, pvalue, std_err = scipy.stats.linregress(x, y)
        slope_rising_list.append(slope_rising)
        intercept_rising_list.append(intercept_rising)
        rvalue_rising_list.append(rvalue_rising)

        # compute slope of the rising limb
        x = t_regime_2.reshape(1,-1)
        y = strm_regime_2.reshape(1,-1)
        slope_falling, intercept_falling, rvalue_falling, pvalue, std_err = scipy.stats.linregress(x, y)
        slope_falling_list.append(slope_falling)
        intercept_falling_list.append(intercept_falling)
        rvalue_falling_list.append(rvalue_falling)
        """
        """
        plt.plot(strm_regime)
        plt.plot(ind_begin, strm_regime[ind_begin], 'o')
        plt.plot(ind_end, strm_regime[ind_end],'o')
        plt.show()
        """
#####################################################################################################
    # create subplots
    nrows = int((count + 3)**(0.5))
    ncols = int(np.ceil((count + 3)/nrows))
    plt.rcParams['mathtext.default'] = 'regular'

    fig, ax = plt.subplots(nrows, ncols)
    fig.set_figheight(8)
    fig.set_figwidth(15)

    # subplot 1
    for ind in range(0,len(tavg_list)):
        ax[0,0].plot(tavg_list[ind], strm_list[ind],'-o', linewidth = 0.5, markersize = 2)
    ax[0,0].legend(window, frameon = False, loc = 'upper left')
    ax[0,0].set_xlabel('Temperature ($^\circ$C)')
    ax[0,0].set_ylabel('Streamflow ($mm\ day^{-1}$)')
    
    # subplots for regression on different limbs
    pcount = -1
    while pcount < len(tavg_list)-1:
        pcount = pcount + 1
        row = int((pcount+1)/ncols)
        col = (pcount+1)/ncols - int((pcount+1)/ncols)
        col = int(np.ceil(col*ncols))
        ax[row,col].scatter(rising_limb_list[pcount][0],rising_limb_list[pcount][1], s=5)
        ax[row,col].scatter(falling_limb_list[pcount][0],falling_limb_list[pcount][1], s=5)
        ax[row,col].plot(rising_limb_list[pcount][0], intercept_rising_list[pcount] + slope_rising_list[pcount]*rising_limb_list[pcount][0])
        ax[row,col].plot(falling_limb_list[pcount][0], intercept_falling_list[pcount] + slope_falling_list[pcount]*falling_limb_list[pcount][0])
        ax[row,col].set_xlabel('Temperature ($^\circ$C)')
        ax[row,col].set_ylabel('Streamflow ($mm\ day^{-1}$)')

    # subplots of slope and intercept time-series (rising limb)
    pcount = pcount + 1
    row = int((pcount+1)/ncols)
    col = (pcount+1)/ncols - int((pcount+1)/ncols)
    col = int(np.ceil(col*ncols))
    ax1 = ax[row,col].twinx()
    ax[row,col].plot(slope_rising_list)
    ax1.plot(intercept_rising_list,'--')
    ax[row, col].legend(['Slope'], frameon = False)
    ax1.legend(['Intercept'], frameon = False)
    ax[row, col].set_xlabel('Time-window')

    # subplots of slope and intercept time-series (falling limb)
    pcount = pcount + 1
    row = int((pcount+1)/ncols)
    col = (pcount+1)/ncols - int((pcount+1)/ncols)
    col = int(np.ceil(col*ncols))
    ax2 = ax[row,col].twinx()
    ax[row,col].plot(slope_falling_list)
    ax2.plot(intercept_falling_list,'--')
    ax[row, col].legend(['Slope'], frameon = False)
    ax2.legend(['Intercept'], frameon = False)
    ax[row, col].set_xlabel('Time-window')

    fig.tight_layout()
    
    # save plot
    save_direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis'
    sname = 'streamflow_temprature_relation.svg'
    filename = save_direc + '/' + basin + '/' + sname
    plt.savefig(filename, dpi = 600)

    sname = 'streamflow_temprature_relation.png'
    filename = save_direc + '/' + basin + '/' + sname
    plt.savefig(filename, dpi = 600)
####################################################################################################################################################

    # write data to textfiles
    # write slope and intercept data
    fname = 'snow_temperature_regime_slope_intercept.txt'
    filename = save_direc + '/' + basin + '/' + fname
    fid = open(filename, 'w')
    fid.write('Rising_limb_slope(mm/day/C)\tRising_limb_intercept(mm/day)\tRising_limb_rvalue\tFalling_limb_slope(mm/day/C)\tFalling_limb_intercept(mm/day)\tFalling_limb_rvalue\n')
    for wind in range(0, len(slope_falling_list)):
        fid.write('%f\t%f\t%f\t%f\t%f\t%f\n'%(slope_rising_list[wind], intercept_rising_list[wind], rvalue_rising_list[wind], slope_falling_list[wind], intercept_falling_list[wind], rvalue_falling_list[wind]))
    fid.close()

    # write time-to-peak data
    fname = 'snow_time_to_peak.txt'
    filename = save_direc + '/' + basin + '/' + fname
    fid = open(filename, 'w')
    fid.write('Time-to_peak(days)\n')
    for wind in range(0,len(tpeak_list)):
        fid.write('%f\n'%(tpeak_list[wind]))
    fid.close()
    
    

    

    
    




