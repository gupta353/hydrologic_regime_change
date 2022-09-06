"""
This is a module tha contains various function to compute different rainfall statistics

Author: Abhinav Gupta (Created: 5 July 2022)

"""
import numpy as np
import datetime

# function to compute number of storms per day
def numStorms(rain):
    inds = np.nonzero(rain)
    inds = inds[0]
    diff = inds[1:]-inds[0:-1]
    storm_inds = []
    for dind in range(0,len(diff)):
        if (diff[dind]==1):
            storm_inds.append([inds[dind],inds[dind+1]])
        else:
            storm_inds.append([inds[dind]])
    
    for sind in range(1,len(storm_inds)):
        if (len(storm_inds[sind])==1 and len(storm_inds[sind-1])==2):
            storm_inds[sind] = ['NaN']

    for sind in range(len(storm_inds)-1,0,-1):
        if len(storm_inds[sind])>=2 and len(storm_inds[sind-1])==2:
            storm_inds[sind-1] = storm_inds[sind-1] + storm_inds[sind]
            storm_inds[sind] = ['NaN']

    storm_inds = [elem for elem in storm_inds if elem[0]!='NaN']
    
    num_storms = len(storm_inds)/len(rain)
    return num_storms

# function to compute mean storm depth (Average over days when prcp was greater than zero)
def meanStormDepth(rain):
    ind = np.nonzero(rain>0)
    ind = ind[0]
    mean_depth = np.nanmean(rain[ind])
    return mean_depth

# function to compute fraction of rain days
def FracRainDays(rain):
    ind = np.nonzero(rain>0)
    ind = ind[0]
    N = len(ind)/len(rain)
    return N

# function to compute frcation of high precipitation days (# of days per year when prcp >= 5*mean(prcp), Addor et al., 2017)
def high_prcp_freq(rain):
    mp = np.nanmean(rain)
    ind = np.nonzero(rain >= 5*mp)
    ind = ind[0]
    N = len(ind)/len(rain)
    return N

# function to compute fraction of low precipitation days (# of days per year when prcp < 1 mm/day, Addor et al., 2017)
def low_prcp_freq(rain):
    ind = np.nonzero(rain < 1)
    ind = ind[0]
    N = len(ind)/len(rain)
    return N

# function to compute average high precipitation duration (Addor et al., 2017)
def avg_high_prcp_duration(rain):
    mp = np.nanmean(rain)
    ind = np.nonzero(rain < 5*mp)
    ind = ind[0]
    rain_x = rain.copy()
    rain_x[ind] = 0
    rain_x[rain_x>0] = 1

    # compute number of consecutive 'ones'
    count = 0
    cont = []
    for ind in range(len(rain_x)):
        if rain_x[ind] == 1:
            count = count + 1
        else:
            cont.append(count)
            count = 0
    cont.append(count)              # it looks incorrect but it is infact correct - no worries
    cont = np.array(cont)
    ind = np.nonzero(cont>0)
    ind= ind[0]
    cont = cont[ind]

    N = np.mean(cont)
    return N

# function to compute average low precipitation duration (Addor et al., 2017)
def avg_low_prcp_duration(rain):
    ind = np.nonzero(rain >= 1)
    ind = ind[0]
    rain_x = rain.copy()
    rain_x[ind] = 10
    rain_x[rain<10] = 1
    rain_x[rain>1] = 0

    # compute number of consecutive 1's
    count = 0
    cont = []
    for ind in range(0,len(rain_x)):
        if rain_x[ind] == 1:
            count = count + 1
        else:
            cont.append(count)
            count = 0
    cont.append(count)
    cont = np.array(cont)
    ind = np.nonzero(cont>0)
    ind= ind[0]
    cont = cont[ind]

    N = np.mean(cont)
    return N

# function to compute high precipitation average
def high_prcp_avg(rain):
    mp = np.nanmean(rain)
    ind = np.nonzero(rain >= 5*mp)
    ind = ind[0]
    rain_x = rain[ind]
    avg = np.nanmean(rain_x)
    return avg

# function to compute low precipitation average
def low_prcp_avg(rain):
    ind = np.nonzero(rain < 1)
    ind = ind[0]
    rain_x = rain[ind]
    avg = np.nanmean(rain_x)
    return avg

# function to compute seasonal rainfall means
def seasonalRain(rain, datenums):

    # convert datenums to months
    mm = []
    for datenum in datenums:
        mm.append(datetime.date.fromordinal(int(datenum)).month)
    mm = np.array(mm)
    rain_x = rain.copy()
    
    # compute OND average depth
    ind = np.nonzero((mm == 10) | (mm == 11) | (mm == 12))
    ind = ind[0]
    OND_depth = np.nanmean(rain_x[ind])

    # compute JFM average depth
    ind = np.nonzero((mm == 1) | (mm == 2) | (mm == 3))
    ind = ind[0]
    JFM_depth = np.nanmean(rain_x[ind])

    # compute AMJ average depth
    ind = np.nonzero((mm == 4) | (mm == 5) | (mm == 6))
    ind = ind[0]
    AMJ_depth = np.nanmean(rain_x[ind])

    # compute JAS average depth
    ind = np.nonzero((mm == 7) | (mm == 8) | (mm == 9))
    ind = ind[0]
    JAS_depth = np.nanmean(rain_x[ind])
    
    return [OND_depth, JFM_depth, AMJ_depth, JAS_depth]