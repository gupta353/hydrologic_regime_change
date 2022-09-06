"""
This module contains function to compute temperature statistics

"""
import numpy as np
import datetime

# function to compute mean seasonal temperature
def seasonalTemp(temp, datenums):

    x = temp.copy()

    mm = []
    for datenum in datenums:
        mm.append(datetime.date.fromordinal(int(datenum)).month)
    mm = np.array(mm)
    
    # mean OND temperature
    ind = np.nonzero((mm==10) | (mm==11) | (mm==12))
    ind = ind[0]
    OND_temp = np.nanmean(x[ind])

    # mean JFM temperature
    ind = np.nonzero((mm==1) | (mm==2) | (mm==3))
    ind = ind[0]
    JFM_temp = np.nanmean(x[ind])

    # mean AMJ temperature
    ind = np.nonzero((mm==4) | (mm==5) | (mm==6))
    ind = ind[0]
    AMJ_temp = np.nanmean(x[ind])

    # mean JAS temperature
    ind = np.nonzero((mm==7) | (mm==8) | (mm==9))
    ind = ind[0]
    JAS_temp = np.nanmean(x[ind])

    sm = [OND_temp, JFM_temp, AMJ_temp, JAS_temp]
    return sm

# function to compute mean of each 5 prctile ranges
def prctileTemp(temp):
    x = temp.copy()
    p = [np.percentile(x,0), np.percentile(x,20), np.percentile(x,40), np.percentile(x,60), np.percentile(x,80), np.percentile(x,100)]
    tm = [] # mean of percentiles
    for ind in range(0,len(p)-1):
        tmp_inds = np.nonzero((x>=p[ind]) & (x<p[ind+1]))
        tmp_inds = tmp_inds[0]
        tm.append(np.nanmean(x[tmp_inds]))
    return tm