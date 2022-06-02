"""
This script creates subplots of important predictor variables based upon the magnitude of their trend.
variables with highest trend magnitude are plotted first.

Author: Abhinav Gupta (Created: 31 Mar 2021)

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os

def normalization(x):
    xf = x.copy()
    xf = xf - np.mean(xf)
    xf = xf/np.std(xf)
    return xf

direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final'

# read data
fname = 'RF_regression_data_rc_rain.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
farima = []
pval = []
pval_diff = []
basin_char = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    farima.append(data_tmp[1:8])
    pval.append(data_tmp[8:15])
    pval_diff.append(data_tmp[15:22])
    basin_char.append(data_tmp[22:])
farima = np.array(farima)
pval = np.array(pval)
pval_diff = np.array(pval_diff)
basin_char = np.array(basin_char)
farima = farima.astype('float64')
pval = pval.astype('float64')
pval_diff = pval_diff.astype('float64')
basin_char = basin_char.astype('float64')

#####################################################################################
# varname for rain watersheds 
varNames = ['Mean runoff (mm day$^{-1}$)','Mean precipitation(mm day$^{-1}$)','PET (mm day$^{-1}$)','Temp ($^\circ$C)','Aridity','Area (km$^2$)','Mean elevation (m)','Mean slope (m km$^{-1}$)','Fraction of forest cover','Prcipitation seasonality','Fraction of snow', 'High precipitation frequency (days yr$^{-1}$)', 'Average high precipitation\n event duration (days)', 'Low precipitation\n frequency (days yr$^{-1}$)','Average low precipitation\n event duration (days)', 'Trend in mean\n storm depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in median\n storm depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in total\n storm depth (mm time-window$^{-1}$)', 'Trend in number\n of rain days (time-window$^{-1}$)', 'Trend in number\n of storms (time-window$^{-1}$)', 'Trend in high\n precipitation frequency (days yr$^{-1}$ time-window$^{-1}$)', 'Trend in average\n high precipitation duration (days event$^{-1}$ time-window$^{-1}$)', 'Trend in average\n high precipitation depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in low\n precipitation frequency (days yr$^{-1}$ time-window$^{-1}$)', 'Trend in average\n low precipitation duration (days event$^{-1}$ time-window$^{-1}$)', 'Trend in average\n low precipitation depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in OND\n average precipitation depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in JFM average\n precipitation depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in AMJ average\n precipitation depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in JAS average\n precipitation depth (mm yr$^{-1}$ time-window$^{-1}$)','Trend in mean $\lambda$', 'Trends in mean $CN$', r'Trend in mean $\alpha/\beta$ (Minutes time-window$^{-1}$)', r'Trend in mean $\alpha/\beta^2$ (Minutes time-window$^{-1}$)', 'Trend in mean $\lambda$ 0-10 percentile', 'Trend in mean $\lambda$ 10-30 percentile', 'Trend in mean $\lambda$ 30-60 percentile', 'Trend in mean $\lambda$ 60-90 percentile', 'Trend in mean $\lambda$ 90-100 percentile', 'Trend in mean $CN$ 0-10 percentile', 'Trend in mean $CN$ 10-30 percentile', 'Trend in mean $CN$ 30-60 percentile', 'Trend in mean $CN$ 60-90 percentile', 'Trend in mean $CN$ 90-100 percentile', r'Trend in mean $\alpha/\beta$\n 0-10 percentile', r'Trend in mean $\alpha/\beta$\n 10-30 percentile', r'Trend in mean $\alpha/\beta$\n 30-60 percentile', r'Trend in mean $\alpha/\beta$\n 60-90 percentile', r'Trend in mean $\alpha/\beta$\n 90-100 percentile', r'Trend in mean $\alpha/\beta^2$\n 0-10 percentile', r'Trend in mean $\alpha/\beta^2$\n 10-30 percentile', r'Trend in mean $\alpha/\beta^2$\n 30-60 percentile', r'Trend in mean $\alpha/\beta^2$\n 60-90 percentile', r'Trend in mean $\alpha/\beta^2$\n 90-100 percentile','Trend in mean minimum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean maximum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in median minimum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in median maximum\n daily temperature ($^\circ$C time-window$^{-1}$)','Trend in standard deviation\n minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in standard deviation\n maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean OND minimum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JFM\n minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean AMJ\n minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JAS\n minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean OND\n maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JFM\n maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean AMJ\n maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JAS\n maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum\n temperature 0-10 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum\n temperature 10-30 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum\n temperature 30-60 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum\n temperature 60-90 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum\n temperature 90-100 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily maximum\n temperature 0-10 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily maximum\n temperature 10-30 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily maximum\n temperature 30-60 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily maximum\n temperature 60-90 percentile ($^\circ$C time-window$^{-1}$)','Trend in mean daily maximum\n temperature 90-100 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in $\Delta T$ ($^\circ$C time-window$^{-1}$)', 'Trend in $s_T$ (yr time-window$^{-1}$)', 'Trend in $\delta_P$ (time-window$^{-1}$)', 'Trend in $s_P$ (yr time-window$^{-1}$)' , 'Trend in $s_d$ (yr time-window$^{-1}$)', 'Depth to bedrock (m)', 'Soil depth (m)', 'Soil porosity', 'Soil conductivity (cm hr$^{-1}$)', 'Maximum soil water content (m)', 'Sand fraction (%)', 'Silt fraction (%)', 'Clay fraction (%)', 'Water fraction (%)', 'Organic frac (%)', 'Other fraction (%)', 'Latitude','Longitude']
varNames_save = ['runoff','prcp','PET','Temp','Aridity','area','elevation','slope','frac_forest','prcp_seasonality','frac_snow','high_prcp_freq','high_prc_dur','low_prcp_freq','low_prcp_dur', 'slope_mean_depth', 'slope_median_depth', 'slope_total_depth', 'slope_num_rain_days', 'slope_num_storms', 'Slope_high_prcp_freq', 'Slope_high_prcp_dur', 'Slope_high_prcp_depth_avg', 'Slope_low_prcp_freq', 'Slope_low_prcp_dur',	'Slope_low_prcp_depth_avg', 'slope_OND_depth', 'slope_JFM_depth', 'slope_AMJ_depth', 'slope_JAS_depth','Slope_lambda', 'Slope_CN', 'Slope_mean_RT', 'Slope_std_RT', 'slope_lambda_10', 'slope_lambda_30', 'slope_lambda_60', 'slope_lambda_90', 'slope_lambda_100', 'slope_CN_10', 'slope_CN_30', 'slope_CN_60', 'slope_CN_90', 'slope_CN_100','slope_mt_10', 'slope_mt_30', 'slope_mt_60', 'slope_mt_90', 'slope_mt_100', 'slope_std_10', 'slope_std_30', 'slope_std_60', 'slope_std_90', 'slope_std_100','slope_min_temp_mean', 'slope_max_temp_mean', 'slope_min_temp_median', 'slope_max_temp_median','slope_min_temp_std', 'slope_max_temp_std', 'slope_OND_min_temp', 'slope_JFM_min_temp', 'slope_AMJ_min_temp', 'slope_JAS_min_temp', 'slope_OND_max_temp', 'slope_JFM_max_temp', 'slope_AMJ_max_temp', 'slope_JAS_max_temp', 'Slope_Min_temp_prct_1', 'Slope_Min_temp_prct_2', 'Slope_Min_temp_prct_3', 'Slope_Min_temp_prct_4', 'Slope_Min_temp_prct_5', 'Slope_Max_temp_prct_1', 'Slope_Max_temp_prct_2', 'Slope_Max_temp_prct_3', 'Slope_Max_temp_prct_4','Slope_Max_temp_prct_5', 'slope_delta_T', 'slope_s_T', 'slope_delta_P', 'slope_s_P', 'slope_s_d', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'water_frac', 'organic_frac', 'other_frac', 'Lat','long']
# varnames for snow watersheds
#varNames = ['Mean runoff (mm day$^{-1}$)','Mean precipitation (mm day$^{-1}$)','PET (mm day$^{-1}$)','Mean temperature ($^\circ$C)','Aridity', 'BFI','Area (km$^2$)','Mean elevation (m)','Mean slope (m km$^{-1}$)','Fraction of forest cover','Prcipitation seasonality','Fraction of snow', 'High precipitation frequency (days yr$^{-1}$)', 'Average high precipitation event duration (days)', 'Low precipitation frequency (days yr$^{-1}$)','Average low precipitation event duration (days)', 'Trend in mean storm\n depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in median storm\n depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in total storm\n depth (mm time-window$^{-1}$)', 'Trend in number of\n rain days (time-window$^{-1}$)', 'Trend in number of\n storms (time-window$^{-1}$)', 'Trend in high precipitation\n frequency (days yr$^{-1}$ time-window$^{-1}$)', 'Trend in average high\n precipitation duration (days event$^{-1}$ time-window$^{-1}$)', 'Trend in average high\n precipitation depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in low precipitation\n frequency (days yr$^{-1}$ time-window$^{-1}$)', 'Trend in average low\n precipitation duration (days event$^{-1}$ time-window$^{-1}$)', 'Trend in average low\n precipitation depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in OND average\n precipitation depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in JFM average\n precipitation depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in AMJ average\n precipitation depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in JAS average\n precipitation depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in mean\n SWE (mm time-window$^{-1}$)', 'Trend in time to peak (time-window$^{-1}$)', 'Trend in rising limb\n slope (mm day$^{-1}$ $^\circ$$C^{-1}$ time-window$^{-1}$)', 'Trend in rising limb\n intercept (mm day$^{-1}$ time-window$^{-1}$)','Trend in mean minimum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean maximum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in median minimum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in median maximum\n daily temperature ($^\circ$C time-window$^{-1}$)','Trend in standard deviation\n minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in standard deviation\n maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean OND minimum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JFM minimum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean AMJ minimum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JAS minimum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean OND maximum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JFM maximum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean AMJ maximum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JAS maximum\n daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum\n temperature 0-10 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum\n temperature 10-30 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum\n temperature 30-60 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum\n temperature 60-90 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum\n temperature 90-100 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily maximum\n temperature 0-10 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily maximum\n temperature 10-30 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily maximum\n temperature 30-60 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily maximum\n temperature 60-90 percentile ($^\circ$C time-window$^{-1}$)','Trend in mean daily maximum\n temperature 90-100 percentile ($^\circ$C time-window$^{-1})$', 'Trend in $\Delta T$ ($^\circ$C time-window$^{-1})$', 'Trend in $s_T$ (yr time-window$^{-1}$)', 'Trend in $\delta_P$ (time-window$^{-1}$)', 'Trend in $s_P$ (yr time-window$^{-1}$)' , 'Trend in $s_d$ (yr time-window$^{-1}$)', 'Depth to bedrock (m)', 'Soil depth (m)', 'Soil porosity', 'Soil conductivity (cm hr$^{-1}$)', 'Maximum water content (m)', 'Sand fraction (%)', 'Silt fraction (%)', 'Clay fraction (%)', 'Water fraction (%)', 'Organic frac (%)', 'Other fraction (%)', 'Latitude','Longitude']
#varNames_save = ['runoff','prcp','PET','Temp','Aridity', 'BFI','area','elevation','slope','frac_forest','prcp_seasonality','frac_snow','high_prcp_freq','high_prc_dur','low_prcp_freq','low_prcp_dur', 'slope_mean_depth', 'slope_median_depth', 'slope_total_depth', 'slope_num_rain_days', 'slope_num_storms', 'Slope_high_prcp_freq', 'Slope_high_prcp_dur', 'Slope_high_prcp_depth_avg', 'Slope_low_prcp_freq', 'Slope_low_prcp_dur',	'Slope_low_prcp_depth_avg', 'slope_OND_depth', 'slope_JFM_depth', 'slope_AMJ_depth', 'slope_JAS_depth','slope_swe', 'trend_tpeak', 'trend_rising_limb_slope', 'trend_rising_limb_intercept','slope_min_temp_mean', 'slope_max_temp_mean', 'slope_min_temp_median', 'slope_max_temp_median','slope_min_temp_std', 'slope_max_temp_std', 'slope_OND_min_temp', 'slope_JFM_min_temp', 'slope_AMJ_min_temp', 'slope_JAS_min_temp', 'slope_OND_max_temp', 'slope_JFM_max_temp', 'slope_AMJ_max_temp', 'slope_JAS_max_temp', 'Slope_Min_temp_prct_1', 'Slope_Min_temp_prct_2', 'Slope_Min_temp_prct_3', 'Slope_Min_temp_prct_4', 'Slope_Min_temp_prct_5', 'Slope_Max_temp_prct_1', 'Slope_Max_temp_prct_2', 'Slope_Max_temp_prct_3', 'Slope_Max_temp_prct_4','Slope_Max_temp_prct_5', 'slope_delta_T', 'slope_s_T', 'slope_delta_P', 'slope_s_P', 'slope_s_d', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'water_frac', 'organic_frac', 'other_frac', 'Lat','long']


y = farima[:,6]
p = pval[:,6]
pdiff = pval_diff[:,6]
folder = 'less_than_1_month'
y1 = y.copy()
y[y>0] = 1
y[y<0] = -1

# read linear trend data
fname = 'trend_data.txt'
save_direc = direc + '/FARIMA_ML_Models/all_rain_watersheds/' + folder 
filename = save_direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
trend_data = []
name = []
for tind in range(1, len(data)):
    data_tmp = data[tind].split()
    trend_data.append([float(x) for x in data_tmp[1:]])
    name.append(data_tmp[0])
trend_data = np.array(trend_data)
p1 = trend_data[:,2]
trend1 = trend_data[:,0]*trend_data[:,3]
trend2 = trend_data[:,4]*trend_data[:,7]
p2 = trend_data[:,6]

ind = np.nonzero((p1<0.05) | (p2<0.05))
final_trend = np.concatenate((trend2[ind[0]].reshape(1,-1).T, trend1[ind[0]].reshape(1,-1).T, ind[0].reshape(1,-1).T), axis = 1)
final_name = [name[ind[0][i]] for i in range(0,len(ind[0]))]

final_trend_abs = np.abs(final_trend)
sort_ind = np.argsort(final_trend_abs[:,0], axis = 0)
final_name = [final_name[i] for i in sort_ind]
final_name = np.flipud(final_name)
final_trend_abs = final_trend_abs[sort_ind]
final_trend_abs = np.flipud(final_trend_abs)

final_name = final_name[0:24]

# plot data
n = len(final_name)
cols = 5
rows = int(n/cols)
if rows*cols<n:
    rows = rows + 1

subpind = []
for row in range(0,rows):
    for col in range(0,cols):
        subpind.append([row,col])

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
fig, ax = plt.subplots(rows, cols, figsize = (15, 8))

for r in range(0,ax.shape[0]):
    for c in range(0,ax.shape[1]):
        ax[r,c].set_axis_off()

count = -1
for name in final_name:

    count = count  + 1
    ax_tmp = ax[subpind[count][0],subpind[count][1]]
    ax_tmp.set_axis_on()
    nind = np.nonzero(np.array(varNames_save) == name)
    x = basin_char[:,nind[0][0]]

    # kernel density
    ind = np.nonzero(y==1)
    x1 = x[ind[0]]
    x1 = np.sort(x1)
    ker1 = scipy.stats.gaussian_kde(x1, bw_method='silverman')
    pdf1 = ker1.evaluate(x1)

    ind = np.nonzero(y==-1)
    x2 = x[ind[0]]
    x2 = np.sort(x2)
    ker2 = scipy.stats.gaussian_kde(x2, bw_method='silverman')
    pdf2 = ker2.evaluate(x2)

    ind = np.nonzero((y==1) & (p<=0.05) & (pdiff<=0.10))
    x3 = x[ind[0]]
    x3 = np.sort(x3)
    ker3 = scipy.stats.gaussian_kde(x3, bw_method='silverman')
    pdf3 = ker3.evaluate(x3)
    
    ind = np.nonzero((y==-1) & (p<=0.05) & (pdiff<=0.10))
    x4 = x[ind[0]]
    x4 = np.sort(x4)
    ker4 = scipy.stats.gaussian_kde(x4, bw_method='silverman')
    pdf4 = ker4.evaluate(x4)

    # plot
    #plt.rc('font', **{'family': 'Arial', 'size': 12})
    ax_tmp.plot(x1,pdf1, alpha = 0.25)
    ax_tmp.plot(x2,pdf2, '-.', alpha = 0.25)
    ax_tmp.plot(x3,pdf3)
    ax_tmp.plot(x4,pdf4, '-.')
    ax_tmp.grid(linestyle = '--')
    ax_tmp.set_xlabel(varNames[nind[0][0]], fontsize = 8)
    # ax_tmp.set_ylabel('Probability density')
    #ax_tmp.legend(['Positive change', 'Negative change', 'Positive change significant', 'Negative change significant'], frameon = False)

fig.legend(['Positive change', 'Negative change', 'Positive change significant', 'Negative change significant'], frameon = False, bbox_to_anchor=(0.55, 0.18))    
plt.tight_layout(pad = 1, h_pad = 0.10, w_pad = 1)

# save plot
sname = 'combined.png'
filename = save_direc + '/' + sname
plt.savefig(filename, dpi = 300)

plt.show()