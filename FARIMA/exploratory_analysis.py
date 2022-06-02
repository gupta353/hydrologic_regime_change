"""
Exploratory data analysis of trends in various quantities 

Author: Abhinav Gupta (Created: 31 Jan 2021)

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
fname = 'RF_regression_data_snow.txt'
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
#varNames = ['Mean runoff (mm day$^{-1}$)','Mean Rainfall (mm day$^{-1}$)','PET (mm day$^{-1}$)','Temp ($^\circ$C)','Aridity','Area (km$^2$)','Mean elevation (m)','Mean slope (m km$^{-1}$)','Fraction of forest cover','Prcipitation seasonality','Fraction of snow', 'High rainfall frequency (days yr$^{-1}$)', 'Average high rainfall event duration (days)', 'Low rainfall frequency (days yr$^{-1}$)','Average low rainfall event duration (days)', 'Trend in mean storm depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in median storm depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in total storm depth (mm time-window$^{-1}$)', 'Trend in number of rain days (time-window$^{-1}$)', 'Trend in number of storms (time-window$^{-1}$)', 'Trend in high rainfall frequency (days yr$^{-1}$ time-window$^{-1}$)', 'Trend in average high rainfall duration (days event$^{-1}$ time-window$^{-1}$)', 'Trend in average high rainfall depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in low rainfall frequency (days yr$^{-1}$ time-window$^{-1}$)', 'Trend in average low rainfall duration (days event$^{-1}$ time-window$^{-1}$)', 'Trend in average low rainfall depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in OND average rainfall depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in JFM average rainfall depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in AMJ average rainfall depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in JAS average rainfall depth (mm yr$^{-1}$ time-window$^{-1}$)','Trend in $\lambda$', 'Trends in $CN$', r'Trend in mean $\alpha/\beta$ (Minutes time-window$^{-1}$)', r'Trend in $\alpha/\beta^2$ (Minutes time-window$^{-1}$)', 'Trend in mean $\lambda$ 0-10 percentile', 'Trend in mean $\lambda$ 10-30 percentile', 'Trend in mean $\lambda$ 30-60 percentile', 'Trend in mean $\lambda$ 60-90 percentile', 'Trend in mean $\lambda$ 90-100 percentile', 'Trend in mean $CN$ 0-10 percentile', 'Trend in mean $CN$ 10-30 percentile', 'Trend in mean $CN$ 30-60 percentile', 'Trend in mean $CN$ 60-90 percentile', 'Trend in mean $CN$ 90-100 percentile', r'Trend in mean $\alpha/\beta$ 0-10 percentile', r'Trend in mean $\alpha/\beta$ 10-30 percentile', r'Trend in mean $\alpha/\beta$ 30-60 percentile', r'Trend in mean $\alpha/\beta$ 60-90 percentile', r'Trend in mean $\alpha/\beta$ 90-100 percentile', r'Trend in mean $\alpha/\beta^2$ 0-10 percentile', r'Trend in mean $\alpha/\beta^2$ 10-30 percentile', r'Trend in mean $\alpha/\beta^2$ 30-60 percentile', r'Trend in mean $\alpha/\beta^2$ 60-90 percentile', r'Trend in mean $\alpha/\beta^2$ 90-100 percentile','Trend in mean minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in median minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in median maximum daily temperature ($^\circ$C time-window$^{-1}$)','Trend in standard deviation minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in standard deviation maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean OND minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JFM minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean AMJ minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JAS minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean OND maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JFM maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean AMJ maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JAS maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum temperature 0-10 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum temperature 10-30 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily minimum temperature 30-60 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily minimum temperature 60-90 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily minimum temperature 90-100 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily maximum temperature 0-10 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily maximum temperature 10-30 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily maximum temperature 30-60 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily maximum temperature 60-90 percentile ($^\circ$C time-window$^{-1}$','Trend in mean daily maximum temperature 90-100 percentile ($^\circ$C time-window$^{-1}$', 'Trend in $\Delta T$ ($^\circ$C time-window$^{-1}$', 'Trend in $s_T$ (yr time-window$^{-1}$)', 'Trend in $\delta_P$ (time-window$^{-1}$)', 'Trend in $s_P$ (yr time-window$^{-1}$)' , 'Trend in $s_d$ (yr time-window$^{-1}$)', 'Depth to bedrock (m)', 'Soil depth (m)', 'Soil porosity', 'Soil conductivity (cm hr$^{-1}$)', 'Maximum water content (m)', 'Sand fraction (%)', 'Silt fraction (%)', 'Clay fraction (%)', 'Water fraction (%)', 'Organic frac (%)', 'Other fraction (%)', 'Latitude','Longitude']
#varNames_save = ['runoff','prcp','PET','Temp','Aridity','area','elevation','slope','frac_forest','prcp_seasonality','frac_snow','high_prcp_freq','high_prc_dur','low_prcp_freq','low_prcp_dur', 'slope_mean_depth', 'slope_median_depth', 'slope_total_depth', 'slope_num_rain_days', 'slope_num_storms', 'Slope_high_prcp_freq', 'Slope_high_prcp_dur', 'Slope_high_prcp_depth_avg', 'Slope_low_prcp_freq', 'Slope_low_prcp_dur',	'Slope_low_prcp_depth_avg', 'slope_OND_depth', 'slope_JFM_depth', 'slope_AMJ_depth', 'slope_JAS_depth','Slope_lambda', 'Slope_CN', 'Slope_mean_RT', 'Slope_std_RT', 'slope_lambda_10', 'slope_lambda_30', 'slope_lambda_60', 'slope_lambda_90', 'slope_lambda_100', 'slope_CN_10', 'slope_CN_30', 'slope_CN_60', 'slope_CN_90', 'slope_CN_100','slope_mt_10', 'slope_mt_30', 'slope_mt_60', 'slope_mt_90', 'slope_mt_100', 'slope_std_10', 'slope_std_30', 'slope_std_60', 'slope_std_90', 'slope_std_100','slope_min_temp_mean', 'slope_max_temp_mean', 'slope_min_temp_median', 'slope_max_temp_median','slope_min_temp_std', 'slope_max_temp_std', 'slope_OND_min_temp', 'slope_JFM_min_temp', 'slope_AMJ_min_temp', 'slope_JAS_min_temp', 'slope_OND_max_temp', 'slope_JFM_max_temp', 'slope_AMJ_max_temp', 'slope_JAS_max_temp', 'Slope_Min_temp_prct_1', 'Slope_Min_temp_prct_2', 'Slope_Min_temp_prct_3', 'Slope_Min_temp_prct_4', 'Slope_Min_temp_prct_5', 'Slope_Max_temp_prct_1', 'Slope_Max_temp_prct_2', 'Slope_Max_temp_prct_3', 'Slope_Max_temp_prct_4','Slope_Max_temp_prct_5', 'slope_delta_T', 'slope_s_T', 'slope_delta_P', 'slope_s_P', 'slope_s_d', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'water_frac', 'organic_frac', 'other_frac', 'Lat','long']
# varnames for snow watersheds
varNames = ['Mean runoff (mm day$^{-1}$)','Mean Rainfall (mm day$^{-1}$)','PET (mm day$^{-1}$)','Temp ($^\circ$C)','Aridity', 'BFI', 'Area (km$^2$)','Mean elevation (m)','Mean slope (m km$^{-1}$)','Fraction of forest cover','Prcipitation seasonality','Fraction of snow', 'High rainfall frequency (days yr$^{-1}$)', 'Average high rainfall event duration (days)', 'Low rainfall frequency (days yr$^{-1}$)','Average low rainfall event duration (days)', 'Trend in mean storm depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in median storm depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in total storm depth (mm time-window$^{-1}$)', 'Trend in number of rain days (time-window$^{-1}$)', 'Trend in number of storms (time-window$^{-1}$)', 'Trend in high rainfall frequency (days yr$^{-1}$ time-window$^{-1}$)', 'Trend in average high rainfall duration (days event$^{-1}$ time-window$^{-1}$)', 'Trend in average high rainfall depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in low rainfall frequency (days yr$^{-1}$ time-window$^{-1}$)', 'Trend in average low rainfall duration (days event$^{-1}$ time-window$^{-1}$)', 'Trend in average low rainfall depth (mm day$^{-1}$ time-window$^{-1}$)', 'Trend in OND average rainfall depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in JFM average rainfall depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in AMJ average rainfall depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in JAS average rainfall depth (mm yr$^{-1}$ time-window$^{-1}$)', 'Trend in mean SWE (mm time-window$^{-1}$)', 'Trend in time to peak (time-window$^{-1}$)', 'Trend in rising limb slope (mm day$^{-1} \circ C^{-1}$ time-window$^{-1}$)', 'Trend in rising limb intercept (mm day$^{-1}$ time-window$^{-1}$)','Trend in mean minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in median minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in median maximum daily temperature ($^\circ$C time-window$^{-1}$)','Trend in standard deviation minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in standard deviation maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean OND minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JFM minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean AMJ minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JAS minimum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean OND maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JFM maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean AMJ maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean JAS maximum daily temperature ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum temperature 0-10 percentile ($^\circ$C time-window$^{-1}$)', 'Trend in mean daily minimum temperature 10-30 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily minimum temperature 30-60 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily minimum temperature 60-90 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily minimum temperature 90-100 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily maximum temperature 0-10 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily maximum temperature 10-30 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily maximum temperature 30-60 percentile ($^\circ$C time-window$^{-1}$', 'Trend in mean daily maximum temperature 60-90 percentile ($^\circ$C time-window$^{-1}$','Trend in mean daily maximum temperature 90-100 percentile ($^\circ$C time-window$^{-1}$', 'Trend in $\Delta T$ ($^\circ$C time-window$^{-1}$', 'Trend in $s_T$ (yr time-window$^{-1}$)', 'Trend in $\delta_P$ (time-window$^{-1}$)', 'Trend in $s_P$ (yr time-window$^{-1}$)' , 'Trend in $s_d$ (yr time-window$^{-1}$)', 'Depth to bedrock (m)', 'Soil depth (m)', 'Soil porosity', 'Soil conductivity (cm hr$^{-1}$)', 'Maximum water content (m)', 'Sand fraction (%)', 'Silt fraction (%)', 'Clay fraction (%)', 'Water fraction (%)', 'Organic frac (%)', 'Other fraction (%)', 'Latitude','Longitude']
varNames_save = ['runoff','prcp','PET','Temp','Aridity','BFI','area','elevation','slope','frac_forest','prcp_seasonality','frac_snow','high_prcp_freq','high_prc_dur','low_prcp_freq','low_prcp_dur', 'slope_mean_depth', 'slope_median_depth', 'slope_total_depth', 'slope_num_rain_days', 'slope_num_storms', 'Slope_high_prcp_freq', 'Slope_high_prcp_dur', 'Slope_high_prcp_depth_avg', 'Slope_low_prcp_freq', 'Slope_low_prcp_dur',	'Slope_low_prcp_depth_avg', 'slope_OND_depth', 'slope_JFM_depth', 'slope_AMJ_depth', 'slope_JAS_depth','slope_swe', 'trend_tpeak', 'trend_rising_limb_slope', 'trend_rising_limb_intercept','slope_min_temp_mean', 'slope_max_temp_mean', 'slope_min_temp_median', 'slope_max_temp_median','slope_min_temp_std', 'slope_max_temp_std', 'slope_OND_min_temp', 'slope_JFM_min_temp', 'slope_AMJ_min_temp', 'slope_JAS_min_temp', 'slope_OND_max_temp', 'slope_JFM_max_temp', 'slope_AMJ_max_temp', 'slope_JAS_max_temp', 'Slope_Min_temp_prct_1', 'Slope_Min_temp_prct_2', 'Slope_Min_temp_prct_3', 'Slope_Min_temp_prct_4', 'Slope_Min_temp_prct_5', 'Slope_Max_temp_prct_1', 'Slope_Max_temp_prct_2', 'Slope_Max_temp_prct_3', 'Slope_Max_temp_prct_4','Slope_Max_temp_prct_5', 'slope_delta_T', 'slope_s_T', 'slope_delta_P', 'slope_s_P', 'slope_s_d', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'water_frac', 'organic_frac', 'other_frac', 'Lat','long']

y = farima[:,6]
p = pval[:,6]
pdiff = pval_diff[:,6]
folder = 'less_than_1_month'
y1 = y.copy()
y[y>0] = 1
y[y<0] = -1

save_direc = direc + '/FARIMA_ML_Models/' + folder
# os.mkdir(save_direc)
write_data=  []
for col_ind in range(0,len(varNames)):

    #if (col_ind != 34) and (col_ind != 48) and (col_ind != 91) and (col_ind != 92): # for rain dominated watersheds
    if (col_ind != 34) and (col_ind != 71) and (col_ind != 72) and (col_ind != 73): # for snow dominated watersheds
        x = basin_char[:,col_ind]
    
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
        plt.rc('font', **{'family': 'Arial', 'size': 12})
        plt.plot(x1,pdf1, alpha = 0.25)
        plt.plot(x2,pdf2, '-.', alpha = 0.25)
        plt.plot(x3,pdf3)
        plt.plot(x4,pdf4, '-.')
        plt.grid(linestyle = '--')
        plt.xlabel(varNames[col_ind])
        plt.ylabel('Probability density')
        plt.legend(['Positive change', 'Negative change', 'Positive change significant', 'Negative change significant'], frameon = False)

        sname = varNames_save[col_ind] + '.png'
        filename = save_direc + '/' + sname
        plt.savefig(filename, dpi = 300)
        plt.close()

        # scatter plots
        # compute slope of x, y relationship including all the watershed data
        slope, inter, rval, pval_slope, stdder = scipy.stats.linregress(x, y1)
        ind = np.nonzero( (p<=0.05) & (pdiff<=0.10))
        xind = x[ind[0]]
        y1ind = y1[ind[0]]
        slope1, inter1, rval1, pval_slope1, stdder1 = scipy.stats.linregress(xind, y1ind)
        write_data.append([varNames_save[col_ind], slope, inter, pval_slope, np.std(x),slope1, inter1, pval_slope1, np.std(xind)])

        plt.rcParams['mathtext.default'] = 'regular'
        plt.scatter(x, y1)
        plt.scatter(xind, y1ind, marker = 's')
        plt.plot(x, inter + slope*x, color = 'black')
        plt.plot(xind, inter1 + slope1*xind, linestyle = '--', color = 'black')
        plt.xlabel(varNames[col_ind])
        plt.ylabel('$\itF$')
        plt.grid(linestyle = '--')

        slope = round(slope*10**4)/10**4
        slope1 = round(slope1*10**4)/10**4
        inter = round(inter*10**4)/10**4
        if pval_slope <= 0.05 and pval_slope1 <= 0.05:
            plt.title(str(slope) + r'$^*$, ' + str(slope1) + r'$^*$')
        elif pval_slope <= 0.05 and pval_slope1 > 0.05:
            plt.title(str(slope) + r'$^*$, ' + str(slope1))
        elif pval_slope > 0.05 and pval_slope1 <= 0.05:
            plt.title(str(slope) + ', ' + str(slope1) + r'$^*$')
        else:
            plt.title(str(slope) + ', ' + str(slope1))
        
        plt.tight_layout()
        
        
        sname = varNames_save[col_ind] + '_scatter.png'
        filename = save_direc + '/' + sname
        plt.savefig(filename, dpi = 300)
        plt.close()
        
        
# write data to textfile
fname = 'trend_data.txt'
filename = save_direc + '/' + fname
fid = open(filename, 'w')
fid.write('Variable_name\tslope\tintercept\tslope_pvalue\tstd\tslope_significant\tintercept_significant\tslope_pvalue_significant\tstd_significant\n')
for wind in range(0,len(write_data)):
    fid.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n'%tuple(write_data[wind]))
fid.close()
