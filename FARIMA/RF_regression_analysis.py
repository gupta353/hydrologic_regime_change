"""
This script creates RF model to identify important variables which are associated with change in hydrologic regime

Author: Abhinav Gupta (Created: 31 Jan 2021)

"""
import numpy as np
import sklearn.ensemble
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import sklearn.manifold
import pickle
import RF_ParamTuning

def normalization(x):
    x = x - np.mean(x)
    x = x/np.std(x)
    return x

direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final'

# read data
fname = 'RF_regression_data_snow.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
farima = []
basin_char = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    farima.append(data_tmp[1:8])
    basin_char.append(data_tmp[8:])
farima = np.array(farima)
basin_char = np.array(basin_char)
farima = farima.astype('float64')
basin_char = basin_char.astype('float64')

#####################################################################################
# random forest classification
# varnames for rain watersheds
#varNames = ['runoff','prcp','PET','Temp','Aridity','area','elevation','slope','frac_forest','prcp_seasonality','frac_snow','high_prcp_freq','high_prc_dur','low_prcp_freq','low_prcp_dur', 'slope_mean_depth', 'slope_median_depth', 'slope_total_depth', 'slope_num_rain_days', 'slope_num_storms', 'Slope_high_prcp_freq', 'Slope_high_prcp_dur', 'Slope_high_prcp_depth_avg', 'Slope_low_prcp_freq', 'Slope_low_prcp_dur',	'Slope_low_prcp_depth_avg', 'slope_OND_depth', 'slope_JFM_depth', 'slope_AMJ_depth', 'slope_JAS_depth','Slope_lambda', 'Slope_CN', 'Slope_mean_RT', 'Slope_std_RT', 'slope_lambda_10', 'slope_lambda_30', 'slope_lambda_60', 'slope_lambda_90', 'slope_lambda_100', 'slope_CN_10', 'slope_CN_30', 'slope_CN_60', 'slope_CN_90', 'slope_CN_100','slope_mt_10', 'slope_mt_30', 'slope_mt_60', 'slope_mt_90', 'slope_mt_100', 'slope_std_10', 'slope_std_30', 'slope_std_60', 'slope_std_90', 'slope_std_100','slope_min_temp_mean', 'slope_max_temp_mean', 'slope_min_temp_median', 'slope_max_temp_median','slope_min_temp_std', 'slope_max_temp_std', 'slope_OND_min_temp', 'slope_JFM_min_temp', 'slope_AMJ_min_temp', 'slope_JAS_min_temp', 'slope_OND_max_temp', 'slope_JFM_max_temp', 'slope_AMJ_max_temp', 'slope_JAS_max_temp', 'Slope_Min_temp_prct_1', 'Slope_Min_temp_prct_2', 'Slope_Min_temp_prct_3', 'Slope_Min_temp_prct_4', 'Slope_Min_temp_prct_5', 'Slope_Max_temp_prct_1', 'Slope_Max_temp_prct_2', 'Slope_Max_temp_prct_3', 'Slope_Max_temp_prct_4','Slope_Max_temp_prct_5', 'slope_delta_T', 'slope_s_T', 'slope_delta_P', 'slope_s_P', 'slope_s_d', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'water_frac', 'organic_frac', 'other_frac', 'Lat','long']
# varname for snow dominated watersheds
varNames = ['runoff','prcp','PET','Temp','Aridity','area','elevation','slope','frac_forest','prcp_seasonality','frac_snow','high_prcp_freq','high_prc_dur','low_prcp_freq','low_prcp_dur', 'slope_mean_depth', 'slope_median_depth', 'slope_total_depth', 'slope_num_rain_days', 'slope_num_storms', 'Slope_high_prcp_freq', 'Slope_high_prcp_dur', 'Slope_high_prcp_depth_avg', 'Slope_low_prcp_freq', 'Slope_low_prcp_dur',	'Slope_low_prcp_depth_avg', 'slope_OND_depth', 'slope_JFM_depth', 'slope_AMJ_depth', 'slope_JAS_depth','slope_swe', 'trend_tpeak', 'trend_rising_limb_slope', 'trend_rising_limb_intercept','slope_min_temp_mean', 'slope_max_temp_mean', 'slope_min_temp_median', 'slope_max_temp_median','slope_min_temp_std', 'slope_max_temp_std', 'slope_OND_min_temp', 'slope_JFM_min_temp', 'slope_AMJ_min_temp', 'slope_JAS_min_temp', 'slope_OND_max_temp', 'slope_JFM_max_temp', 'slope_AMJ_max_temp', 'slope_JAS_max_temp', 'Slope_Min_temp_prct_1', 'Slope_Min_temp_prct_2', 'Slope_Min_temp_prct_3', 'Slope_Min_temp_prct_4', 'Slope_Min_temp_prct_5', 'Slope_Max_temp_prct_1', 'Slope_Max_temp_prct_2', 'Slope_Max_temp_prct_3', 'Slope_Max_temp_prct_4','Slope_Max_temp_prct_5', 'slope_delta_T', 'slope_s_T', 'slope_delta_P', 'slope_s_P', 'slope_s_d', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'water_frac', 'organic_frac', 'other_frac', 'Lat','long']

y = farima[:,6]
x = basin_char
y[y<0] = -1
y[y>=0] = 1

# normalization of x values
"""
for ind in range(0,x.shape[1]):
    x[:,ind] = normalization(x[:,ind])
"""
# parameter tuning
n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf = RF_ParamTuning.RFClassParamTuning(x,y)

# modleing fitting with tuned parameters
rgc = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, min_samples_leaf  = min_samples_leaf, max_features = max_features, min_samples_split = min_samples_split, max_depth = max_depth, oob_score = True, n_jobs = -1)
rgc.fit(x,y)
importances = rgc.feature_importances_

# save the model
save_direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final/FARIMA_ML_Models'
sname = 'less_than_1_month_new_model'
filename = save_direc + '/' + sname 
pickle.dump(rgc, open(filename, 'wb'))

# bar plot
plt.bar(varNames,importances)
plt.xticks(rotation = 90)
plt.title(str(rgc.oob_score_))
plt.tight_layout()
plt.show()