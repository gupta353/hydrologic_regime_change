"""
This script loads FARIMA ML models

Author: Abhinav Gupta (Created: 10 Feb 2022)

"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble
import RF_ParamTuning

direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final'
save_direc = direc
fname_model = 'less_than_1_month_new_model'
col_ind = 6                    # column index of corresponding response variable in textfile 
filename = direc + '/FARMA_ML_Models/' + fname_model

varNames = ['runoff','prcp','PET','Temp','Aridity','area','elevation','slope','frac_forest','prcp_seasonality','frac_snow','high_prcp_freq','high_prc_dur','low_prcp_freq','low_prcp_dur', 'slope_mean_depth', 'slope_total_depth', 'slope_num_rain_days', 'slope_num_storms', 'Slope_low_prcp_freq', 'Slope_low_prcp_dur', 'Slope_low_prcp_depth_avg', 'Slope_high_prcp_freq', 'Slope_high_prcp_dur',	'Slope_high_prcp_depth_avg', 'Slope_lambda', 'Slope_CN', 'Slope_mean_RT', 'Slope_std_RT', 'slope_min_temp', 'slope_max_temp', 'slope_OND_min_temp', 'slope_JFM_min_temp', 'slope_AMJ_min_temp', 'slope_JAS_min_temp', 'slope_OND_max_temp', 'slope_JFM_max_temp', 'slope_AMJ_max_temp', 'slope_JAS_max_temp', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'water_frac', 'organic_frac', 'other_frac', 'Lat','long']
model = pickle.load(open(filename,'rb'))
importances = model.feature_importances_
oob_preds = model.oob_decision_function_

# read data on predictor variables
fname = 'RF_regression_data_rc_rain.txt'
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

# 
y = farima[:,col_ind]
y[y<=0] = -1
y[y>0] = 1

import_inds  = np.array(range(0,len(importances)))
importances = importances.reshape(1,-1).T
import_inds = import_inds.reshape(1,-1).T
importances = np.concatenate((importances, import_inds) ,axis = 1)
import_sort = importances[np.argsort(importances[:,0])]
import_sort = np.flipud(import_sort)
pred_inds = import_sort[:,1]

score = []
for pind in range(0,20):
    inds = pred_inds[0:pind+1]
    inds = inds.astype('int')
    x = basin_char[:,inds]
    n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf = RF_ParamTuning.RFClassParamTuning(x,y)
    rgc = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, min_samples_leaf  = min_samples_leaf, max_features = max_features, min_samples_split = min_samples_split, max_depth = max_depth, oob_score = True, n_jobs = -1)
    rgc.fit(x,y)
    score.append(rgc.oob_score_)

plt.plot(range(1,len(score)+1), score)
plt.show()