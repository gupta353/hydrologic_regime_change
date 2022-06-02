"""
This script prepares data for RF regression

Author: Abhinav Gupta (Created: 31 Jan 2022)

"""

##############################################################
# read basin hydrometeorologic data
direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata'

fname = 'basin_annual_hydrometeorology_characteristics_daymet.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
hydmet = [] # basin, streamflow, prcp, PET, Temp
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    hydmet.append([data_tmp[1], float(data_tmp[2]), float(data_tmp[3]), float(data_tmp[4]), float(data_tmp[5])])

# read lat, long and area data
fname = 'gauge_information.txt'
filename = direc + '/' + fname
fid = open(filename,'r')
gauge_info = fid.readlines()
fid.close()
lat_long =  []
for gind in range(1,len(gauge_info)):
    data_tmp = gauge_info[gind].split('\t')
    lat_long.append([data_tmp[1],float(data_tmp[3]),float(data_tmp[4])])

# read basin physical characteristics data
fname = 'basin_physical_characteristics.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
phy_charac = [] # basin, area, mean elevation, slope, fraction of forest
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    phy_charac.append([data_tmp[1], float(data_tmp[2]), float(data_tmp[3]), float(data_tmp[4]), float(data_tmp[5])])

#######################################################################################
# read hydro data
direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/camels_attributes_v2.0/camels_attributes_v2.0'
fname = 'camels_hydro.txt'
filename = direc + '/' + fname
fid = open(filename)
data = fid.readlines()
fid.close()
BFI = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split(';')
    BFI.append([data_tmp[0], float(data_tmp[4])])
#######################################################################################
# read precipitation seasonality data
direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/camels_attributes_v2.0/camels_attributes_v2.0'
fname = 'camels_clim.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
clim = []   # prcp seasonality, frac snow, high prcp freq, high prc dur, low prcp freq, low prcp dur
for rind in range(1,len(data)):
    data_tmp = data[rind].split(';')
    clim.append([data_tmp[0], float(data_tmp[3]), float(data_tmp[4]), float(data_tmp[6]), float(data_tmp[7]), float(data_tmp[9]), float(data_tmp[10])])

# read soil data
fname = 'camels_soil.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
soil = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split(';')
    soil.append(data_tmp)
    soil[rind-1] = [soil[rind-1][0]] + [float(soil[rind-1][i]) for i in range(1,len(soil[rind-1]))]

#######################################################################################
# read rainfall-statistics-change data
direc = 'D:/Research/non_staitionarity/codes/results/rainfall_statistics'
fname = 'rainfall_statistics_change.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
rain_stat = []
for rind in  range(1,len(data)):
    data_tmp = data[rind].split()
    rain_stat.append(data_tmp)
    rain_stat[rind-1] = [rain_stat[rind-1][0]] + [float(rain_stat[rind-1][i]) for i in range(1, len(rain_stat[rind-1]))]

#######################################################################################
# read temperature-statistics-change data
direc = 'D:/Research/non_staitionarity/codes/results/temperature_statistics'
fname = 'temperature_stats_change.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
temp_stat = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    temp_stat.append([data_tmp[0]] + [float(x) for x in data_tmp[1:-2]])

#######################################################################################
# read swe-statistics-change data
direc = 'D:/Research/non_staitionarity/codes/results/snow_statistics'
fname = 'snow_statistics_change.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
snow_stat = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    snow_stat.append([data_tmp[0], float(data_tmp[1])])

########################################################################################
# read rc physical data for rainfall
direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis'
fname = 'all_rain_watershed_changes.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
rc_rain = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    rc_rain.append([data_tmp[0]] + [float(x) for x in data_tmp[1:-2]])

########################################################################################
# read snow signature change data
direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis'
fname = 'snow_signature_changes_in_time.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
snow_sig = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    snow_sig.append([data_tmp[0], float(data_tmp[1]), float(data_tmp[2]), float(data_tmp[3])])

#######################################################################################
# read climate indices data
direc = 'D:/Research/non_staitionarity/codes/results/climate_indices'
fname = 'climate_indices_change.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
clim_indices = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    clim_indices.append([data_tmp[0], float(data_tmp[2]), float(data_tmp[3]), float(data_tmp[5]), float(data_tmp[6]), float(data_tmp[7])])

#######################################################################################
# read FARIMA change data
direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final'
fname = 'changePSD_statSgfcnc.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
farima = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    farima.append([data_tmp[0]] + [float(x) for x in data_tmp[1:-2]])

# read aggregated FARIMA change data
"""
fname = 'changePSD_aggregated.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
farima_aggr = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    farima_aggr.append([data_tmp[0], float(data_tmp[2]), float(data_tmp[3])])
"""
###############################################################################################
# arrange all data in a list
# all watersheds
"""
write_data = []
for f_ind in range(0,len(farima)):
    basin = farima[f_ind][0]
    ind_fr_agg = [i for i in range(0,len(farima_aggr)) if farima_aggr[i][0]==basin]
    ind_hm = [i for i in range(0,len(hydmet)) if hydmet[i][0]==basin]
    ind_ll = [i for i in range(0,len(lat_long)) if lat_long[i][0]==basin]
    ind_ph = [i for i in range(0,len(phy_charac)) if phy_charac[i][0]==basin]
    ind_cl = [i for i in range(0,len(clim)) if clim[i][0]==basin]
    ind_rn_st = [i for i in range(0,len(rain_stat)) if rain_stat[i][0]==basin]

    write_data_tmp = [basin] + farima[f_ind][1:] + farima_aggr[ind_fr_agg[0]][1:] + hydmet[ind_hm[0]][1:] + [hydmet[ind_hm[0]][3]/hydmet[ind_hm[0]][2]] + phy_charac[ind_ph[0]][1:] + clim[ind_cl[0]][1:] + rain_stat[ind_rn_st[0]][1:5] + lat_long[ind_ll[0]][1:]
    write_data.append(write_data_tmp)
"""

# all rain watersheds
"""
write_data = []
for f_ind in range(0,len(rc_rain)):
    basin = rc_rain[f_ind][0]
    if len(basin) == 7:
        basin = '0' + basin

    ind_fr = [i for i in range(0,len(farima)) if farima[i][0]==basin]
    #ind_fr_agg = [i for i in range(0,len(farima_aggr)) if farima_aggr[i][0]==basin]
    ind_hm = [i for i in range(0,len(hydmet)) if hydmet[i][0]==basin]
    ind_ll = [i for i in range(0,len(lat_long)) if lat_long[i][0]==basin]
    ind_ph = [i for i in range(0,len(phy_charac)) if phy_charac[i][0]==basin]
    ind_cl = [i for i in range(0,len(clim)) if clim[i][0]==basin]
    ind_rn_st = [i for i in range(0,len(rain_stat)) if rain_stat[i][0]==basin]
    ind_temp_st = [i for i in range(0,len(rain_stat)) if temp_stat[i][0]==basin]
    ind_clim_indices = [i for i in range(0,len(clim_indices)) if clim_indices[i][0]==basin]
    ind_soil = [i for i in range(0,len(soil)) if soil[i][0]==basin]

    if len(ind_fr) != 0:
        write_data_tmp = [basin] + farima[ind_fr[0]][1:] + hydmet[ind_hm[0]][1:] + [hydmet[ind_hm[0]][3]/hydmet[ind_hm[0]][2]] + phy_charac[ind_ph[0]][1:] + clim[ind_cl[0]][1:] + rain_stat[ind_rn_st[0]][1:-2] + rc_rain[f_ind][1:] + temp_stat[ind_temp_st[0]][1:] + clim_indices[ind_clim_indices[0]][1:] + soil[ind_soil[0]][1:] + lat_long[ind_ll[0]][1:]
        write_data.append(write_data_tmp)
#################################################################################################

# write data to a textfile
fname = 'RF_regression_data_rc_rain.txt'
filename =  direc + '/' + fname
fid = open(filename, 'w')
fid.write('Basin\tgreater_than_1_year\t4_months_to_1_year\t1_month_to_4_months\t2_weeks_to_1_month\tless_than_2_weeks\t1_month_1_year\tless_than_1_month\tpval1\tpval2\tpval3\tpval4\tpval5\tpval6\tpval7\tpval_diff1\tpval_diff2\tpval_diff3\tpval_diff4\tpval_diff5\tpval_diff6\tpval_diff7\trunoff\tprcp\tPET\tTemp\tAridity\tarea\televation\tslope\tfrac_forest\tprcp_seasonality\tfrac_snow\thigh_prcp_freq\thigh_prc_dur\tlow_prcp_freq\tlow_prcp_dur\tslope_mean_depth\tslope_median_depth\tslope_total_depth\tslope_num_rain_days\tslope_num_storms\tSlope_high_prcp_freq\tSlope_high_prcp_dur\tSlope_high_prcp_depth_avg\tSlope_low_prcp_freq\tSlope_low_prcp_dur\tSlope_low_prcp_depth_avg\tslope_OND_depth\tSlope_JFM_depth\tSlope_AMJ_depth\tSlope_JAS_depth\tslope_lambda\tslope_CN\tslope_mean_time\tslope_std_time\tslope_lambda_10\tslope_lambda_30\tslope_lambda_60\tslope_lambda_90\tslope_lambda_100\tslope_CN_10\tslope_CN_30\tslope_CN_60\tslope_CN_90\tslope_CN_100\tslope_mt_10\tslope_mt_30\tslope_mt_60\tslope_mt_90\tslope_mt_100\tslope_std_10\tslope_std_30\tslope_std_60\tslope_std_90\tslope_std_100\tslope_min_temp_mean\tslope_max_temp_mean\tslope_min_temp_median\tslope_max_temp_median\tslope_min_temp_std\tslope_max_temp_std\tslope_OND_min_temp\tslope_JFM_min_temp\tslope_AMJ_min_temp\tslope_JAS_min_temp\tslope_OND_max_temp\tslope_JFM_max_temp\tslope_AMJ_max_temp\tslope_JAS_max_temp\tSlope_Min_temp_prct_1\tSlope_Min_temp_prct_2\tSlope_Min_temp_prct_3\tSlope_Min_temp_prct_4\tSlope_Min_temp_prct_5\tSlope_Max_temp_prct_1\tSlope_Max_temp_prct_2\tSlope_Max_temp_prct_3\tSlope_Max_temp_prct_4\tSlope_Max_temp_prct_5\tslope_delta_T\tslope_s_T\tslope_delta_P\tslope_s_P\tslope_s_d\tsoil_depth_pelletier\tsoil_depth_statsgo\tsoil_porosity\tsoil_conductivity\tmax_water_content\tsand_frac\tsilt_frac\tclay_frac\twater_frac\torganic_frac\tother_frac\tLat\tlong\n')
formatspec = '%s\t' + '%f\t'*116 + '%f\n' 
for wind in range(0, len(write_data)):
    fid.write(formatspec%tuple(write_data[wind]))
fid.close()
"""

# all snow dominated watersheds
write_data = []
for f_ind in range(0,len(snow_sig)):
    basin = snow_sig[f_ind][0]

    ind_fr = [i for i in range(0,len(farima)) if farima[i][0]==basin]
    #ind_fr_agg = [i for i in range(0,len(farima_aggr)) if farima_aggr[i][0]==basin]
    ind_hm = [i for i in range(0,len(hydmet)) if hydmet[i][0]==basin]
    ind_bfi = [i for i in range(0,len(BFI)) if BFI[i][0]==basin]
    ind_ll = [i for i in range(0,len(lat_long)) if lat_long[i][0]==basin]
    ind_ph = [i for i in range(0,len(phy_charac)) if phy_charac[i][0]==basin]
    ind_cl = [i for i in range(0,len(clim)) if clim[i][0]==basin]
    ind_rn_st = [i for i in range(0,len(rain_stat)) if rain_stat[i][0]==basin]
    ind_temp_st = [i for i in range(0,len(rain_stat)) if temp_stat[i][0]==basin]
    ind_clim_indices = [i for i in range(0,len(clim_indices)) if clim_indices[i][0]==basin]
    ind_soil = [i for i in range(0,len(soil)) if soil[i][0]==basin]
    ind_swe = [i for i in range(0,len(snow_stat)) if snow_stat[i][0]==basin]

    if len(ind_fr) != 0:
        write_data_tmp = [basin] + farima[ind_fr[0]][1:] + hydmet[ind_hm[0]][1:] + [hydmet[ind_hm[0]][3]/hydmet[ind_hm[0]][2]] + BFI[ind_bfi[0]][1:] + phy_charac[ind_ph[0]][1:] + clim[ind_cl[0]][1:] + rain_stat[ind_rn_st[0]][1:-2] + snow_stat[ind_swe[0]][1:] + snow_sig[f_ind][1:] + temp_stat[ind_temp_st[0]][1:] + clim_indices[ind_clim_indices[0]][1:] + soil[ind_soil[0]][1:] + lat_long[ind_ll[0]][1:]
        write_data.append(write_data_tmp)

# write data to a textfile
fname = 'RF_regression_data_snow.txt'
filename =  direc + '/' + fname
fid = open(filename, 'w')
fid.write('Basin\tgreater_than_1_year\t4_months_to_1_year\t1_month_to_4_months\t2_weeks_to_1_month\tless_than_2_weeks\t1_month_1_year\tless_than_1_month\tpval1\tpval2\tpval3\tpval4\tpval5\tpval6\tpval7\tpval_diff1\tpval_diff2\tpval_diff3\tpval_diff4\tpval_diff5\tpval_diff6\tpval_diff7\trunoff\tprcp\tPET\tTemp\tAridity\tBFI\tarea\televation\tslope\tfrac_forest\tprcp_seasonality\tfrac_snow\thigh_prcp_freq\thigh_prc_dur\tlow_prcp_freq\tlow_prcp_dur\tslope_mean_depth\tslope_median_depth\tslope_total_depth\tslope_num_rain_days\tslope_num_storms\tSlope_high_prcp_freq\tSlope_high_prcp_dur\tSlope_high_prcp_depth_avg\tSlope_low_prcp_freq\tSlope_low_prcp_dur\tSlope_low_prcp_depth_avg\tslope_OND_depth\tSlope_JFM_depth\tSlope_AMJ_depth\tSlope_JAS_depth\tslope_swe\ttrend_tpeak\ttrend_rising_limb_slope\ttrend_rising_limb_intercept\tslope_min_temp_mean\tslope_max_temp_mean\tslope_min_temp_median\tslope_max_temp_median\tslope_min_temp_std\tslope_max_temp_std\tslope_OND_min_temp\tslope_JFM_min_temp\tslope_AMJ_min_temp\tslope_JAS_min_temp\tslope_OND_max_temp\tslope_JFM_max_temp\tslope_AMJ_max_temp\tslope_JAS_max_temp\tSlope_Min_temp_prct_1\tSlope_Min_temp_prct_2\tSlope_Min_temp_prct_3\tSlope_Min_temp_prct_4\tSlope_Min_temp_prct_5\tSlope_Max_temp_prct_1\tSlope_Max_temp_prct_2\tSlope_Max_temp_prct_3\tSlope_Max_temp_prct_4\tSlope_Max_temp_prct_5\tslope_delta_T\tslope_s_T\tslope_delta_P\tslope_s_P\tslope_s_d\tsoil_depth_pelletier\tsoil_depth_statsgo\tsoil_porosity\tsoil_conductivity\tmax_water_content\tsand_frac\tsilt_frac\tclay_frac\twater_frac\torganic_frac\tother_frac\tLat\tlong\n')
formatspec = '%s\t' + '%f\t'*97 + '%f\n' 
for wind in range(0, len(write_data)):
    fid.write(formatspec%tuple(write_data[wind]))
fid.close()
