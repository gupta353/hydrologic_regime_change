
import os
import datetime
import numpy as np
import FarimaModule
from  statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import output_Files
import plotFARIMA

direc = 'D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data'
 
listFile = os.listdir(direc + '/complete_watersheds_8')

#listFile = ['10172700_GLEAMS_CAMELS_data.txt','11237500_GLEAMS_CAMELS_data.txt','07180500_GLEAMS_CAMELS_data.txt']

""" listFile = listFile[0:300]
completed_files = os.listdir('D:/Research/non_staitionarity/codes/results/FARIMA_results')
completed_files = [completed_files[i]+'_GLEAMS_CAMELS_data.txt' for i in range(0,len(completed_files))]
listFile =  list(set(listFile) - set(completed_files)) """

# read the basin list after autocorrelation validation

fname = 'basin_list_significant_autocorr.txt'
filename = direc + '/' + fname
fid = open(filename,'r')
basin_list_data = fid.readlines()
fid.close()
basin_list = []
for basin in basin_list_data:
    basin_tmp = basin.split()[0]
    basin_tmp = basin_tmp + '_GLEAMS_CAMELS_data.txt'
    if basin_tmp in listFile:
        basin_list.append(basin_tmp)
listFile = basin_list


# define time block size for Hurst-exponent calculation as equally spaced values on log-scale
log_m_list = np.concatenate(([0.7,1],np.arange(1.06,2.45,0.03)))
m_list = 10**log_m_list
m_list = m_list.astype(int)
m_list = m_list.tolist()

#  parameters of 10-year moving windows
wlen = 3650     # length of the window (in days)
mstep = 365*3    # time-step by which a window in moved (in days)

p_max = 5      # maixmum order of autoregressive polynomial
q_max = 5     # maximum order of moving average polynomial

fs = 1        # sampling frequency for periodogram computation

begin_year = 1980
end_year = 2010
#['02053800_GLEAMS_CAMELS_data.txt']
#for fname in [listFile[0]]:

def implement(fname,p_max,q_max,m_list,wlen,mstep,fs):

    station_id = fname.split('_')[0]
    save_dir = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_reanalysis_after_autocorr/' + station_id
    os.mkdir(save_dir)

    print(fname)
    #read streamflow data
    filename = direc + '/complete_watersheds_8/' + fname
    fid = open(filename,'r')
    data = fid.readlines()
    fid.close()
    strm = []
    for ind in range(1,len(data)):
        data_tmp = data[ind].split()
        date_tmp = datetime.date(int(data_tmp[0]),int(data_tmp[1]),int(data_tmp[2]))
        strm.append([date_tmp.toordinal(),float(data_tmp[9])*0.028])  # multiplicative factor converting cfs to cms

    # select data to use
    begin_date = datetime.date(begin_year,10,1)
    end_date = datetime.date(end_year,9,30)
    begin_datenum = begin_date.toordinal()
    end_datenum = end_date.toordinal()

    ind1 = [i for i in range(0,len(strm)) if (strm[i][0] == begin_datenum)]
    ind2 = [i for i in range(0,len(strm)) if (strm[i][0] == end_datenum)]

    strm = np.array(strm)
    strm_data = strm[ind1[0]:ind2[0]+1,1] 
    datenums = strm[ind1[0]:ind2[0]+1,0]

    # identify seasonal component using lowess
    frac_val = 0.01
    it_val = 1
    seasonal_comp = FarimaModule.seasonalCompLowess(strm_data,frac_val,it_val)
    seasonal_comp_daily_avg = FarimaModule.seasonalCompAvg(strm_data) # seasonal component computed by daily averaging method

    # deseasonalize streamflows
    deseason_strm = FarimaModule.deseasonalize(strm_data,seasonal_comp)
    N = deseason_strm.shape[0] # number of days at which data is available

    # plot and save data
    plotFARIMA.plotStrmData(strm_data,seasonal_comp,seasonal_comp_daily_avg,deseason_strm,save_dir)
    output_Files.strmDataText(datenums,strm_data,seasonal_comp,deseason_strm,save_dir)

    params_windows = []
    order_windows = []
    conf_auto = []     # confidence interval estimated by statsmodles built-in method
    residuals = []
    constant_terms = []
    constant_term_conf = []
    window = 0
    for ind in range(0,N-wlen+1,mstep):
        print(ind)
        window = window + 1

        data = deseason_strm[ind:ind+wlen]
        
        # calculate Hurst exponent using Aggregated Variance method
        result = FarimaModule.HexpoVarAggregate(data,m_list)
        Hvar = result[0]
        plotFARIMA.plotVarScale(log_m_list,result[5],Hvar,result[1],result[2],save_dir,'Hexpo_AggVar_' + str(window))

        # calculate Hurst exponent using R/S method
        t_steps = range(0,wlen,365)
        result = FarimaModule.HexpoRS(data,m_list,t_steps)
        H_R_by_S = result[0]
        plotFARIMA.plotR_by_S(result[4], result[5], H_R_by_S, result[1], save_dir,'H_R_by_S_' + str(window))

        # average H value
        H = (Hvar + H_R_by_S)/2 
        print(H)

        # apply differencing operator and determine model order
        d = H - 0.5
        pt = 100
        y = FarimaModule.diffOp(data,d,pt)         # y is supposed to be ARIMA(p,0,q)

        tic = time.time()
        order = FarimaModule.ARMAorderDetermine(y,p_max,q_max)
        toc = time.time()
        print('Time taken to determine the model order :', toc-tic)
        p = order[0]
        q = order[1]
        
        # compute periodogra of the deseasonalized streamflow data
        one_sided = True
        f, I = FarimaModule.periodogramAG(data,fs,one_sided)
        f = f[1:]
        I = I[1:]
        plotFARIMA.plotPeriodogram(f,I,save_dir,'periodogram_' + str(window))

        # estimate the parameters using iterative d method
        print('Iterative d optimization begins')
        params, conf, resid, conf_module, constant_term = FarimaModule.ParEstIterd(data,p,q,one_sided,pt,f,I)
        params_comb = np.concatenate((params.reshape(1,-1).T,conf),axis  = 1)
        params_windows.append(params_comb)
        conf_auto.append(conf_module)
        order_windows.append([p,q])
        residuals.append(resid)
        constant_terms.append(constant_term)
        constant_term_conf.append(conf_module[0,:])
    
    # extract parameter values
    params_ar, params_025_ar, params_975_ar, params_ma, params_025_ma, params_975_ma, params_d_sig, params_025_d_sig, params_975_d_sig, max_length_ar, max_length_ma = output_Files.rearrangeParams(order_windows, params_windows,p_max,q_max)
    residuals = np.array(residuals).T
    #######################################################################################################################################
    # define name of the parameters
    param_names_ar = ['AR' + str(i) for i in range(1,max_length_ar+1)]
    param_names_ma = ['MA' + str(i) for i in range(1,max_length_ma+1)]
    param_names = param_names_ar + param_names_ma + ['d', 'sigma']
    
    ########################################################################################################################################
    # save estimated coefficients to a textfile
    output_Files.outText(params_ar, params_025_ar, params_975_ar, params_ma, params_025_ma, params_975_ma, params_d_sig, params_025_d_sig, params_975_d_sig, residuals, constant_terms,
station_id,save_dir,param_names_ar,param_names_ma,max_length_ar,max_length_ma)

    # write the confidence interval obatined by the default method to a textfile
    output_Files.confAutoText(conf_auto,order_windows,max_length_ar, max_length_ma,param_names_ar,param_names_ma,station_id,save_dir) 
   
   # compute residual autocorrelations and plot residual data
    acfs, acf_confints, qstats, pvalues = FarimaModule.autoCorrFarima(residuals)
    output_Files.autocorrText(acfs,qstats,pvalues,save_dir)

    # plots
    for rpind in range(0,residuals.shape[1]):
        plotFARIMA.plotResidual(residuals[:,rpind],save_dir,'residuals_plot_'+ str(rpind))
    ########################################################################################################################################
    # plot estimated coefficients
    plotFARIMA.plotParEst(save_dir,station_id)

    return None

# run in parallel
""" if __name__ == '__main__':
    pool = mp.Pool(4)
    results = [pool.apply(implement, args=(fname,p_max,q_max,m_list,wlen,mstep,fs)) for fname in listFile]
    pool.close() """

if __name__ == '__main__':
    # start 10 worker processes
    inputs = [(fname,p_max,q_max,m_list,wlen,mstep,fs) for fname in listFile]
    inputs = inputs
    with mp.Pool(processes=10) as pool:
      tic = time.time()
      results = pool.starmap(implement,inputs)
      toc = time.time()
    print(toc-tic)
    pool.close()

"""
fname = '14138900_GLEAMS_CAMELS_data.txt'
#fname = listFile[0]
tic = time.time()
implement(fname,p_max,q_max,m_list,wlen,mstep,fs)
toc = time.time()
print(toc-tic)
"""