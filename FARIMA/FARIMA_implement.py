
import os
import datetime
import numpy as np
import FarimaModule
from  statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import output_Files

direc = 'D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data'
 
listFile = os.listdir(direc + '/complete_watersheds')

# define time block size for Hurst-exponent calculation as equally spaced values on log-scale
log_m_list = np.concatenate(([0.7],np.arange(1,2.45,0.03)))
m_list = 10**log_m_list
m_list = m_list.astype(int)
m_list = m_list.tolist()

#  parameters of 10-year moving windows
wlen = 3650     # length of the window (in days)
mstep = 365*3    # time-step by which a window in moved (in days)

p_max = 2      # maixmum order of autoregressive polynomial
q_max = 2     # maximum order of moving average polynomial

fs = 1        # sampling frequency for periodogram computation

#['02053800_GLEAMS_CAMELS_data.txt']
#for fname in [listFile[0]]:

def implement(fname,p_max,q_max,m_list,wlen,mstep,fs):

    print(fname)
    #read streamflow data
    filename = direc + '/complete_watersheds/' + fname
    fid = open(filename,'r')
    data = fid.readlines()
    fid.close()
    strm = []
    for ind in range(1,len(data)):
        data_tmp = data[ind].split()
        date_tmp = datetime.date(int(data_tmp[0]),int(data_tmp[1]),int(data_tmp[2]))
        strm.append([date_tmp.toordinal(),float(data_tmp[9])*0.028])  # multiplicative factor converting cfs to cms

    # select data to use
    begin_date = datetime.date(1980,10,1)
    end_date = datetime.date(2014,9,30)
    begin_datenum = begin_date.toordinal()
    end_datenum = end_date.toordinal()

    ind1 = [i for i in range(0,len(strm)) if (strm[i][0] == begin_datenum)]
    ind2 = [i for i in range(0,len(strm)) if (strm[i][0] == end_datenum)]

    strm = np.array(strm)
    strm_data = strm[ind1[0]:ind2[0],1] 

    # identify seasonal component using lowess
    frac_val = 0.02
    it_val = 2
    seasonal_comp = FarimaModule.seasonalCompLowess(strm_data,frac_val,it_val)

    # deseasonalize streamflows
    deseason_strm = FarimaModule.deseasonalize(strm_data,seasonal_comp)

    N = deseason_strm.shape[0] # number of days at which data is available

    params_windows = []
    order_windows = []
    conf_auto = []     # confidence interval estimated by statsmodles built-in method
    for ind in range(0,N-wlen+1,mstep):
        print(ind)

        data = deseason_strm[ind:ind+wlen]
        
        # calculate Hurst exponent using Aggregated Variance method
        result = FarimaModule.HexpoVarAggregate(data,m_list)
        Hvar = result[0]

        # calculate Hurst exponent using R/S method
        t_steps = range(0,3650,365)
        result = FarimaModule.HexpoRS(data,m_list,t_steps)
        H_R_by_S = result[0]

        # average H value
        H = (Hvar + H_R_by_S)/2 
        print(H)

        # apply differencing operator
        d = H - 0.5
        pt = 100
        y = FarimaModule.diffOp(data,d,pt)         # y is supposed to be ARIMA(p,0,q)

        tic = time.time()
        order = FarimaModule.ARMAorderDetermine(y,p_max,q_max)
        toc = time.time()
        print('Time taken to determine the model order :', toc-tic)
        p = order[0]
        q = order[1]

        # estimate the parameters using iterative d method
        one_sided = True
        f, I = FarimaModule.periodogramAG(data,fs,one_sided)
        f = f[1:]
        I = I[1:]
        print('Iterative d optimization begins')
        params, conf, conf_module = FarimaModule.ParEstIterd(data,p,q,one_sided,pt,f,I,d)
        params_comb = np.concatenate((params.reshape(1,-1).T,conf),axis  = 1)
        params_windows.append(params_comb)
        conf_auto.append(conf_module)
        order_windows.append([p,q])

    # extract parameter values
    params_ar, params_025_ar, params_975_ar, params_ma, params_025_ma, params_975_ma, params_d_sig, params_025_d_sig, params_975_d_sig, max_length_ar, max_length_ma = output_Files.rearrangeParams(order_windows, params_windows,p_max,q_max)

    #######################################################################################################################################
    # define name of the parameters
    param_names_ar = ['AR' + str(i) for i in range(1,max_length_ar+1)]
    param_names_ma = ['MA' + str(i) for i in range(1,max_length_ma+1)]
    param_names = param_names_ar + param_names_ma + ['d', 'sigma']
    
    ########################################################################################################################################
    # save estimated coefficients to a textfile
    station_id = fname.split('_')[0]
    save_dir = 'D:/Research/non_staitionarity/codes/results/FARIMA_results/' + station_id
    os.mkdir(save_dir)
    output_Files.outText(params_ar, params_025_ar, params_975_ar, params_ma, params_025_ma, params_975_ma, params_d_sig, params_025_d_sig, params_975_d_sig,
station_id,save_dir,param_names_ar,param_names_ma,max_length_ar,max_length_ma)

    # write the confidence interval obatined by the default method to a textfile
    output_Files.confAutoText(conf_auto,order_windows,max_length_ar, max_length_ma,param_names_ar,param_names_ma,station_id,save_dir) 
   
    ########################################################################################################################################
    # plot estimated coefficients
    numPlots = max_length_ar + max_length_ma + 2
    rows = int((numPlots)**0.5)
    cols = int(np.ceil((max_length_ar + max_length_ma + 2)/rows))
    dict = {'fontname' : 'arial', 'size' : 10}

    # plot AR coefficients
    count = 0
    for ind in range(0,params_ar.shape[1]):

        count = count + 1
        param_tmp = params_ar[:,ind]
        param_025_tmp = params_025_ar[:,ind]
        param_975_tmp = params_975_ar[:,ind]
        plt.subplot(rows,cols,count)
        plt.plot(param_tmp, color = 'blue',linewidth = 1.5)
        plt.plot(param_025_tmp, color = 'blue', linestyle = '--',linewidth = 1.5)
        plt.plot(param_975_tmp,  color = 'blue',linestyle = '--',linewidth = 1.5)
        plt.title(param_names[count-1], y = 0.05, x = 0.8)

    # plot MA coefficients
    for ind in range(0,params_ma.shape[1]):

        count = count + 1
        param_tmp = params_ma[:,ind]
        param_025_tmp = params_025_ma[:,ind]
        param_975_tmp = params_975_ma[:,ind]
        plt.subplot(rows,cols,count)
        plt.plot(param_tmp,color = 'blue',linewidth = 1.5)
        plt.plot(param_025_tmp,color = 'blue', linestyle = '--',linewidth = 1.5)
        plt.plot(param_975_tmp,color = 'blue', linestyle = '--',linewidth = 1.5)
        plt.title(param_names[count-1], y = 0.05, x = 0.8)    

    # plot d values
    count = count + 1
    param_tmp = params_d_sig[:,0]
    param_025_tmp = params_025_d_sig[:,0]
    param_975_tmp = params_975_d_sig[:,0]
    plt.subplot(rows,cols,count)
    plt.plot(param_tmp,color = 'blue',linewidth = 1.5)
    plt.plot(param_025_tmp,color = 'blue', linestyle = '--',linewidth = 1.5)
    plt.plot(param_975_tmp,color = 'blue', linestyle = '--',linewidth = 1.5)
    plt.title(param_names[count-1], y = 0.05, x = 0.8)

    # plot sigma values
    count = count + 1
    param_tmp = params_d_sig[:,1]
    param_025_tmp = params_025_d_sig[:,1]
    param_975_tmp = params_975_d_sig[:,1]
    plt.subplot(rows,cols,count)
    plt.plot(param_tmp,color = 'blue',linewidth = 1.5)
    plt.plot(param_025_tmp,color = 'blue', linestyle = '--',linewidth = 1.5)
    plt.plot(param_975_tmp,color = 'blue', linestyle = '--',linewidth = 1.5)
    plt.title(param_names[count-1], y = 0.05, x = 0.8)  

    ctitle = fname.split('_')[0]
    plt.suptitle(ctitle)

    # save plot
    sname = ctitle + '.png'
    filename = save_dir + sname
    plt.savefig(filename)
    plt.close()

    return None

# run in parallel
""" if __name__ == '__main__':
    pool = mp.Pool(4)
    results = [pool.apply(implement, args=(fname,p_max,q_max,m_list,wlen,mstep,fs)) for fname in listFile]
    pool.close() """

""" if __name__ == '__main__':
    # start 4 worker processes
    inputs = [(fname,p_max,q_max,m_list,wlen,mstep,fs) for fname in listFile]
    inputs = inputs[0:5]
    with mp.Pool(processes=5) as pool:
      tic = time.time()
      results = pool.starmap(implement,inputs)
      toc = time.time()
    print(toc-tic)
    pool.close() """

#fname = '02053800_GLEAMS_CAMELS_data.txt'
fname = listFile[0]
tic = time.time()
implement(fname,p_max,q_max,m_list,wlen,mstep,fs)
toc = time.time()
print(toc-tic)