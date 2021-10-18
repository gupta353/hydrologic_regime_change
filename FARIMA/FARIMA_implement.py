
import os
import datetime
import numpy as np
import FarimaModule
from  statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

direc = 'D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data'
 
listFile = os.listdir(direc + '/complete_watersheds')
H = []
#read streamflow data
for fname in ['02053800_GLEAMS_CAMELS_data.txt']:#listFile:
   
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

    # calculate Hurst exponent using Aggregated Variance method
    log_m_list = np.concatenate(([0.7],np.arange(1,2.45,0.03)))
    m_list = 10**log_m_list
    m_list = m_list.astype(int)
    m_list = m_list.tolist()
    result = FarimaModule.HexpoVarAggregate(deseason_strm,m_list)
    # plot
    """     plt.scatter(log_m,log_variances)
        plt.plot([0,3],[intercept,intercept - 3])
        plt.show() """

    # apply differencing operator
    d = result[0] - 0.5
    p = 100
    y = FarimaModule.diffOp(deseason_strm,d,p)


    p_max = 5      # maixmum order of autoregressive polynomial
    q_max = 5     # maximum order of moving average polynomial
    """ order = FarimaModule.ARMAorderDetermine(y,p_max,q_max)
    print(d)
    print(order)
    p = order[0]
    q = order[1] """

    # apply ARIMA model to 10-year moving windows for fixed p, q, and d
    wlen = 3650     # length of the window (in days)
    mstep = 365*3    # time-step by which a window in moved (in days)
    N = y.shape[0] # number of days at which data is available

    model_fit = []
    for ind in range(0,N-3650+1,mstep):
        print(ind)

        y_tmp = y[ind:ind+3650]
        order = FarimaModule.ARMAorderDetermine(y_tmp,p_max,q_max)
        print(d)
        print(order)
        p = order[0]
        q = order[1]

        model = ARIMA(y_tmp,order = (p,0,q))
        model_fit.append(model.fit())

    # extract parameter values
    params_ar = []
    params_ma = []
    params_cons_sig2 = []
    params_025_ar = []
    params_025_ma = []
    params_025_cons_sig2 = []
    params_975_ar = []
    params_975_ma = []
    params_975_cons_sig2 = []
    for ind in range(0,len(model_fit)):

        p = model_fit[ind].model_orders['ar']
        q = model_fit[ind].model_orders['ma']
        conf = np.array(model_fit[ind].conf_int(0.05)) # computation of confidence interval

        if (p != 0):
            params_ar.append(list(model_fit[ind].arparams))
            params_025_ar.append(list(conf[1:p+1,0]))
            params_975_ar.append(list(conf[1:p+1,1]))

        else :
            params_ar.append([0]*p_max)
            params_025_ar.append([0]*p_max)
            params_975_ar.append([0]*p_max)

        if (q != 0):
            params_ma.append(list(model_fit[ind].maparams))
            params_025_ma.append(list(conf[p+1:-1,0]))
            params_975_ma.append(list(conf[p+1:-1,1]))

        else :
            params_ma.append([0]*q_max)
            params_025_ma.append([0]*q_max)
            params_975_ma.append([0]*q_max)
            
        params_cons_sig2.append([model_fit[ind].params[0],model_fit[ind].params[-1]])
        params_025_cons_sig2.append([conf[0,0],conf[-1,0]])      
        params_975_cons_sig2.append([conf[0,1],conf[-1,1]])


    row_length_ar = [len(params_ar[i]) for i in range(0,len(params_ar))]
    max_length_ar = max(row_length_ar)

    row_length_ma = [len(params_ma[i]) for i in range(0,len(params_ma))]
    max_length_ma = max(row_length_ma)

    for ind in range(0,len(params_ar)):
        if (len(params_ar[ind]) < max_length_ar):
            diff = max_length_ar - len(params_ar[ind]) 
            params_ar[ind] = params_ar[ind] + [0]*diff
            params_025_ar[ind] = params_025_ar[ind] + [0]*diff
            params_975_ar[ind] = params_975_ar[ind] + [0]*diff

    for ind in range(0,len(params_ma)):
        if (len(params_ma[ind]) < max_length_ma):
            diff = max_length_ma - len(params_ma[ind]) 
            params_ma[ind] = params_ma[ind] + [0]*diff
            params_025_ma[ind] = params_025_ma[ind] + [0]*diff
            params_975_ma[ind] = params_975_ma[ind] + [0]*diff

    params_ar = np.array(params_ar)
    params_ma = np.array(params_ma)
    params_025_ar = np.array(params_025_ar)
    params_025_ma = np.array(params_025_ma)
    params_975_ar = np.array(params_975_ar)
    params_975_ma = np.array(params_975_ma)
    params_cons_sig2 = np.array(params_cons_sig2)
    params_025_cons_sig2 = np.array(params_025_cons_sig2)
    params_975_cons_sig2 = np.array(params_975_cons_sig2)

    # define name of the parameters
    param_names_ar = ['AR' + str(i) for i in range(1,p_max+1)]
    param_names_ma = ['MA' + str(i) for i in range(1,p_max+1)]
    param_names = param_names_ar + param_names_ma + ['cons', 'sigma2']

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
        plt.plot(param_tmp)
        plt.plot(param_025_tmp)
        plt.plot(param_975_tmp)
        plt.title(param_names[count-1], y = 0.05, x = 0.8)

    # plot MA coefficients
    for ind in range(0,params_ma.shape[1]):

        count = count + 1
        param_tmp = params_ma[:,ind]
        param_025_tmp = params_025_ma[:,ind]
        param_975_tmp = params_975_ma[:,ind]
        plt.subplot(rows,cols,count)
        plt.plot(param_tmp)
        plt.plot(param_025_tmp)
        plt.plot(param_975_tmp)
        plt.title(param_names[count-1], y = 0.05, x = 0.8)    

    ctitle = fname.split('_')[0]
    plt.suptitle(ctitle)

    # save plot
    save_dir = 'D:/Research/non_staitionarity/codes/results/FARIMA_preliminary_results/'
    sname = ctitle + '.png'
    filename = save_dir + sname
    plt.savefig(filename)
    plt.close()