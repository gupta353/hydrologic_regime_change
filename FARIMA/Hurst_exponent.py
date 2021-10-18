"""
This script contains several functions to implement FARIMA model

Author: Abhinav Gupta (Created: 22 Sep 2021)

"""

import numpy as np
import datetime
import statistics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import os
import statsmodels.api as sm
from  statsmodels.tsa.arima.model import ARIMA
import math

# identify the seasonal component from the daily data using average of the streamflows corresponding to a day
def seasonalCompAvg(strm_data):
    
    strm_avg = []
    for ind in range(0,365):
        strm_avg_tmp = strm_data[range(ind,strm_data.shape[0],365)]
        strm_avg.append(strm_avg_tmp.mean())

    # plot seasonal component of streamflow
 """    plt.plot(range(1,366),strm_avg)
    plt.show(block=False)
    plt.pause(2)
    plt.close() """

# identify the seasonal component from the daily data using LOWESS method   
def seasonalCompLowess(strm_data,frac_val,it_val):
    
    strm_day = np.array([])
    days = []
    strm_avg = []
    for ind in range(0,365):
        strm_tmp = strm_data[range(ind,strm_data.shape[0],365)]
        strm_day = np.concatenate((strm_day,strm_tmp))
        day_num = [ind+1]*strm_tmp.shape[0]
        days = days + day_num
        strm_avg.append(strm_tmp.mean())
    days = np.array(days)
      
    # LOWESS method
    lowess = sm.nonparametric.lowess
    xvals = np.array(range(1,366))
    days = days.astype('float64')
    xvals = xvals.astype('float64')
    strm_day = strm_day
    lowess_out = lowess(strm_day, days, frac=frac_val, it=it_val, delta=0.0, xvals=xvals, is_sorted=False, missing='drop', return_sorted=True)
    
    # plot seasonal component of streamflow
    """
    #plt.scatter(days,strm_day,s=0.5)
    plt.plot(xvals,lowess_out,color='black')
    plt.plot(xvals,strm_avg,color='red')
    plt.show(block = False)
    plt.pause(2)
    plt.close()
    """

# deseasonlize daily streamflows
def deseasonalize(strm_data,lowess_out):

    deseason_strm = []
    for ind in range(0,strm_data.shape[0],365):
        strm_data_tmp = strm_data[ind:ind+365]
        tmp =  strm_data_tmp - lowess_out[0:strm_data_tmp.shape[0]]
        tmp = tmp.tolist()
        deseason_strm = deseason_strm + tmp
    deseason_strm = np.array(deseason_strm)

# Calculate Hurst exponent using aggregated variance method
def HexpoVarAggregate(strm_data,m_list):
    
    variances = []
   # m_list = range(1,250,2)
   # strm_data = deseason_strm
    for m in m_list:
        avg = []
        for ind in range(0,strm_data.shape[0],m):
            tmp_data = strm_data[ind:ind+m]
            avg.append(tmp_data.mean())
        variances.append(statistics.stdev(avg)**2)

    log_variances = np.log(variances)/np.log(10)
    log_m = np.log(m_list)/np.log(10)

    # regression analysis using scipy

    log_m_reg = log_m[1:100]
    log_variances_reg = log_variances[1:100]
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_m_reg,log_variances_reg)
    Htmp = 1 + slope/2

"""     print('H = ' + str(Htmp))
    print('intercept = ' + str(intercept))
    print('standardr error = ' + str(std_err))
    print('95 confidence interval = ' + str(1 + slope/2 - 0.98*std_err) + '-' + str(1 + slope/2 + 0.98*std_err)) """

    # plot
"""     plt.scatter(log_m,log_variances)
    plt.plot([0,3],[intercept,intercept - 3])
    plt.show() """
    
    # regression analysis using sklearn
"""     log_m_reg = log_m[1:100]
    log_variances_reg = log_variances[1:100]
    log_m_reg = log_m_reg.reshape(-1,1)
    reg = LinearRegression().fit(log_m_reg,log_variances_reg)
    print('H = ' + str(1 + reg.coef_/2))
    print('intercept = ' + str(reg.intercept_))
    # plot
    plt.scatter(log_m,log_variances)
    plt.plot([0,3],[reg.intercept_,reg.intercept_ - 3])
    plt.show() """
    

    # Calculate Hurst exponent using rescaled range method
def HexpoRS(strm_data,m_list,t_steps):
    # m_list = range(4,500,2)
    # t_steps = range(0,10000,1000)
    # strm_data = deseason_strm
    Rrescaled = []
    for m in m_list:

        R_by_S_m = []       # R/s values for each time-step for fixed value of m 

        for t_steps_tmp in t_steps:
            strm_data_tmp = strm_data[t_steps_tmp:t_steps_tmp+m+1]
            strm_data_tmp_cum = strm_data_tmp.cumsum()
            dev_data = strm_data_tmp_cum[1:strm_data_tmp_cum.shape[0]] - strm_data_tmp_cum[0] # change in storage w.r.t. first time-step
            avg = dev_data[-1]/m        # average change in storage during the period of m time-steps

            # compute maximum deviation
            dev_data = dev_data - np.array(range(1,m+1,1))*avg
            R = dev_data.max()-dev_data.min()
            S = np.sqrt(np.sum((strm_data_tmp - avg)**2)/m)
            R_by_S_m.append(R/S)
        
        Rrescaled.append(R_by_S_m)
        
    log_Rrescaled = np.log(Rrescaled)/np.log(10)
    log_m = np.log(m_list)/np.log(10)

    # regression analysis using scipy
    # prepare data
    
    log_m_reg = log_m[5:90]
    log_Rrescaled_reg = log_Rrescaled[5:90,:]
    log_m_reg = log_m_reg.reshape(-1,1)
    log_m_reg = np.tile(log_m_reg,(1,len(t_steps)))
    log_m_reg = log_m_reg.reshape(log_m_reg.shape[0]*log_m_reg.shape[1],1,order='C')
    log_Rrescaled_reg = log_Rrescaled_reg.reshape(log_Rrescaled_reg.shape[0]*log_Rrescaled_reg.shape[1],1,order = 'C')

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_m_reg[:,0],log_Rrescaled_reg[:,0])
    Htmp = slope
 
# plot
"""     for ind in range(0,log_Rrescaled.shape[1]):
        plt.scatter(log_m,log_Rrescaled[:,ind])
    plt.plot([0,2.5],[intercept,intercept+2.5])
    plt.plot([0,2.5],[intercept,intercept+2.5*0.5])
    plt.title(str(slope))
    plt.show(block = False)
    plt.pause(5)
    plt.close() """

    # using average R/S values over different time-steps
"""     log_Rrescaled = np.mean(log_Rrescaled,axis=1)
    log_m_reg = log_m[5:90]
    log_Rrescaled_reg = log_Rrescaled[5:90]


    slope, intercept, r_value, p_value, std_err = stats.linregress(log_m_reg,log_Rrescaled_reg)
    #print('H = ' + str(slope))
    #print('intercept = ' + str(intercept))
    #print('standardr error = ' + str(std_err))
    #print('95 confidence interval = ' + str( slope - 1.96*std_err) + '-' + str(slope + 1.96*std_err))

    
    plt.scatter(log_m,log_Rrescaled)
    plt.plot([0,2.5],[intercept1,intercept1+2.5*0.5])
    plt.plot([0,2.5],[intercept1,intercept1+2.5])
    plt.title(str(slope1))
    plt.show(block = False)
    plt.pause(5)
    plt.close()"""

# compute the time-series as result of applying differencing operator (1-B)^d
def diffOp(strm_data,p):
    p = 100 # number of summation terms to be for differencing operator
    append_data = [0]*p
    strm_data = np.concatenate((np.array(append_data),strm_data))

    d = Htmp - 0.5

    # compute coefficients
    b_k = []
    for k in range(0,101):
        b_k.append(math.gamma(d+k)/math.gamma(d)/math.gamma(k+1))
    b_k = np.array(b_k)

    # apply filter
    y = []
    for ind in range(p,strm_data.shape[0]):
        tmp = strm_data[ind-p:ind+1]
        y.append(np.sum(tmp*b_k))
    y = np.array(y)

# apply ARMA model to differnced time-series y
# order of the model
def ARMAorderDetermine(y,p_max,q_max):

    p_max = 5      # maixmum order of autoregressive polynomial
    q_max = 5      # maximum order of moving average polynomial
    AIC_vals = []
    for p in range(1,p_max+1):
        for q in range(1,q_max+1):
            model = ARIMA(y,order = (p,0,q))
            model_fit = model.fit()
            AIC_vals.append([p,q,model_fit.aic])
    AIC_vals = np.array(AIC_vals)

    # order corresponding to minimum AIC
    AIC_min = AIC_vals[:,2].min()
    ind = np.nonzero(AIC_vals[:,2] == AIC_min)
    p = int(AIC_vals[ind,0])
    q = int(AIC_vals[ind,1])


