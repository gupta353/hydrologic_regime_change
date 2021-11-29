"""
This script contains several functions to implement FARIMA model

Author: Abhinav Gupta (Created: 22 Sep 2021)

"""

import numpy as np
import statistics
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from  statsmodels.tsa.arima.model import ARIMA
import math
from scipy import signal
import scipy.special as sc
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

# identify the seasonal component from the daily data using average of the streamflows corresponding to a day
def seasonalCompAvg(strm_data):
    
    strm_avg = []
    for ind in range(0,365):
        strm_avg_tmp = strm_data[range(ind,strm_data.shape[0],365)]
        strm_avg.append(strm_avg_tmp.mean())
    return strm_avg

# identify the seasonal component from the daily data using LOWESS method   
def seasonalCompLowess(strm_data,frac_val,it_val):
    # inputs: strm_data = array-like containing streamflow data
    #         frac_val = fraction of sampled to be used for local regression
    #         it_val = number of lowess iterations
    # outputs: lowess_out = Averaged values obtained by lowess

    # rearrange data according to days 
    strm_day = np.array([])
    days = []
    for ind in range(0,365):
        strm_tmp = strm_data[range(ind,strm_data.shape[0],365)]
        strm_day = np.concatenate((strm_day,strm_tmp))
        day_num = [ind+1]*strm_tmp.shape[0]
        days = days + day_num
    days = np.array(days)
      
    # LOWESS method
    lowess = sm.nonparametric.lowess
    xvals = np.array(range(1,366))
    days = days.astype('float64')
    xvals = xvals.astype('float64')
    lowess_out = lowess(strm_day, days, frac=frac_val, it=it_val, delta=0.0, xvals=xvals, is_sorted=False, missing='drop', return_sorted=True)
    
    return lowess_out

# deseasonlize daily streamflows
def deseasonalize(strm_data,lowess_out):
    # inputs: strm_data = streamflow data
    #         lowess_out = seasonal component of the streamflow data
    # outputs: deseason_strm = deseasonalized streamflow data
    deseason_strm = []
    for ind in range(0,strm_data.shape[0],365):
        strm_data_tmp = strm_data[ind:ind+365]
        tmp =  strm_data_tmp - lowess_out[0:strm_data_tmp.shape[0]]
        tmp = tmp.tolist()
        deseason_strm = deseason_strm + tmp
    deseason_strm = np.array(deseason_strm)

    return deseason_strm

# Calculate Hurst exponent using aggregated variance method
def HexpoVarAggregate(strm_data,m_list):
    # inputs: strm_data = data for which Hurst exponent is to be computed (deseasonalized streamflows for daily scale data)
    #         m_list = number of aggregation time-steps which will be used for computation of Hurst exponent
    # outputs: H = Hurst exponent
    #          slope = slope of the fitted regression line
    #          intercept = intercept of the regression analysis
    #          H_025 = lower limit of the 95% confidence interval
    #          H_975 = upper limit of 95% confidence interval
    #          log_variances = logarithm (base 10) of variances at different aggregation scales

    variances = []
    for m in m_list:
        avg = []
        for ind in range(0,strm_data.shape[0],m):
            tmp_data = strm_data[ind:ind+m]
            avg.append(tmp_data.mean())
        variances.append(statistics.stdev(avg)**2)

    log_variances = np.log(variances)/np.log(10)
    log_m = np.log(m_list)/np.log(10)

    # regression analysis using scipy
    log_m_reg = log_m[1:]
    log_variances_reg = log_variances[1:]
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_m_reg,log_variances_reg)
    H = 1 + slope/2
    H_025 = H - 0.98*std_err
    H_975 = H + 0.98*std_err

    return H, slope, intercept, H_025, H_975, log_variances

# Calculate Hurst exponent using R/S method
def HexpoRS(strm_data,m_list,t_steps):
    # inputs:  strm_data = data for which hurst exponent is to be computed (deseasonalized streamflows for daily scale data)
    #          m_list =    number of aggregation time-steps
    #          t_steps =   time-steps at which R/S statistic will be computed
    # outputs: H =        Hurst exponent
    #          intercept = intercept of regression analysis
    #          H_025 = lower limit of the 95% confidence interval
    #          H_975 = upper limit of 95% confidence interval

    Rrescaled = []
    for m in m_list:

        R_by_S_m = []       # R/S values for each time-step for fixed value of m 

        for t_steps_tmp in t_steps:
            strm_data_tmp = strm_data[t_steps_tmp:t_steps_tmp+m+1]
            strm_data_tmp_cum = strm_data_tmp.cumsum()
            dev_data = strm_data_tmp_cum[1:strm_data_tmp_cum.shape[0]] - strm_data_tmp_cum[0] # change in storage w.r.t. first time-step
            avg = dev_data[-1]/m        # average change in storage during the period of m time-steps

            # compute maximum deviation
            dev_data = dev_data - np.array(range(1,m+1,1))*avg
            R = dev_data.max()-dev_data.min()
            S = np.sqrt(np.sum((strm_data_tmp[1:] - avg)**2)/m)
            R_by_S_m.append(R/S)
        
        Rrescaled.append(R_by_S_m)
        
    log_Rrescaled = np.log(Rrescaled)/np.log(10)
    log_m = np.log(m_list)/np.log(10)

    # regression analysis using scipy
    # prepare data
    
    log_m_reg = log_m[1:]
    log_Rrescaled_reg = log_Rrescaled[1:,:]
    log_m_reg = log_m_reg.reshape(-1,1)
    log_m_reg = np.tile(log_m_reg,(1,len(t_steps)))
    log_m_reg = log_m_reg.reshape(log_m_reg.shape[0]*log_m_reg.shape[1],1,order='C')
    log_Rrescaled_reg = log_Rrescaled_reg.reshape(log_Rrescaled_reg.shape[0]*log_Rrescaled_reg.shape[1],1,order = 'C')

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_m_reg[:,0],log_Rrescaled_reg[:,0])
    H = slope
    H_025 = H - 1.96*std_err
    H_975 = H + 1.96*std_err
    
    return H, intercept, H_025, H_975, log_m_reg, log_Rrescaled_reg

# compute the time-series as result of applying differencing operator (1-B)^d
def diffOp(strm_data,d,p):
    # inputs:  strm_data = data for which hurst exponent is to be computed (deseasonalized streamflows for daily scale data)
    #          d = order of the differencing operator (between 0 and 0.5 to model long-term persistence)
    #          p = number of previous time-steps to be used for the computation
    # outputs: y = time-series after applying differencing operator (y = (1-B)^d*strm_data)

    append_data = [0]*p
    strm_data = np.concatenate((np.array(append_data),strm_data))

    # compute coefficients
    b_k = []
    for k in range(0,p + 1):
        if (-d+k < 0):
            b_k.append(math.gamma(-d+k)/math.gamma(-d)/math.gamma(k+1))
        else:
            b_k.append((np.exp(sc.gammaln(-d+k) - sc.gammaln(k+1)))/math.gamma(-d))
    b_k = np.array(b_k, dtype = object)

    # apply filter
    y = []
    for ind in range(p,strm_data.shape[0]):
        tmp = strm_data[ind-p:ind+1]
        tmp = tmp.reshape(1,-1)
        tmp = np.fliplr(tmp)
        y.append(np.sum(tmp*b_k))
    y = np.array(y)

    return y

# compute the time-series as result of applying inverse of differencing operator, that is, (1-B)^(-d)
def invdiffOp(strm_data,d,p):
    # inputs:  strm_data = data for which hurst exponent is to be computed (deseasonalized streamflows for daily scale data)
    #          d = order of the differencing operator (between 0 and 0.5 to model long-term persistence)
    #          p = number of previous time-steps to be used for the computation
    # outputs: y = time-series after applying differencing operator (y = (1-B)^(-d)*strm_data)

    append_data = [0]*p
    strm_data = np.concatenate((np.array(append_data),strm_data))

    # compute coefficients
    b_k = []
    for k in range(0,p + 1):
        #b_k.append(math.gamma(d+k)/math.gamma(d)/math.gamma(k+1))
        b_k.append((np.exp(sc.gammaln(d+k) - sc.gammaln(k+1)))/math.gamma(d))
    b_k = np.array(b_k, dtype = object)

    # apply filter
    y = []
    for ind in range(p,strm_data.shape[0]):
        tmp = strm_data[ind-p:ind+1]
        tmp = tmp.reshape(1,-1)
        tmp = np.fliplr(tmp)
        y.append(np.sum(tmp*b_k))
    y = np.array(y)

    return y

# Apply ARMA model [ARIMA(p,d=0,q)] order of the differnced time-series using AIC  
def ARMAorderDetermine(y,p_max,q_max):
    # inputs: y = time-series to which ARMA model is to be applied
    #         p_max = maximum order of the auto-regressive polynomial
    #         q_max = maximum order of the moving-average polynomial
    # outputs: p = optimal order of auto-regressive polynomial
    #          q = optimal order of the moving average polynomial
    
    AIC_vals = []
    for p in range(0,p_max+1):
        for q in range(0,q_max+1):
            print(p,q)
            model = ARIMA(y,order = (p,0,q),enforce_stationarity=True)
            model.initialize_approximate_diffuse()
            model_fit = model.fit(method_kwargs={'maxiter': 5000})
            #print('**************************************************************')
            #print(model_fit.mle_retvals)
            #print('**************************************************************')
            AIC_vals.append([p,q,model_fit.aic])
    AIC_vals = np.array(AIC_vals)
    
    # order corresponding to minimum AIC
    AIC_min = AIC_vals[:,2].min()
    ind = np.nonzero(AIC_vals[:,2] == AIC_min)
    p = int(AIC_vals[ind,0])
    q = int(AIC_vals[ind,1])
    
    if (p == p_max or q==q_max):
        AIC_vals = AIC_vals.tolist()
        for p in range(0,11):
            for q in range(6,11):
                print(p,q)
                model = ARIMA(y,order = (p,0,q),enforce_stationarity=True)
                model.initialize_approximate_diffuse()
                model_fit = model.fit(method_kwargs={'maxiter': 5000})
                AIC_vals.append([p,q,model_fit.aic])
                #print('**************************************************************')
                #print(model_fit.mle_retvals)
                #print('**************************************************************')
        for p in range(6,11):
            for q in range(0,6):
                print(p,q) 
                model = ARIMA(y,order = (p,0,q),enforce_stationarity=True)
                model.initialize_approximate_diffuse()
                model_fit = model.fit(method_kwargs={'maxiter': 5000})
                AIC_vals.append([p,q,model_fit.aic])
                #print('**************************************************************')
                #print(model_fit.mle_retvals)
                #print('**************************************************************')
    AIC_vals = np.array(AIC_vals)
    
    AIC_min = AIC_vals[:,2].min()
    ind = np.nonzero(AIC_vals[:,2] == AIC_min)
    p = int(AIC_vals[ind,0])
    q = int(AIC_vals[ind,1])
    
    return p,q

# computation of periodogram of time-series
def periodogramAG(x,fs,one_sided):
    # inputs: x  = time-series of which periodogram is to be computed
    #         fs = sampling frequency (number of samples per unit time)
    #         one_sided = True or False (If 'True' then spectra at only non-negative values will be returned)
    f, pxx_den = signal.periodogram(x,fs,return_onesided = one_sided)

    return f, pxx_den

# Autoregressive polynomial
def autoregressPol(p,ar,x):
    # inputs:  p = order of auto-regressive polynomial
    #          x = input values at which polynomial is to be computed
    #          ar = parameters of AR polynomial inlcuding the zero lag term 
    #              (ar polynomial: phi_p(B) = 1 - \sum_{j=1}^{p}\phi_j*B^j; the AR model: X_t = \sum_{j=1}^{p}\phi_j*X_(t-j); the array 'ar'
    #                should contain equal to negative of phi values)
    # output: phi_p = value of autoregressive polynomial at values contained in x

        power = np.arange(0,p+1)    # power of exponential
        phi_p = []
        for ind in range(0,x.shape[0]):
            vals = x[ind]**power
            vals = ar*vals 
            phi_p.append(np.sum(vals))
        phi_p = np.array(phi_p)

        return phi_p

# moving average polynomial
def movingAvgPol(q,ma,x):
    # inputs:  q = order of moving-average polynomial
    #          x = input values at which polynomial is to be computed
    #          ma = parameters of MA polynomial
    # output: shi_q = value of autoregressive polynomial at values contained in x

        power = np.arange(0,q+1)    # power of exponential
        shi_q = []
        for ind in range(0,x.shape[0]):
            vals = x[ind]**power
            vals = ma*vals 
            shi_q.append(np.sum(vals))
        shi_q = np.array(shi_q)

        return shi_q

# theoretical power spectral density of FARIMA model
def pxx_denFARIMA(p,d,q,ar,ma,f,sigma_eps,one_sided):

    # inputs: p = order of autoregressive polynomial
    #         q = order of moving-average polynomial
    #         d = differencing order
    #         ar = parameters of AR polynomial inlcuding the zero lag term 
    #              (ar polynomial: phi_p(B) = 1 - \sum_{j=1}^{p}\phi_j*B^j; the AR model: X_t = \sum_{j=1}^{p}\phi_j*X_(t-j))
    #         ma = parameters of the MA polynomial including the zero lag term
    #         f = frequencies at which the power spectral density is to be computed (in cycles per unit time)
    #         sigma_eps = standard deviation of white noise
    #         one_sided = True or False (If 'True' then spectra at only non-negative values will be returned)
    # outputs: pxx_den_theory = power spectral density of FARIMA model

    pxx_epsilon = sigma_eps**2          # Note: the factor of 1/2/np.pi has not been included beucase signal.periodogram function does not include it
    omegas = f*2*np.pi
    iota = (-1)**0.5
    exp_neg_iota_omega = np.exp(-iota*omegas)

    # auto-regressive polynomial
    phi_p = autoregressPol(p,ar,exp_neg_iota_omega)
 
    # moving average polynomials
    shi_q = movingAvgPol(q,ma,exp_neg_iota_omega)

    if (one_sided == True):
        pxx_den_theory = 2*(abs(1 - exp_neg_iota_omega))**(-2*d)*(abs(shi_q))**2*(abs(phi_p))**(-2)*pxx_epsilon
        # note: the factor of 2 has been included becuase of one-sided computation
    else:
        pxx_den_theory = (abs(1 - exp_neg_iota_omega))**(-2*d)*(abs(shi_q))**2*(abs(phi_p))**(-2)*pxx_epsilon

    return pxx_den_theory

# computations of approximate negative log-likelihood function
def WhittleLogLikeli(pxx_den,pxx_den_theory,contains_zero_frequency):
    # inputs:  pxx_den = periodogram of observed time-series (one sided inlcuding at frequency zero)
    #          pxx_den_theory =  theoretical power spectral density of the FARIMA model (one-sided including at frequency zero)
    # outputs: Lw = Whittle's approximate negative log likelihood (-loglik, http://people.stern.nyu.edu/churvich/TimeSeries/Handouts/Whittle.pdf, date accessed: 7 Oct 2021)

    if (contains_zero_frequency == True):
        log_pxx_den_theory = np.log(pxx_den_theory[1:])
        ratio_pxx_den = pxx_den[1:]/pxx_den_theory[1:]
        Lw = np.sum(log_pxx_den_theory) + np.sum(ratio_pxx_den)
    else :
        log_pxx_den_theory = np.log(pxx_den_theory)
        ratio_pxx_den = pxx_den/pxx_den_theory
        Lw = np.sum(log_pxx_den_theory) + np.sum(ratio_pxx_den)

    return Lw

# Objective function computation
def objectiveFnc(theta,p,q,f,pxx_den,one_sided,contains_zero_frequency):
    # inputs: theta = a numpy array of parameters (arparams, maparams, d, sigma_eps)
    #                 arparams = autoregressive parameters without zero lag term; phis such that X_t = \sum_{j=1}^{n}phi_j*X_(t-j)
    #                 maparams = moving-avergae parameters without zero lag term
    #                 sigma_eps = standard deviation of white noise
    #                 d = differencing order
    #         p =     order of autoregressive polynomial (it is not optimized)
    #         q =     order of moving-average polynomial (it is not optimized)
    #         f =     frequencies at which the theoretical power spectral density need to be computed (in cycles per unit time)
    #         pxx_den = one-sided periodogram of the observed time series
    # outputs: Lw = Whittle's approximate likelihood (negative log-lilekihood)

    arparams = theta[0:p]
    maparams = theta[p:p+q]
    d = theta[p+q]
    sigma_eps = theta[p+q+1]

    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    
    
    pxx_den_theory = pxx_denFARIMA(p,d,q,ar,ma,f,sigma_eps,one_sided)

    Lw = WhittleLogLikeli(pxx_den,pxx_den_theory,contains_zero_frequency)

    return Lw

# first partial derivatives of the Whittle's approximate likelihood
def jacLw(p,q,theta,f,one_sided,I):

    # inputs: p = order of autoregressive polynomial
    #         q = order of moving average polynomial
    #         theta = parameters of the model (arparams, maparams, sigma_eps, d)
    #                 (sigma_eps = standard deviation of white noise)
    #         f = frequencies at which theoretical power spectral density is to be computed
    #             (f does not include zero frequency)
    #         one_sided = True or False (if True, the psd will be computed for positive frequency only  )
    #         I = periodogram of the observations (does not include values at zero frequency)
    # outputs: f = (analytical partial derivatives w.r.t arparams, maparams, d, and sigma_eps, respectively)
    arparams = theta[0:p]
    maparams = theta[p:p+q]
    d = theta[p+q]
    sigma_eps = theta[p+q+1]

    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]

    # power spectral density at different frequencies
    pxx_den_theory = pxx_denFARIMA(p,d,q,ar,ma,f,sigma_eps,one_sided)

    # define iota and angular frequencies
    iota = (-1)**0.5
    omega = f*2*np.pi
    exp_neg_iota_omega = np.exp(-iota*omega)

    # derivative w.r.t. d
    term1 = 1 - I/pxx_den_theory
    term2 = np.log(abs(1-exp_neg_iota_omega))
    partial_Lw_d = -2*np.sum(term1*term2)

    # derivative w.r.t. shi_k for k = 1 to q 
    krange = range(1,q+1)
    ma_pol = movingAvgPol(q,ma,exp_neg_iota_omega)
    mterm2 = abs(ma_pol)**(-2)
    partial_Lw_shi = []
    for k in krange:
        mterm3 = (exp_neg_iota_omega**k)*np.conj(ma_pol) + ((np.conj(exp_neg_iota_omega))**k)*ma_pol
        partial_Lw_shi_k = np.sum(term1*mterm2*mterm3)
        partial_Lw_shi.append(partial_Lw_shi_k)
    partial_Lw_shi = np.array(partial_Lw_shi)

    # derivative w.r.t. phi_k for k = 1 to p
    krange = range(1,p+1)
    ar_pol = autoregressPol(p,ar,exp_neg_iota_omega)
    aterm2 = abs(ar_pol)**(-2)
    partial_Lw_phi = []
    for k in krange:
        aterm3 = (exp_neg_iota_omega**k)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**k)*ar_pol
        partial_Lw_phi_k = np.sum(term1*aterm2*aterm3)
        partial_Lw_phi.append(partial_Lw_phi_k)
    partial_Lw_phi = np.array(partial_Lw_phi)

    # derivatibe w.r.t. sigma_eps
    partial_Lw_sigma_eps = 2*np.sum(term1)/sigma_eps

    # arrange derivatives in a list
    f = partial_Lw_phi.tolist() + partial_Lw_shi.tolist() + [partial_Lw_d] + [partial_Lw_sigma_eps]
    return f

#   Hessian of the Whittle's approximate likelihood w.r.t. its parameters
def hessLw(p,q,theta,f,one_sided,I):

    # inputs: p = order of autoregressive polynomial
    #         q = order of moving average polynomial
    #         theta = parameters of the model (arparams, maparams, sigma_eps, d)
    #                 (sigma_eps = standard deviation of white noise)
    #         f = frequencies at which theoretical power spectral density is to be computed
    #             (f does not include zero frequency)
    #         one_sided = True or False (if True, the psd will be computed for positive frequency only  )
    #         I = periodogram of the observations (does not include values at zero frequency)
    # outputs: Hessian_mat = Hessian matrix
     
    arparams = theta[0:p]
    maparams = theta[p:p+q]
    d = theta[p+q]
    sigma_eps = theta[p+q+1]

    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    
    # power spectral density at different frequencies
    pxx_den_theory = pxx_denFARIMA(p,d,q,ar,ma,f,sigma_eps,one_sided)

    # define iota and angular frequencies
    iota = (-1)**0.5
    omega = f*2*np.pi
    exp_neg_iota_omega = np.exp(-iota*omega)

    term1 = I/pxx_den_theory
    term2 = np.log(abs(1 - exp_neg_iota_omega))
    partial2_Lw_d2 = 4*(np.sum(term1*(term2**2)))

    # mixed second partial of d and shi_k(s)
    krange = range(1,q+1)
    ma_pol = movingAvgPol(q,ma,exp_neg_iota_omega)
    mdterm3 = abs(ma_pol)**(-2)
    partial2_Lw_shi_d = []
    for k in krange:
        mdterm4 = (exp_neg_iota_omega**k)*np.conj(ma_pol) + np.conj(exp_neg_iota_omega**k)*ma_pol
        partial2_Lw_shik_d = -2*np.sum(term1*term2*mdterm3*mdterm4)
        partial2_Lw_shi_d.append(partial2_Lw_shik_d.real)

    # mixed second partial of d and phi_k(s)
    krange = range(1,p+1)
    ar_pol = autoregressPol(p,ar,exp_neg_iota_omega)
    adterm3 = abs(ar_pol)**(-2)
    partial2_Lw_phi_d = []
    for k in krange:
        adterm4 = (exp_neg_iota_omega**k)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**k)*ar_pol
        partial2_Lw_phik_d = -2*np.sum(term1*term2*adterm3*adterm4)
        partial2_Lw_phi_d.append(partial2_Lw_phik_d.real)

    # mixed second partial of d and sigma_eps
    partial2_Lw_sigmaeps_d = -4*np.sum(term1*term2/sigma_eps)

    # second partial derivatives of shi_k(s) and shi_l(s)
    krange = range(1,q+1)
    lrange = krange

    partial2_Lw_shi_shi = np.zeros((q,q))
    for k in krange:
        for l in lrange:
            mmterm1 = -1 + 2*term1
            mmterm2 = (abs(ma_pol))**(-4)
            mmterm3 = (exp_neg_iota_omega**k)*np.conj(ma_pol) + np.conj(exp_neg_iota_omega**k)*ma_pol
            mmterm4 = (exp_neg_iota_omega**l)*np.conj(ma_pol) + np.conj(exp_neg_iota_omega**l)*ma_pol
            mmterm5 = 1 - term1
            mmterm6 = (abs(ma_pol))**(-2)
            mmterm7 = exp_neg_iota_omega**(l-k) + np.conj(exp_neg_iota_omega**(l-k))
            partial2_shik_shil = np.sum(mmterm1*mmterm2*mmterm3*mmterm4) + np.sum(mmterm5*mmterm6*mmterm7)
            partial2_Lw_shi_shi[k-1,l-1] = partial2_shik_shil.real

    # mixed second partial derivatives of phi_k(s) and shi_l(s) 
    krange = range(1,q+1)   # MA coefficients 
    lrange = range(1,p+1)   # AR coefficients
    partial2_Lw_shi_phi = np.zeros((q,p))
    for k in krange:
        for l in lrange:
            amterm2 = (abs(ma_pol))**(-2)
            amterm3 = (abs(ar_pol))**(-2)
            amterm4 = (exp_neg_iota_omega**k)*np.conj(ma_pol) + np.conj(exp_neg_iota_omega**k)*ma_pol  # MA term
            amterm5 = (exp_neg_iota_omega**l)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**l)*ar_pol  # AR term
            partial2_Lw_phil_shik = np.sum(term1*amterm2*amterm3*amterm4*amterm5)
            partial2_Lw_shi_phi[k-1,l-1] = partial2_Lw_phil_shik.real


    # mixed second partial derivatives of shi_k(s) and sigma_epsilon
    k = 1
    smterm2 = 1/sigma_eps
    smterm3 = abs(ma_pol)**(-2)
    partial2_Lw_sigmaeps_shi = []
    for k in krange:
        smterm4 = (exp_neg_iota_omega**k)*np.conj(ma_pol) + np.conj(exp_neg_iota_omega**k)*ma_pol
        partial2_sigmaeps_shik = 2*np.sum(term1*smterm2*smterm3*smterm4)
        partial2_Lw_sigmaeps_shi.append(partial2_sigmaeps_shik.real)

    # mixed second partial derivatives of phi_k(s) and phi_l(s)
    krange = range(1,p+1)
    lrange = krange
    aaterm1 = abs(ar_pol)**(-4)
    aaterm4 = 1 - term1
    aaterm5 = abs(ar_pol)**(-2)
    partial2_Lw_phi_phi = np.zeros((p,p))
    for k in krange:
        for l in lrange:
            
            aaterm2 = (exp_neg_iota_omega**k)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**k)*ar_pol
            aaterm3 = (exp_neg_iota_omega**l)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**l)*ar_pol
            aaterm6 = exp_neg_iota_omega**(l-k) + np.conj(exp_neg_iota_omega**(l-k))
            partial2_Lw_phik_phil = np.sum(aaterm1*aaterm2*aaterm3) - np.sum(aaterm4*aaterm5*aaterm6)
            partial2_Lw_phi_phi[k-1,l-1] = partial2_Lw_phik_phil.real


    # mixed second partial derivatives of LW w.r.t. phi_k(s) and sigma_eps
    krange = range(1,p+1)
    saterm2 = 1/sigma_eps
    saterm3 = abs(ar_pol)**(-2)
    partial2_Lw_sigmaeps_phi = np.zeros((p,1))
    for k in krange:
        saterm4 = (exp_neg_iota_omega**k)*np.conj(ar_pol) + np.conj(exp_neg_iota_omega**k)*ar_pol
        partial2_Lw_sigmaeps_phi[k-1] = (2*np.sum(term1*saterm2*saterm3*saterm4)).real

    # second partial derivative of Lw w.r.t. sigma_eps
    partial2_Lw_sigmaeps2 = -2*len(f)/sigma_eps**2 + (6/sigma_eps**2)*np.sum(term1)

    # arrange all the elemenst into a  matrix - Hessian matrix
    partial2_Lw_shi_phi = np.array(partial2_Lw_shi_phi)
    partial2_Lw_phi_phi = np.array(partial2_Lw_phi_phi)
    partial2_Lw_shi_shi = np.array(partial2_Lw_shi_shi)
    partial2_Lw_phi_d = np.array(partial2_Lw_phi_d).reshape(1,-1)
    partial2_Lw_shi_d = np.array(partial2_Lw_shi_d).reshape(1,-1)
    partial2_Lw_sigmaeps_phi = np.array(partial2_Lw_sigmaeps_phi).reshape(1,-1)
    partial2_Lw_sigmaeps_shi = np.array(partial2_Lw_sigmaeps_shi).reshape(1,-1)
    partial2_Lw_d2 = np.array(partial2_Lw_d2).reshape(1,-1) 
    partial2_Lw_sigmaeps_d = np.array(partial2_Lw_sigmaeps_d).reshape(1,-1)
    partial2_Lw_sigmaeps2 = np.array(partial2_Lw_sigmaeps2).reshape(1,-1)

    Hessian_mat1 = np.concatenate((partial2_Lw_phi_phi,partial2_Lw_shi_phi,partial2_Lw_phi_d,partial2_Lw_sigmaeps_phi))
    Hessian_mat2 = np.concatenate((partial2_Lw_shi_phi.T,partial2_Lw_shi_shi,partial2_Lw_shi_d,partial2_Lw_sigmaeps_shi))
    Hessian_mat3 = np.concatenate((partial2_Lw_phi_d.T,partial2_Lw_shi_d.T,partial2_Lw_d2,partial2_Lw_sigmaeps_d))
    Hessian_mat4 = np.concatenate((partial2_Lw_sigmaeps_phi.T,partial2_Lw_sigmaeps_shi.T,partial2_Lw_sigmaeps_d,partial2_Lw_sigmaeps2))
    Hessian_mat = np.concatenate((Hessian_mat1,Hessian_mat2,Hessian_mat3,Hessian_mat4),axis = 1)

    return np.real(Hessian_mat)

# iterative d method for parameter estimation of FARIMA model
def ParEstIterd(data,p,q,one_sided,pt,f,I):
    
    dvals = np.arange(0,0.51,0.01)
    AIC_vals = []
    for d in dvals:
        if (d != 0):
            y = diffOp(data,d,pt)
        else:
            y = data
        model = ARIMA(y,order = (p,0,q),enforce_stationarity=True)
        model.initialize_approximate_diffuse()
        model_fit = model.fit(method_kwargs={'maxiter': 5000})
        #print('**************************************************************')
        #print(model_fit.mle_retvals)
        #print('**************************************************************')
        AIC_vals.append([d,model_fit.aic])

    # find the optimal d (= dopt)value corresponding to minimum AIC and estimate parameters corresponding to the optimal d value  
    AIC_vals = np.array(AIC_vals)
    minind = np.nonzero(AIC_vals[:,1] == AIC_vals[:,1].min())
    dopt = AIC_vals[minind[0],0]
    if (dopt != 0):
        y = diffOp(data,dopt,pt)
    else:
        y = data
    model = ARIMA(y,order = (p,0,q),enforce_stationarity=True)
    model.initialize_approximate_diffuse()
    model_fit = model.fit(method_kwargs={'maxiter': 5000})
    #print('**************************************************************')
    #print(model_fit.mle_retvals)
    #print('**************************************************************')

    # extract parameters
    if (p != 0):
        arparams_est = model_fit.arparams
    else:
        arparams_est = np.array([0]*p)
    
    if (q != 0):
        maparams_est = model_fit.maparams
    else:
        maparams_est = np.array([0]*q)

    sigma_est = (model_fit.params[-1])**0.5
    theta = np.r_[arparams_est,maparams_est,dopt[0],sigma_est]
    constant = model_fit.params[0]

    residuals = model_fit.resid

    # estimate 95% confidence interval
    hess = hessLw(p,q,theta,f,one_sided,I)
    Sigma2 = np.linalg.inv(hess)
    stds = np.diag(Sigma2)**0.5
    conf_hessian = np.concatenate(((theta - 1.96*stds).reshape(1,-1).T, (theta + 1.96*stds).reshape(1,-1).T),axis = 1)
    conf_module = np.array(model_fit.conf_int(0.05))

    # Bound the confidence intervals of 'd' between 0.0 and 0.5 
    conf_hessian[-2,0] = np.max([conf_hessian[-2,0],0])
    conf_hessian[-2,1] = np.min([conf_hessian[-2,1],0.5])
    
    return theta, conf_hessian, residuals, conf_module, constant

# Ljung-Box test for autocorrelation of white noise (residuals)
def LjungBoxFarima(residuals):
    # inputs: residuals = residuals for each window (an array_like: each column contains residuals for one window) 
    # outputs: Ljugn-Box test results
    #          (lbvalue: Ljugn-Box test statistic)
    #          (pvalue: p value)
    
    lbvalues = []
    pvalues = []
    for ind in range(0,residuals.shape[1]):
        lbvalue, pvalue = acorr_ljungbox(residuals[:,ind], lags = 20)
        lbvalues.append(lbvalue)
        pvalues.append(pvalue)

    return lbvalues, pvalues

# computation of autocorrelation of residuals
def autoCorrFarima(residuals):
    # inputs: residuals = residuals for each window (an array_like: each column contains residuals for one window)

    acfs = []
    confints = []
    qstats = []
    pvalues = []
    for ind in range(0,residuals.shape[1]):
        acf_vals, confint, qstat, pvalue = acf(residuals[:,ind], nlags = 100, qstat = True, fft = True, alpha = 0.05)
        acfs.append(acf_vals)
        confints.append(confint)
        qstats.append(qstat)
        pvalues.append(pvalue)

    return acfs, confints, qstats, pvalues