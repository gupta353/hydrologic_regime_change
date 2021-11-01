"""
This script writes the outputs obtained by 'FARIMA_implement.py' to a textfile

Author: Abhinav Gupta (23 Oct 2021)

"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Reararnges the parameters in more accesible format
def rearrangeParams(order_windows, params_windows,p_max,q_max):
    # inputs:  order_windows = AR and MA polynomials order for each moving window (a list of lists)
    #          params_windows = a list of parameters for each window
               # p_max = maximum value of AR order
               # q_max = maximum value of MA order
    # outputs: Parameters re-arranged in a more accessible manner (see the return statement)
     
    params_ar = []
    params_ma = []
    params_d_sig = []
    params_025_ar = []
    params_025_ma = []
    params_025_d_sig = []
    params_975_ar = []
    params_975_ma = []
    params_975_d_sig = []
    for ind in range(0,len(params_windows)):

        p = order_windows[ind][0]
        q = order_windows[ind][1]

        if (p != 0):
            params_ar.append(list(params_windows[ind][0:p,0]))
            params_025_ar.append(list(params_windows[ind][0:p,1]))
            params_975_ar.append(list(params_windows[ind][0:p,2]))

        else :
            params_ar.append([0]*p_max)
            params_025_ar.append([0]*p_max)
            params_975_ar.append([0]*p_max)

        if (q != 0):
            params_ma.append(list(params_windows[ind][p:p+q,0]))
            params_025_ma.append(list(params_windows[ind][p:p+q,1]))
            params_975_ma.append(list(params_windows[ind][p:p+q,2]))

        else :
            params_ma.append([0]*q_max)
            params_025_ma.append([0]*q_max)
            params_975_ma.append([0]*q_max)
            
        params_d_sig.append([params_windows[ind][p+q,0],params_windows[ind][p+q+1,0]])
        params_025_d_sig.append([params_windows[ind][p+q,1],params_windows[ind][p+q+1,1]])
        params_975_d_sig.append([params_windows[ind][p+q,2],params_windows[ind][p+q+1,2]])

    # pad zeros at the end of each column so every column has the same length
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
    params_d_sig = np.array(params_d_sig)
    params_025_d_sig = np.array(params_025_d_sig)
    params_975_d_sig = np.array(params_975_d_sig)

    return params_ar, params_025_ar, params_975_ar, params_ma, params_025_ma, params_975_ma, params_d_sig, params_025_d_sig, params_975_d_sig, max_length_ar, max_length_ma

# 
def outText(params_ar, params_025_ar, params_975_ar, params_ma, params_025_ma, params_975_ma, params_d_sig, params_025_d_sig, params_975_d_sig, residuals,
station_id,save_dir,param_names_ar,param_names_ma,max_length_ar,max_length_ma):
    # mean values of the parameters
    write_mean_data = np.concatenate((params_ar,params_ma,params_d_sig),axis = 1)
    sfname = station_id + '_coefficient_estimates_mean.txt'
    filename = save_dir + '/' + sfname
    fid = open(filename,'w')
    header = '\t'.join(param_names_ar) + '\t' + '\t'.join(param_names_ma) + '\td\tsigma\n'
    fid.write(header)
    formatspec = '\t'.join(['%f']*max_length_ar) + '\t' + '\t'.join(['%f']*max_length_ma) + '\t%f\t' + '%f\n'

    for ind in range(0,write_mean_data.shape[0]):
        fid.write(formatspec %tuple(write_mean_data[ind,:]))
    fid.close()

    # 2.5th percentile of the parameters
    write_025_data = np.concatenate((params_025_ar,params_025_ma,params_025_d_sig),axis = 1)
    sfname = station_id + '_coefficient_estimates_025.txt'
    filename = save_dir + '/' + sfname
    fid = open(filename,'w')
    header = '\t'.join(param_names_ar) + '\t' + '\t'.join(param_names_ma) + '\td\tsigma\n'
    fid.write(header)
    for ind in range(0,write_025_data.shape[0]):
        fid.write(formatspec %tuple(write_025_data[ind,:]))
    fid.close()

    # 97..5th percentile of the parameters
    write_975_data = np.concatenate((params_975_ar,params_975_ma,params_975_d_sig),axis = 1)
    sfname = station_id + '_coefficient_estimates_975.txt'
    filename = save_dir + '/' + sfname
    fid = open(filename,'w')
    header = '\t'.join(param_names_ar) + '\t' + '\t'.join(param_names_ma) + '\td\tsigma\n'
    fid.write(header)
    for ind in range(0,write_975_data.shape[0]):
        fid.write(formatspec %tuple(write_975_data[ind,:]))
    fid.close()

    # write residual data
    # header string
    numWindows = residuals.shape[1]
    header_string = []
    for ind in range(1,numWindows+1):
        header_string.append('window_' + str(ind))
    header_string = '\t'.join(header_string)
    header_string = header_string + '\n'

    sfname = 'residuals.txt'
    filename = save_dir + '/' + sfname
    fid = open(filename,'w')
    fid.write(header_string)
    for wind in range(0,len(residuals)):
        tmp = tuple(residuals[wind,:])
        fid.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' %tmp)
    fid.close()

    return None

# write the confidence intervals obtained by default statsmodel method to a textfile
def confAutoText(conf_auto,order_windows,max_length_ar, max_length_ma,param_names_ar,param_names_ma,station_id,save_dir):
    # inputs: conf_auto = confidence intervals obtained for each window (a list of lists)
    #         order_windows = AR and MA polynomials order for each moving window (a list of lists)

    # extract AR, MA and sig parameters
    param_025_ar = []
    param_975_ar = []
    param_025_ma = []
    param_975_ma = []
    sig_025 = []
    sig_975 = []
    for oind in range(0,len(order_windows)):
        p = order_windows[oind][0]
        q = order_windows[oind][1]

        if (p != 0):
            param_025_ar.append(list(conf_auto[oind][1:p+1,0]))
            param_975_ar.append(list(conf_auto[oind][1:p+1,1]))
        else:
            param_025_ar.append([0]*max_length_ar)
            param_975_ar.append([0]*max_length_ar)

        if (q != 0):
            param_025_ma.append(list(conf_auto[oind][p+1:p+q+1,0]))
            param_975_ma.append(list(conf_auto[oind][p+1:p+q+1,1]))
        else:
            param_025_ma.append([0]*max_length_ma)
            param_975_ma.append([0]*max_length_ma)

        sig_025.append((conf_auto[oind][p+q+1,0])**0.5)
        sig_975.append((conf_auto[oind][p+q+1,1])**0.5)

    # pad with zeros to equalize the length of params in each window
    for aind in range(0,len(param_025_ar)):
        diff = max_length_ar - len(param_025_ar[aind])
        if (diff != 0):
            param_025_ar[aind] = param_025_ar[aind] + [0]*diff
            param_975_ar[aind] = param_975_ar[aind] + [0]*diff

    for mind in range(0,len(param_025_ma)):
        diff = max_length_ma - len(param_025_ma[mind])
        if (diff != 0):
            param_025_ma[mind] = param_025_ma[mind] + [0]*diff
            param_975_ma[mind] = param_975_ma[mind] + [0]*diff
    param_025_ar = np.array(param_025_ar)
    param_975_ar = np.array(param_975_ar)
    param_025_ma = np.array(param_025_ma)
    param_975_ma = np.array(param_975_ma)
    sig_025 = np.array(sig_025).reshape(1,-1).T
    sig_975 = np.array(sig_975).reshape(1,-1).T

    # write to a textfile
    # 2.5th percentile of the parameters
    write_025_data = np.concatenate((param_025_ar,param_025_ma,sig_025),axis = 1)
    sfname = station_id + '_coefficient_estimates_025_default.txt'
    filename = save_dir + '/' + sfname
    fid = open(filename,'w')
    header = '\t'.join(param_names_ar) + '\t' + '\t'.join(param_names_ma) + '\tsigma\n'
    fid.write(header)
    formatspec = '\t'.join(['%f']*max_length_ar) + '\t' + '\t'.join(['%f']*max_length_ma) + '\t%f\n'
    for ind in range(0,write_025_data.shape[0]):
        fid.write(formatspec %tuple(write_025_data[ind,:]))
    fid.close()

    # 97.5th percentile of the parameters
    write_975_data = np.concatenate((param_975_ar,param_975_ma,sig_975),axis = 1)
    sfname = station_id + '_coefficient_estimates_975_default.txt'
    filename = save_dir + '/' + sfname
    fid = open(filename,'w')
    header = '\t'.join(param_names_ar) + '\t' + '\t'.join(param_names_ma) + '\tsigma\n'
    fid.write(header)
    formatspec = '\t'.join(['%f']*max_length_ar) + '\t' + '\t'.join(['%f']*max_length_ma) + '\t%f\n'
    for ind in range(0,write_975_data.shape[0]):
        fid.write(formatspec %tuple(write_975_data[ind,:]))
    fid.close()

    return

def strmDataText(datenums,strm,seasonal,deseason,save_dir):

    # inputs: strm = streamflow data
    #         seasonal = seasonal component of strm
    #         deseasonal  = deseasonalized streamflow data

    # write strm and deseason data
    fname = 'strm.txt'
    filename = save_dir + '/' + fname
    fid = open(filename,'w')
    fid.write('Date\tStreamflow(CMS)\tDeseasonalized_streamflow(CMS)\n')
    for wind in range(0,strm.shape[0]):
        datenum_tmp = datenums[wind]
        date_tmp = datetime.fromordinal(int(datenum_tmp))
        date_tmp = str(date_tmp.year) + '/' + str(date_tmp.month) + '/' + str(date_tmp.day)
        fid.write('%s\t%f\t%f\n' %(date_tmp, strm[wind], deseason[wind]))
    fid.close()

    # write seasonal component data
    fname = 'seasonal_component.txt'
    filename = save_dir + '/' + fname
    fid = open(filename,'w')
    fid.write('Day\tSeasonal_component(CMS)\n')
    for wind in range(0,seasonal.shape[0]):
        fid.write('%d\t%f\n' %(wind+1,seasonal[wind]))
    fid.close()

    return None

# write autocorrelation data to textfiles
def autocorrText(acfs,qstats,pvalues,save_dir):

    acfs = np.array(acfs).T
    numWindows = acfs.shape[1]

    # header string
    header_string = []
    for ind in range(1,numWindows+1):
        header_string.append('window_' + str(ind))
    header_string = '\t'.join(header_string)
    header_string = header_string + '\n'

    # write acf data
    sname = 'autocorrelations_means.txt'
    filename = save_dir + '/' + sname
    fid = open(filename,'w')
    fid.write(header_string)
    for wind in range(0,acfs.shape[0]):
        acf_tmp = tuple(acfs[wind,:])
        fid.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n'%acf_tmp)
    fid.close()

    # write qstat data
    qstats = np.array(qstats).T
    sname = 'autocorrelations_LB_stats.txt'
    filename = save_dir + '/' + sname
    fid = open(filename,'w')
    fid.write(header_string)
    for wind in range(0,qstats.shape[0]):
        qstat_tmp = tuple(qstats[wind,:])
        fid.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n'%qstat_tmp)
    fid.close()

    # write pvalues of qstats
    pvalues = np.array(pvalues).T
    sname = 'autocorrelations_LB_stats_pvalues.txt'
    filename = save_dir + '/' + sname
    fid = open(filename,'w')
    fid.write(header_string)
    for wind in range(0,pvalues.shape[0]):
        pvalue_tmp = tuple(pvalues[wind,:])
        fid.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n'%pvalue_tmp)
    fid.close()