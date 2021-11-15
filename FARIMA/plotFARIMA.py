"""
This script reads parameter estimates of FARIMA model and plots them

Author: Abhinav Gupta (Date: 25 Oct 2021)
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
import FarimaModule
import output_Files

# read estimates of parameter mean values
def plotParEst(save_dir,station_id):

    fname  = station_id + '_coefficient_estimates_mean.txt'
    filename = save_dir + '/' + fname
    fid = open(filename,'r')
    data_mean = fid.readlines()
    fid.close()
    header_mean = data_mean[0].split()
    header_mean[-1] = r'$\sigma$'
    header_mean[-2] = r'$H$'
    del data_mean[0]
    for rind in range(0,len(data_mean)):
        data_tmp = data_mean[rind].split()
        data_mean[rind] = np.array(data_tmp, dtype = float)
    data_mean = np.array(data_mean)

    # read estimates of parameter lower bound of 95% confidence interval
    fname  = station_id + '_coefficient_estimates_025.txt'
    filename = save_dir + '/' + fname
    fid = open(filename,'r')
    data_025 = fid.readlines()
    fid.close()
    header_025 = data_025[0].split()
    del data_025[0]
    for rind in range(0,len(data_025)):
        data_tmp = data_025[rind].split()
        data_025[rind] = np.array(data_tmp, dtype = float)
    data_025 = np.array(data_025)

    # read estimates of parameter upper bound of 95% confidence interval
    fname  = station_id + '_coefficient_estimates_975.txt'
    filename = save_dir + '/' +fname
    fid = open(filename,'r')
    data_975 = fid.readlines()
    fid.close()
    header_975 = data_975[0].split()
    del data_975[0]
    for rind in range(0,len(data_975)):
        data_tmp = data_975[rind].split()
        data_975[rind] = np.array(data_tmp, dtype = float)
    data_975 = np.array(data_975)

    # add 0.5 to d values to convert them to H values
    data_mean[:,-2] = data_mean[:,-2] + 0.5
    data_025[:,-2] = data_025[:,-2] + 0.5
    data_975[:,-2] = data_975[:,-2] + 0.5

    # plot estimated coefficients
    numPlots = len(header_mean)
    rows = int((numPlots)**0.5)
    cols = int(np.ceil(numPlots/rows))

    # plot coefficients
    count = 0
    rcount = 0
    ccount= 0
    numWindows = data_mean.shape[0]
    windows = np.arange(1,numWindows+1)
    cc = list(range(0,cols))*rows
    rc = list(range(0,rows))*cols
    rc.sort()
    
    plt.rcParams.update({'font.family':'Arial'})
    fig, ax = plt.subplots(rows,cols)

    for r in range(0,ax.shape[0]):
        for c in range(0,ax.shape[1]):
            ax[r,c].set_axis_off()

    for ind in range(0,data_mean.shape[1]):

        count = count + 1
        rcount = rc[ind]
        ccount = cc[ind]
        param_tmp = data_mean[:,ind]
        param_025_tmp = data_025[:,ind]
        param_975_tmp = data_975[:,ind]

        ax[rcount,ccount].set_axis_on()
        ax[rcount,ccount].plot(windows, param_tmp, color = 'tab:blue',linewidth = 1)
        ax[rcount,ccount].plot(windows, param_025_tmp, color = 'tab:blue', linestyle = '--',linewidth = 1)
        ax[rcount,ccount].plot(windows, param_975_tmp,  color = 'tab:blue',linestyle = '--',linewidth = 1)
        ax[rcount,ccount].set_title(header_mean[ind], y = 0.78, x = 0.18, fontsize=8, fontname = 'Arial')
        ax[rcount,ccount].set_xticks([1,3,5,7,9])
        tmp1 = np.concatenate((param_025_tmp.reshape(1,-1).T,param_tmp.reshape(1,-1).T),axis=0)
        tmp2 = np.concatenate((param_975_tmp.reshape(1,-1).T,param_tmp.reshape(1,-1).T),axis=0)
        ymin = round(np.nanmin(tmp1),2)
        ymax = round(np.nanmax(tmp2),2)
        yint = round((ymax - ymin)/4,3)
        if (ymin-ymax !=0):
            ytick_labels = np.arange(ymin,ymax + yint,yint)
        else:
            ytick_labels = np.arange(ymin - 0.1,ymin + 0.1 + 0.05,0.05)
        ax[rcount,ccount].set_yticks(ytick_labels)
        ax[rcount,ccount].tick_params(axis = 'both', which = 'major', labelsize = 8)
    
    ctitle = fname.split('_')[0]
    fig.suptitle(ctitle,fontname = 'Arial', fontsize = 10, fontweight = 'bold')
    fig.text(0.5, 0.01, 'Moving window number', ha='center', fontname = 'Arial', fontsize = 10, fontweight = 'bold') # common x-axis label
    fig.text(0.001, 0.5, 'Parameter estimates', va='center', rotation = 'vertical', fontname = 'Arial', fontsize = 10, fontweight = 'bold') # common y-axis label
    fig.legend(['Mean','95% confidence interval'], prop={'family':'Arial', 'size':8}, loc = 'upper left', ncol = 2, frameon = False)

    fig.tight_layout()  

    # save plot
    sname = ctitle + '.svg'
    filename = save_dir + '/' + sname
    fig.savefig(filename,dpi = 300)

    sname = ctitle + '.png'
    filename = save_dir + '/' + sname
    fig.savefig(filename, dpi = 300)

    #plt.show()
    plt.close()

    return None

def plotStrmData(strm,seasonal,seasonal_daily_avg,deseason,save_dir):
    # inputs: strm = streamflow data
    #         seasonal = seasonal component of strm
    #         seasonal_daily_avg = seasonal component computed by daily averaging method
    #         deseasonal  = deseasonalized streamflow data
    # output: plots of the three quantities as subplots

    plt.rcParams['mathtext.default'] = 'regular'
    fig, ax  = plt.subplots(3,1)
    N = strm.shape[0]

    # plot strm
    ax[0].plot(strm, color = 'tab:Blue', linewidth = 1)
    ax[0].set_xlabel('Day since 01 Oct 1980 (including the start day)', fontname = 'Arial', fontsize = 9)
    ax[0].set_ylabel('Streamflow' +  r'($m^3 s^{-1}$)',fontname = 'Arial', fontsize = 9)
    ax[0].set_xticks(np.arange(1,N+1,1000))
    ax[0].tick_params(axis = 'both', which = 'major', labelsize = 8)

    # plot seasonal component
    ax[1].plot(seasonal, color = 'tab:Blue', linewidth = 1)
    ax[1].plot(seasonal_daily_avg, color = 'tab:orange', linewidth = 1, linestyle = '--')
    ax[1].set_xlabel('Day since 01 Oct 1980 (including the start day)', fontname = 'Arial', fontsize = 9)
    ax[1].set_ylabel('Seasonal component of\nstreamflow' + r'($m^3 s^{-1}$)', fontname = 'Arial', fontsize = 9)
    ax[1].set_xticks(np.arange(1,366,30))
    ax[1].legend(['LOWESS','Simple averaging'], frameon = False, prop={'family':'Arial', 'size':9})
    ax[1].tick_params(axis = 'both', which = 'major', labelsize = 8)

    # plot deasonalized streamflow data
    ax[2].plot(deseason, color = 'tab:Blue', linewidth = 1)
    ax[2].set_xlabel('Day since 01 Oct 1980 (including the start day)', fontname = 'Arial', fontsize = 9)
    ax[2].set_ylabel('Deseasonalized\nstreamflow' + r'($m^3 s^{-1}$)', fontname = 'Arial', fontsize = 9)
    ax[2].set_xticks(np.arange(1,N+1,1000))
    ax[2].tick_params(axis = 'both', which = 'major', labelsize = 8)

    fig.tight_layout() 

    # save plot
    sname = 'strm_data.svg'
    filename = save_dir + '/' + sname
    plt.savefig(filename, dpi = 300)

    sname = 'strm_data.png'
    filename = save_dir + '/' + sname
    plt.savefig(filename, dpi = 300)

    plt.close()

    return None

# plot residual data
def plotResidual(resid,save_dir,sname):
    # inputs: resid = residual time series that need to be plotted
    #         save_dir = directory where plots will be saved
    #         sname = name of the file to be saved 
    
    plt.rcParams['mathtext.default'] = 'regular'

    ax1 = plt.subplot(2,1,1)
    ax1.plot(resid,linewidth = 1)
    ax1.set_ylabel('Residuals' + r'($m^3 s^{-1}$)', fontsize = 8, fontname = 'Arial')
    ax1.set_xlabel('Time-step (days)', fontsize = 8, fontname = 'Arial')
    ax1.tick_params(labelsize = 8)

    ax2 = plt.subplot(2,2,3)
    qqplot(resid, ax = ax2, line = 'q')
    ax2.lines[0]._markersize = 2
    ax2.set_ylabel('Sample quantiles', fontsize = 8, fontname = 'Arial')
    ax2.set_xlabel('Theoretical quantiles', fontsize = 8, fontname = 'Arial')
    ax2.legend(['Residuals', 'Line fitted through quartiles'], frameon = False, prop={'family':'Arial', 'size':8})
    ax2.tick_params(labelsize = 8)
    

    ax3 = plt.subplot(2,2,4)
    plot_acf(resid, ax = ax3)
    ax3.set_title("")
    ax3.set_ylabel('Autocorrelation', fontsize = 8, fontname = 'Arial')
    ax3.set_xlabel('Lags (Days)', fontsize = 8, fontname = 'Arial')
    ax3.lines[1]._markersize = 2
    ax3.lines[1]._linewidth = 1
    ax3.tick_params(labelsize = 8)
    
    plt.tight_layout()

    #plt.show()
    # save plot
    filename = save_dir + '/' + sname + '.svg'
    plt.savefig(filename, dpi = 300)

    filename = save_dir + '/' + sname + '.png'
    plt.savefig(filename, dpi = 300)

    plt.close()

    return None

# plot scale vs aggregated variances (used in the computations of Hurst exponent)
def plotVarScale(log_m,log_variances,H,slope,intercept,save_dir,sname):
    # inputs: log_m         = Logarithm of scales (base 10)
    #         log_variances = Logarithm of aggregated variances (base 10)
    #         H             = Hurst Exponent
    #         save_dir      = direcotry where the plot will be saved
    # outputs: a plot

    plt.rcParams['mathtext.default'] = 'regular'
    plt.rc('xtick', labelsize = 10)
    plt.rc('ytick', labelsize = 10)
    plt.rc('font',**{'family':'Arial'})

    # construct regressionn line data
    log_m_lines = np.concatenate(([0],log_m))
    yreg = intercept + slope*log_m_lines
    y_wihoutpersistence = intercept + (-1)*log_m_lines

    plt.scatter(log_m,log_variances)
    plt.plot(log_m_lines,yreg, color = 'Black',linewidth = 0.8)
    plt.plot(log_m_lines,y_wihoutpersistence, color = 'Red',linewidth = 0.8)
    plt.xlabel(r'$log_{10}(\itm)$', fontsize = 10, fontname = 'Arial')
    plt.ylabel(r'$log_{10}(\it\sigma_{m}^2)$',fontsize = 10, fontname = 'Arial')
    plt.legend(['Regression line','H = 0.5 line','Observations'], frameon = False, loc = 'upper right')
    plt.title('H = ' + str(round(H,2)))

    # save the plot
    sname_tmp = sname + '.svg'
    filename = save_dir + '/' + sname_tmp
    plt.savefig(filename, dpi = 300)

    sname_tmp = sname + '.png'
    filename = save_dir + '/' + sname_tmp
    plt.savefig(filename, dpi = 300)

    plt.close()

    return None

# plot R/S statistic against against m
def plotR_by_S(log_m,log_Rrescaled,slope, intercept,save_dir,sname):
    
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rc('xtick', labelsize = 10)
    plt.rc('ytick', labelsize = 10)
    plt.rc('font',**{'family':'Arial'})

    # construct regression line data
    y = intercept + slope*log_m

    plt.scatter(log_m,log_Rrescaled, s = 1)
    plt.plot(log_m,y, linewidth = 1.5, color = 'Black')
    plt.ylabel(r'$log_{10}(\itR/\itS)$')
    plt.xlabel(r'$log_{10}(\itm)$')
    plt.legend(['Regeression Line', 'Observations'], frameon = False, loc = 'upper left')
    plt.title('H = ' + str(round(slope,2)))

    # save plots
    sname_tmp = sname + '.svg'
    filename = save_dir + '/' + sname_tmp
    plt.savefig(filename)

    sname_tmp = sname + '.png'
    filename = save_dir + '/' + sname_tmp
    plt.savefig(filename)

    plt.close()

    return None

# plot periodogram data
def plotPeriodogram(f,I,save_dir,sname):
    # inputs: f = frequencies at which periodogram is to be plotted
    #         I = periodogram values at frequencies in f
    # outputs: plot 

    plt.rcParams['mathtext.default'] = 'regular'
    plt.rc('xtick', labelsize = 10)
    plt.rc('ytick', labelsize = 10)
    plt.rc('font',**{'family':'Arial'})

    plt.loglog(f,I)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density')
    plt.title('Periodogram of\ndeseasonalized streamflows')

    # save plot
    sname_tmp = sname + '.svg'
    filename = save_dir + '/' + sname_tmp
    plt.savefig(filename, dpi = 300)

    sname_tmp = sname + '.png' 
    filename = save_dir + '/' + sname_tmp
    plt.savefig(filename, dpi = 300)
    
    plt.close()

    return None
#plotParEst('D:/Research/non_staitionarity/codes/results/FARIMA_results/01055000','01055000')

""" filename = 'D:/Research/non_staitionarity/codes/results/FARIMA_results/01022500/residuals.txt'
fid = open(filename,'r')
data = fid.readlines()
fid.close()
del data[0]

residuals = []
for ind in range(0,len(data)):
    resid_tmp = data[ind].split()
    residuals.append(resid_tmp)
residuals = np.array(residuals, dtype = float)

save_dir = 'D:/Research/non_staitionarity/codes/results/FARIMA_results/01022500'
plotResidual(residuals[:,1],save_dir,'residuals_plot_1')

acfs, confints, qstats, pvalues = FarimaModule.autoCorrFarima(residuals)
output_Files.autocorrText(acfs, qstats, pvalues, save_dir) """

#plotParEst('D:/Research/non_staitionarity/codes/results/FARIMA_results/01030500','01030500')