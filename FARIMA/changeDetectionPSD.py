"""

This script analyses the change in PSD

Author: Abhinav Gupta (Created: 09 Nov 2021)

"""

import numpy as np
import os
import statsmodels.api as sm
import FarimaModule
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# read gauage information
direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata'
fname = 'gauge_information.txt'
filename = direc + '/' + fname

fid = open(filename,'r')
gauge_info = fid.readlines()
fid.close()

gauge_info_data=  []
for gind in range(1,len(gauge_info)):
    data_tmp = gauge_info[gind].split('\t')
    gauge_info_data.append([data_tmp[1],float(data_tmp[3]),float(data_tmp[4])])
gauge_info_data = np.array(gauge_info_data)

wlen = 3650
mstep = 365*3
N = 3650            # number of time steps at which streamflow is available in each window
fs = 1              # sampling frequency (number of samples per day)
fmax = fs/2         # maximum frequncy according to aliasing effect
#fregions = [[0, 2*10**(-3)], [2*10**(-3), 10**(-2)], [10**(-2), 2*10**(-2)],[2*10**(-2), 10**(-1)],[10**(-1), 2.5*10**(-1)],[2.5*10**(-1), 5*10**(-1)]]
fregions = [[0, 1/365], [1/365, 1/120],[1/120, 1/30],[1/30, 1/14],[1/14, 0.5]]

# frequency values at which PSD is to be computed 
fint = 2*fmax/N
f = np.arange(0, fmax, fint)
f = f[1:]
one_sided = True

direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results'
listFile = os.listdir(direc)

local_dir_ind = -1
write_data = []
for local_dir in listFile:
    if os.path.isdir(direc + '/' + local_dir):
        
        local_dir_ind = local_dir_ind + 1
        print(local_dir_ind)
        fname = local_dir + '_coefficient_estimates_mean.txt'
        filename = direc + '/' + local_dir + '/' + fname

        fid = open(filename,'r')
        data = fid.readlines()
        fid.close()

        params_name = data[0].split()
        # identify AR and MA parameters
        boolean = []
        for pind in range(0,len(params_name)-2):
            boolean.append(any(string in params_name[pind] for string in ['AR']))
        boolean = np.array(boolean)
        ar_locs = np.nonzero(boolean == True)
        ma_locs = np.nonzero(boolean == False)
        p = ar_locs[0].shape[0]
        q = ma_locs[0].shape[0]

        # extract coefficient values
        coeffs = []
        for ind in range(1,len(data)):
            data_tmp = data[ind].split()
            coeffs.append(data_tmp)
        coeffs = np.array(coeffs)
        coeffs = coeffs.astype(float)

        numWindows = coeffs.shape[0]

        #### extract PSD values
        pxxden_theory = []
        for winind in range(0,numWindows):
            arparams = coeffs[winind,0:p]
            maparams = coeffs[winind,p:p+q]
            d = coeffs[winind,p+q]
            sigma_eps = coeffs[winind,p+q+1]

            ar = np.r_[1, -arparams]
            ma = np.r_[1, maparams]
            pxx_tmp = FarimaModule.pxx_denFARIMA(p,d,q,ar,ma,f,sigma_eps,one_sided)
            sigmax2 = simpson(pxx_tmp,2*np.pi*f, even = 'avg')
            pxx_tmp = pxx_tmp/sigmax2
            pxxden_theory.append(pxx_tmp)

        # plot normalized psds for different windows
        """
        plt.figure()
        plt.rcParams.update({'font.family':'Arial'})
        for winind in range(0,numWindows):
            plt.loglog(f,pxxden_theory[winind])
        plt.legend(list(range(1,numWindows+1)), frameon = False, loc = 'lower left')
        plt.xlabel('Frequency (Cycles per day)')
        plt.ylabel('Normalized power spectral density')

        # save figure
        sname = 'normalized_PSD.png'
        filename = direc + '/' + local_dir + '/' + sname
        plt.savefig(filename, dpi = 300)

        sname = 'normalized_PSD.svg'
        filename = direc + '/' + local_dir + '/' + sname
        plt.savefig(filename, dpi = 300)

        plt.close('all')
        """
        # divide the entire psd into different frequnency region and compute the area under the curve in the three regions
        areas = []
        for rind in range(0,len(fregions)):
            frmin = fregions[rind][0]
            frmax = fregions[rind][1]
            ind = [i for i in range(0,len(f)) if (f[i]>frmin and f[i]<=frmax)]
            ftmp = f[ind]
            areas_tmp = []
            for winind in range(0,numWindows):
                pxxtmp = pxxden_theory[winind][ind]
                areas_tmp.append(simpson(pxxtmp,2*np.pi*ftmp, even = 'avg'))
            areas.append(areas_tmp)

        # plot 'areas' data
        """
        x = list(range(1,numWindows+1))
        numPlots = len(fregions)
        rows = int((numPlots)**0.5)
        cols = int(np.ceil(numPlots/rows))
        fig, ax  = plt.subplots(rows,cols)
        plt.rcParams.update({'font.family':'Arial'})
        frequency_regions = ['> 1Year','6 months to a year', '4 to 6 months', '1 to 4 months', '2 week to 1 month', '1 week to 2 weeks', '< 1 week'  ]

        for r in range(0,ax.shape[0]):
            for c in range(0,ax.shape[1]):
                ax[r,c].set_axis_off()

        find = -1
        for rind in range(0,rows):
            for cind in range(0,cols):
                find = find + 1
                if find < len(fregions):
                    ax[rind,cind].set_axis_on()
                    ax[rind,cind].plot(x,areas[find],'-o')
                    ax[rind,cind].set_title('Frequency range :\n ' + str(frequency_regions[find]), fontsize = 10)
        fig.text(0.5,0.01,'Moving Window number', ha = 'center', weight = 'bold')
        fig.text(0.01,0.5,'Moving Window number', va = 'center', rotation = 'vertical', weight = 'bold')

        fig.tight_layout()

        # save figure
        sname = 'psd_areas.png'
        filename = direc + '/' + local_dir + '/' + sname
        plt.savefig(filename, dpi = 300)

        sname = 'psd_areas.svg'
        filename = direc + '/' + local_dir + '/' + sname
        plt.savefig(filename, dpi = 300)

        plt.close('all')
        """
        # compute slope of the areas over different time-windows
        slopes = []
        x = list(range(1,numWindows+1))
        x = np.array(x)
        x = np.concatenate((np.ones((numWindows,1)),x.reshape(1,-1).T), axis = 1)
        for ind in range(0,len(fregions)):
            y = np.array(areas[ind])
            model = sm.OLS(y, x)
            results = model.fit()
            slopes.append(results.params[1])
        
        gind = np.nonzero(gauge_info_data[:,0] == local_dir)
        write_data.append([local_dir] + slopes + [float(gauge_info_data[gind,1]),float(gauge_info_data[gind,2])])

# write areas data to a textfile along with
sname = 'changePSD_1.txt'
filename = direc + '/' + sname
fid = open(filename, 'w')
fid.write('Gauge_id\tgreater_than_oneyear\t4months_oneyear\t1month_to_4months\t2week_to_1month\tless_than_2week\tLat\tLong\n')
for wind in range(0,len(write_data)):
    fid.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n'%tuple(write_data[wind]))
fid.close()


