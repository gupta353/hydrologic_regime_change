"""

This script computes statistical distribution of area under different frequency regions of a psd, for
different time-windows

"""

import numpy as np
import os
import statsmodels.api as sm
import FarimaModule
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import multiprocessing as mp

np.random.seed(10)

# function to compute area under different frequency regions
def psdArea(pxxden_theory, f, fregions):
    areas_samp = []
    for rind in range(0,len(fregions)):             # for loop to iterate oveer different frequency regions
        frmin = fregions[rind][0]
        frmax = fregions[rind][1]
        ind = [i for i in range(0,len(f)) if (f[i]>frmin and f[i]<=frmax)]
        ftmp = f[ind]
        pxxtmp = pxxden_theory[ind]
        areas_samp.append(simpson(pxxtmp,2*np.pi*ftmp, even = 'avg'))
    
    return areas_samp

## function to compute p-value as defined below
# Let m1 = mean area under psd of the given frequency region in first time-window
# Let m2 = mean area under psd of the given frequency region in last time-window
# if m1 < m2 : p-value = (P(a1 > m2) + P(a2 < m1))/2 :: (a1 = random variable denoting area under psd the given frequency region)
# if m1 > m2 : p-value = probability that (P(a1 < m2) + P(a2>m1))/2
def p_value(areas):
    A = areas.copy()
    m1 = np.mean(A[0])

    pval = []
    for indtmp in range(1,len(A)):
        m2 = np.mean(A[indtmp])
        if m1 <= m2:
            p_val1 = np.nonzero(A[0]>m2)
            p_val1 = len(p_val1[0])/len(A[0])
            p_val2 = np.nonzero(A[indtmp]<m1)
            p_val2 = len(p_val2[0])/len(A[indtmp])
            p_val = (p_val1 + p_val2)/2
        else:
            p_val1 = np.nonzero(A[0]<m2)
            p_val1 = len(p_val1[0])/len(A[0])
            p_val2 = np.nonzero(A[indtmp]>m1)
            p_val2 = len(p_val2[0])/len(A[indtmp])
            p_val = (p_val1 + p_val2)/2
        pval.append(p_val)
    return pval

# function to create and save box plot
def plot_boxplot(areas_tmp, save_direc, title_label, fnum):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.boxplot(areas_tmp)
    plt.ylabel('Contribution of the frequency region to total variance')
    plt.xlabel('Time-window')
    plt.title(title_label)

    filename = save_direc + '/' + 'areas_boxplot_' + str(fnum) + '.png'
    plt.savefig(filename, dpi = 300)

    filename = save_direc + '/' + 'areas_boxplot_' + str(fnum) + '.svg'
    plt.savefig(filename)

    plt.close()
    return None
###############################################################################################################
direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final'

# frequency region under which areas will be computed
fregions = [[0, 1/365], [1/365, 1/120], [1/120, 1/30], [1/30, 1/15], [1/15, 1/2], [1/365, 1/30],[1/30, 1/2]]
# label of the frequency region under which areas will be computed
fregions_label = ['Greater than 1-year timescales', '4-months to 1-year timescales', '1-month to 4-months timescales', '2-weeks to 1-month timescales', 'Less than 2-weeks timescales', '1-month to 1-year timescales', 'Less than 1-month timescales']

nsamp = 1000         # number of samples to be drawn from the posterior distribution of parameters

# read the list of basins which need to be removed
fname = 'basin_list_significant_autocorr.txt'
filename = direc + '/' + fname
fid = open(filename,'r')
data = fid.readlines()
fid.close()
basin_list = []
for basin in data:
    basin_list.append(basin.split()[0])

# read the list of basins which need to be removed for other reasons
fname = 'basin_list_to_be_removed.txt'
filename = direc + '/' + fname
fid = open(filename,'r')
data = fid.readlines()
fid.close()
for basin in data:
    basin_list.append(basin.split()[0])  
#######################################################################################################
wlen = 3650
mstep = 365*3
N = 3650            # number of time steps at which streamflow is available in each window
fs = 1              # sampling frequency (number of samples per day)
fmax = fs/2         # maximum frequncy according to aliasing effect
#fregions = [[0, 2*10**(-3)], [2*10**(-3), 10**(-2)], [10**(-2), 2*10**(-2)],[2*10**(-2), 10**(-1)],[10**(-1), 2.5*10**(-1)],[2.5*10**(-1), 5*10**(-1)]]

# frequency values at which PSD is to be computed 
fint = 2*fmax/N
f = np.arange(0, fmax, fint)
f = f[1:]
one_sided = True

listFile = os.listdir(direc)

write_data = []
#for local_dir in [listFile[436]]:

## basins that already contain box plots
"""
listFile_rem  = []
count = 0
for basin in listFile:
    filename = direc + '/' + basin + '/' + 'areas_p_values.txt'
    if os.path.isfile(filename):
        count = count + 1
    else:
        listFile_rem.append(basin)
"""
###############################################################################################
def implement(local_dir, basin_list):

    print(local_dir)

    if os.path.isdir(direc + '/' + local_dir) and (local_dir not in basin_list) and (local_dir != 'FARIMA_ML_Models'):

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

        # read deseasonalized streamflow data
        fname = 'strm.txt'
        filename = direc + '/' + local_dir + '/' + fname
        fid = open(filename, 'r')
        data = fid.readlines()
        fid.close()
        deseason_strm = []
        for rind in range(1,len(data)):
            data_tmp = data[rind].split()
            deseason_strm.append(float(data_tmp[2]))
        deseason_strm = np.array(deseason_strm)

        N1 = len(deseason_strm)
        ## compute area under PSD for each frequency region and each time-window
        areas_window = [] 
        num_win = -1        # moving window number
        for ind in range(0,N1-wlen+1,mstep): # for loop fot time-windows
            num_win = num_win + 1           
            strm_tmp = deseason_strm[ind:ind+wlen]

            # compute periodogram
            f_tmp, I_tmp = FarimaModule.periodogramAG(strm_tmp,fs,True)
            f_tmp = f_tmp[1:]
            I_tmp = I_tmp[1:]

            # compute Hessian of Whittle's likelihood
            hess = FarimaModule.hessLw(p,q,coeffs[num_win,:],f_tmp,True,I_tmp)
            Sigma2 = np.linalg.inv(hess)

            # add nugget to ensure that the covarince matrix if positive definite
            #nug = 0.01*np.ones((Sigma2.shape[0],1))
            #Sigma2 = Sigma2 + np.diag(nug)
            
            # drawn samples from the posterior distribution of FARIMA model parameters
            samps = np.random.multivariate_normal(coeffs[num_win,:], Sigma2, size = (nsamp,1))

            # compute FARIMA theoretical periodogram for each of the drawn parameter sets
            pxx = []
            for sind in range(0,len(samps)):
                theta_tmp = samps[sind][0]
                arparams = theta_tmp[0:p]
                maparams = theta_tmp[p:p+q]
                d = theta_tmp[p+q]
                sigma_eps = theta_tmp[p+q+1]

                ar = np.r_[1, -arparams]
                ma = np.r_[1, maparams]
                pxx_tmp = FarimaModule.pxx_denFARIMA(p,d,q,ar,ma,f,sigma_eps,True)
                sigmax2 = simpson(pxx_tmp,2*np.pi*f, even = 'avg')
                pxx_tmp = pxx_tmp/sigmax2
                pxx.append(pxx_tmp)

            #for pind in range(0,len(pxx)):
            #    plt.loglog(f,pxx[pind])
            
            # compute area under each psd and under different frequency regions
            areas =  []
            for pxx_ind in range(0,len(pxx)):
                areas_samp = psdArea(pxx[pxx_ind], f, fregions)
                areas.append(areas_samp)
            areas_window.append(np.array(areas))
        
        # compute the p-value of between the first and last time-window
        # Let m1 = mean area under psd of the given frequency region in first time-window
        # Let m2 = mean area under psd of the given frequency region in last time-window
        # if m1 < m2 : p-value = probability that a1 > m2 (a1 = random variable denoting area under psd the given frequency region)
        # if m1 > m2 : p-value = probability that a1 < m2
        p = []      # list containig lists of p-value for each frequency region
        save_direc = direc + '/' + local_dir
        for find in range(0,len(fregions)): # for loop for frequency regions
            areas_tmp = []
            for twind in range(0,len(areas_window)): # for loop for time-windows
                areas_tmp.append(areas_window[twind][:,find])
            p_val_tmp = p_value(areas_tmp)
            plot_boxplot(areas_tmp, save_direc, fregions_label[find], find)
            p.append(p_val_tmp)    
        p = np.array(p).T

        # write p-value data to a textfile
        sname = 'areas_p_values.txt'
        filename = save_direc + '/' + sname
        fid = open(filename, 'w')
        fid.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%tuple(fregions_label))
        for wind in range(0,p.shape[0]):
            fid.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\n'%tuple(p[wind,:]))
        fid.close()

        # write areas_window data to a textfile

        for aind in range(0,len(areas_window)):
            areas_tmp = areas_window[aind]
            sname = 'areas_distribution_' + str(aind) + '.txt'
            filename = save_direc + '/' + sname
            fid = open(filename, 'w')
            fid.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%tuple(fregions_label))
            for wind in range(0,len(areas_tmp)):
                fid.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\n'%tuple(areas_tmp[wind,:]))
            fid.close()

    return None

#implement('01013500', basin_list)

# implement the script for different watersheds in parallel
if __name__ == '__main__':
    # start 10 worker processes
    inputs = [(fname,basin_list) for fname in listFile[300:]]
    inputs = inputs
    with mp.Pool(processes=10) as pool:
      results = pool.starmap(implement,inputs)
    pool.close()          



