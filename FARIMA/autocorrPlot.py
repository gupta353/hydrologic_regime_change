"""
This script reads autocorrelation data for each time-window and for each watershed and plots the boxplot of autocorrelation upto 30 lags 

Author: Abhinav Gupta (Created on: 29 Nov 2021)

"""
import os
import numpy as np
import matplotlib.pyplot as plt

direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final'
window = 9      # time-window for which auto-correlationd data is to be plotted 

corr_thresh = 0.15  # threshold below which correlation is assumed negligible
numTimeLags = 30    # number of time lags upto which auto-correlations are considered
zero_thresh = 1     # number of time-windows where authocorrelation was greater than the threshold for atleast one time lag  

listFile = os.listdir(direc)

# read the list of basins for which the FARIMA model could not be validated
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

# read correlation data
fname = 'autocorrelations_means.txt'
corr_list = []
for local_dir in listFile:

    if os.path.isdir(direc + '/' + local_dir) and (local_dir not in basin_list) and (local_dir != 'FARIMA_ML_Models'):
        
        filename = direc + '/' + local_dir + '/' + fname

        fid = open(filename)
        data = fid.readlines()
        fid.close()

        corrs = []

        for corr_ind in range(1,len(data)):
            corrs.append(data[corr_ind].split())
        corrs = np.array(corrs)
        corrs = corrs.astype('float')
        corrs  = np.delete(corrs,0,0)
        corrs = corrs[0:numTimeLags,:]

        corr_list.append(corrs)
           
# plot data
plot_data = []
for ind in range(0,len(corr_list)):
    if (corr_list[ind].shape[1] - 1 >= window-1):
        plot_data.append(corr_list[ind][:,window-1])
plot_data = np.array(plot_data)

# create plot
plt.figure(figsize=(10,5))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
flierprops = dict(marker='o',  markersize=2, linestyle='none')
medianprops = dict(color = 'Tab:Blue')

for lag in range(0,plot_data.shape[1]):
    filtered_data = plot_data[~np.isnan(plot_data[:,lag]),lag]
    plt.boxplot(filtered_data, positions = [lag+1], flierprops = flierprops, medianprops = medianprops, widths = 0.5)
plt.xlabel('Time lags (Days)')
plt.ylabel('Auto-correlation')
plt.text(1, 0.14, 'Time window = ' + str(window))

# save plot
sname = 'autocorr_window_' + str(window) + '.svg'
filename = direc + '/' + sname
plt.savefig(filename, dpi = 300)

sname = 'autocorr_window_' + str(window) + '.png'
filename = direc + '/' + sname
plt.savefig(filename, dpi = 300)

plt.show()
plt.close()


