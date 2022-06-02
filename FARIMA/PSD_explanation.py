"""
This script explors the relationship between contribtuion of different frequency regions to streamflow variance
and watershed hydrolo-meteorological properties

Author: Abhinav Gupta (Created: 30 March 2022)

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# read hydro data
direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/camels_attributes_v2.0/camels_attributes_v2.0'
fname = 'camels_hydro.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
hydro_tmp = []
for rind in range(0,len(data)):
    data_tmp = data[rind].split(';')
    hydro_tmp.append(data_tmp)

# read climate data
fname = 'camels_clim.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
clim_tmp = []
for rind in range(0,len(data)):
    data_tmp = data[rind].split(';')
    clim_tmp.append(data_tmp)

# read watershed area data
fname = 'camels_topo.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
topo_tmp = []
for rind in range(0,len(data)):
    data_tmp = data[rind].split(';')
    topo_tmp.append(data_tmp)

# read trend-in-area-under-PSD data
direc = 'D:/Research/non_staitionarity/codes/results/FARIMA_results_final'
 
fname = 'changePSD_statSgfcnc.txt'
filename = direc + '/' + fname
fid = open(filename)
data = fid.readlines()
fid.close()
farima = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    farima.append([data_tmp[0]] + [float(x) for x in data_tmp[1:-2]])

# read area under first time-window
y = []
col_ind = 3
for ind in range(0,len(farima)):
    basin = farima[ind][0]
    fname = 'area_optimal_param_set.txt'
    filename = direc + '/' + basin + '/' + fname
    fid = open(filename)
    data = fid.readlines()
    fid.close()
    data_tmp = data[1].split()
    y.append(float(data_tmp[col_ind]))
y = np.array(y)

# read d vlaues for watersheds
d = []
for ind in range(0,len(farima)):
    basin = farima[ind][0]
    fname = basin + '_coefficient_estimates_mean.txt'
    filename = direc + '/' + basin + '/' + fname
    fid = open(filename)
    data = fid.readlines()
    fid.close()
    dtmp = []
    for rind in range(1,len(data)):
        data_tmp = data[rind].split()
        dtmp.append(float(data_tmp[-2]))
    d.append(np.nanmean(dtmp))
d = np.array(d)


# identify rows in hydro, clim, and topo data that should be kept
hydro = []
clim = []
topo = []
for ind in range(0,len(farima)):
    basin = farima[ind][0]
    bind = [i for i in range(0,len(hydro_tmp)) if hydro_tmp[i][0] == basin]
    hydro.append(hydro_tmp[bind[0]][1:])

    bind = [i for i in range(0,len(hydro_tmp)) if clim_tmp[i][0] == basin]
    clim.append(clim_tmp[bind[0]][1:])

    bind = [i for i in range(0,len(topo_tmp)) if topo_tmp[i][0] == basin]
    topo.append(topo_tmp[bind[0]][1:])


hydro = np.array(hydro)
hydro = hydro.astype('float64')
clim = np.array(clim)
#clim = clim.astype('float64')
topo = np.array(topo)
topo = topo.astype('float64')
farima = np.array(farima)
farima = farima[:,1:]
farima = farima.astype('float64')

# plot data
x = topo[:,4]
x = x.astype('float64')
slope, _, rval, pval, _ = scipy.stats.linregress(x, d)
x1 = clim[:,3]
x1 = x1.astype('float64')
ind = np.nonzero(x1>0.40)
ind = ind[0]
plt.scatter(x, d)
plt.scatter(x[ind], d[ind])
plt.title('slope =' + str(round(slope*100)/100) + ', p = ' + str(round(pval*1000)/1000)  + ', R2 = ' + str(round((rval**2)*1000)/1000))
plt.grid(linestyle = '--')
# plt.show()

plt.xlabel('Drainage area (Km2)')
plt.ylabel(r'$d$')
plt.legend(['$f_{snow}<0.40$', '$f_{snow}>0.40$'], frameon = False)
#plt.show()

# save plot
sname = 'd_vs_area.svg'
direc = 'D:/Research/non_staitionarity/codes/results/miscellaneous_plots'
filename = direc + '/' + sname
plt.savefig(filename, dpi = 300)
