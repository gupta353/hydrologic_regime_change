"""
This script categorizes the CAMELS watersheds into:

(1) Purely rain driven
(2) Rain dominated
(3) Mixed rain and snow
(4) Snow dominated

Author: Abhinav Gupta (Created: 06 Jan 2022)

"""

def writeFiles(direc, sname, basin_list):
    filename  = direc + '/' + sname
    fid = open(filename, 'w')
    for wind in range(0,len(basin_list)):
        fid.write('%s\n'%basin_list[wind])
    return None

direc = 'D:/Research/non_staitionarity/codes/results/rain_snow_dominated'


# read data
fname = 'rain_snow_dominated_combined.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
metrices = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    metrices.append([data_tmp[0], float(data_tmp[1]), float(data_tmp[2]), float(data_tmp[3]), float(data_tmp[4]), float(data_tmp[5]), float(data_tmp[6])])

# purely rain driven watersheds
ind = [i for i in range(0,len(metrices)) if metrices[i][3]<1 and metrices[i][4]<0.10 and metrices[i][6]<0.10] 
watersheds = [metrices[i][0] for i in ind]
sname = 'unresponsive_watersheds_1.txt'
#writeFiles(direc, sname, watersheds)

a = 1
