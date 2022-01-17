"""
This script combines snow-rain-dominated differentiation data from different files into one textfile

Author: Abhinav Gupta (Created: 06- Dec -2021)

"""

direc = 'D:/Research/non_staitionarity/codes/results/rain_snow_dominated'

# filenames to be read
fnames = ['rain_snow_dominated_' + str(i) + '.txt' for i in range(0,13)]

# read data from each file
write_data = []
for fname in fnames:
    
    filename = direc + '/' + fname
    fid = open(filename, 'r')
    data = fid.readlines()
    fid.close()
    for rind in range(1,len(data)):
        data_tmp = data[rind].split('\t')
        write_data.append([data_tmp[0], float(data_tmp[1]), float(data_tmp[2]), float(data_tmp[3]), float(data_tmp[4]), float(data_tmp[5]), float(data_tmp[6])])

# write data to textfile
sname = 'rain_snow_dominated_combined.txt'
filename = direc + '/' + sname
fid = open(filename, 'w')
fid.write('Gauge\tLat\tLong\tMaximum_swe_regime\tRatio_of_maximum_swe_to_tot_strm\tRatio_of_total_rain_to_tot_strm\tCorr_rain_strm\n')
for wind in range(0,len(write_data)):
    fid.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\n'%tuple(write_data[wind]))
fid.close()
