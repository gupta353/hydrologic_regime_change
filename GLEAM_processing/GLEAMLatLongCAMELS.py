"""
This script extracts lat-long of the centroids of GLEAM grids that intersect with CAMELS basins 

"""
import os
import re
direc = 'D:/Research/non_staitionarity/data/GLEAMS'

# number of distinct grids
dir_list = direc + '/1980'
list = os.listdir(dir_list)
numFiles = len(list)

# read lat-longs of each grid in a for loop
lat_lon_data = []
for count in range(1,numFiles-9,1):
    
    # read textfile
    fname = 'E_'+ str(count) + '.txt'
    filename  = direc + '/1980/' + fname
    fid = open(filename,'r')
    data = fid.readlines()
    fid.close()

    # read longitude and latitude from 'data'
    vals = re.split('\t|\n',data[0])
    lon_tmp = float(vals[0].split('Longitude = ')[1])
    lat_tmp = float(vals[1].split('Latitude = ')[1])
    lat_lon_data.append([count,lon_tmp,lat_tmp])

# write data to a textfile
fname = 'lat_lon_data.txt'
filename = direc + '/evaporation_data_combined/' + fname
fid = open(filename,'w')
fid.write('Serial_number\tLongitude\tLatitude\n')
for wind in range(0,len(lat_lon_data)):
    fid.write('%d\t%f\t%f\n' %(lat_lon_data[wind][0],lat_lon_data[wind][1],lat_lon_data[wind][2]))
fid.close()