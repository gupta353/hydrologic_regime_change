"""

This script extracts GLEAM actual evaporation data for the grids that intersect with CAMELS basins for the years 1980 to 1981

Author: Abhinav Gupta (Created: 6 Sep 2021)
"""

import netCDF4 as nc
import numpy as np
import re

# directory containing daily data
direc = 'D:/Research/non_staitionarity/data/GLEAMS'

# define lat-long thresholds (in decimal degrees)
lon_west_thresh = -125.51
lon_east_thresh = -66.58
lat_north_thresh = 49.56
lat_south_thresh = 26.00

# read lat-long data of the centroids of the grids that are required to compute areal average evaporation in basins contained in CAMELS dataset
fname = 'GLEAM_grids_HCDN_intersect1_ALber_Equ_area.txt'
filename = direc + '/' + 'USA_grid_lat_long' + '/' + fname
fid = open(filename, 'r')
lat_lon_data_lst = fid.readlines()
fid.close()
# extract lat-long data
lon_lat_data = []
lon_data = []
lat_data = []
for ind in range(1, len(lat_lon_data_lst)):
    lat_lon_data_tmp = lat_lon_data_lst[ind].split()
    lon_data.append(float(lat_lon_data_tmp[16]))
    lat_data.append(float(lat_lon_data_tmp[17]))
    lon_lat_data.append([lon_data[-1], lat_data[-1]])

years = range(1980, 2021, 1) # years for which evaporation is to be read

# read netcdf file for all year in a for loop
for year_tmp in [1980]:#years:
    year = str(year_tmp)
    fn = direc + '/' + year + '/' + 'E_' + year + '_GLEAM_v3.5a.nc'
    ds = nc.Dataset(fn)

    # Extract lat-long and evapotranspiration information
    lon = ds['lon']
    lat = ds['lat']
    E = ds['E']
    del(ds)

    # identify indices of lat-long in the netcdf file which belong to centroids of grids relevant to CAMELS data basins
    lon_inds = [i for i in range(0, len(lon)) if (lon[i] in lon_data)]
    lat_inds = [i for i in range(0, len(lat)) if (lat[i] in lat_data)]
    lon_lat_inds = [(i, j) for i in lon_inds for j in lat_inds if ([lon[i], lat[j]] in lon_lat_data)]

    # create separate files for a particular lat-long
    count = 0                               # counter for the sequence number of textfile
    for ind in range(0, len(lon_lat_inds)):

        lon_ind = lon_lat_inds[ind][0]
        lat_ind = lon_lat_inds[ind][1]
        lon_tmp = lon[lon_ind]
        lat_tmp = lat[lat_ind]
        E_tmp = E[:, lon_ind, lat_ind]

        # write data to a textfile
        count = count + 1
        fname = 'E_' + str(count) + '.txt'
        filename = direc + '/' + str(year) + '/' + fname
        # lat-long of the centroids of the grid to be written in the text file
        header_1 = 'Longitude = ' + \
            str(lon_tmp) + '\t' + 'Latitude = ' + str(lat_tmp)

        fid = open(filename, 'w')
        fid.write(header_1+'\nActual_evaporation(mm/day)\n')
        for find in range(0, len(E_tmp)):
            fid.write("%f\n" % (E_tmp[find]))
        fid.close()