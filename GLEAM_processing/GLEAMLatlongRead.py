"""
This file extracts lat-long values of the centroids of the grids contained in coterminous USA

Author: Abhinav Gupta (Created: 6 Sep 2021)

"""

import netCDF4 as nc

# directory containing daily data
direc = 'D:/Research/non_staitionarity/data/GLEAMS'

# define lat-long thresholds (in decimal degrees)
lon_west_thresh = -125.51
lon_east_thresh = -66.58
lat_north_thresh = 49.56
lat_south_thresh = 26.00
 
# read netcdf file for a year
year = '1980'
fn = direc + '/' + year + '/' + 'E_' + year + '_GLEAM_v3.5a.nc'
ds = nc.Dataset(fn)

# Extract lat-long and evapotranspiration information
lon = ds['lon']
lat = ds['lat']

# craete a list of lat-longs falling in USA
geoCoords = []
for lon_ind in range(0,len(lon)):
   for lat_ind in range(0,len(lat)):
       lat_tmp = lat[lat_ind]
       lon_tmp = lon[lon_ind]
       
       if (lon_tmp>=lon_west_thresh and lon_tmp<=lon_east_thresh and lat_tmp<=lat_north_thresh and lat_tmp>=lat_south_thresh):
           geoCoords.append([lon_tmp,lat_tmp])
           

# save lat-long data to a textfile
fname = direc+'/USA_grid_centroid_lat_long.txt'
fid = open(fname,'w')
fid.write('longitude\tlatitude\n')
for count in range(0,len(geoCoords)):
    fid.write("%f\t%f\n" % (geoCoords[count][0],geoCoords[count][1]))
fid.close()
