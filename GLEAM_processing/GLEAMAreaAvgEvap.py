

"""
This script computes areal average evaporation (using GLEAM data source) for all the CAMELS basins
Author: Abhinav Gupta (Created: 9 Sep 2021)

"""
import numpy as np
import os

direc = 'D:/Research/non_staitionarity/data/GLEAMS'

# create a new folder to save textfiles
new_folder_name = 'CAMELS_areal_avg_evaporation'
os.mkdir(direc + '/' + new_folder_name)

## read instersection data
fname = 'GLEAM_grids_HCDN_intersect1_ALber_Equ_area.txt'
filename = direc + '/USA_grid_lat_long/' + fname
fid = open(filename,'r')
data = fid.readlines()
fid.close()

intersect_data = []
# Extract HCDN-FID, hru_id of CAMELS basins and (lat, long) of GLEAM grids
for rind in range(1,len(data),1):
    data_tmp = data[rind].split()
    intersect_data.append([int(data_tmp[1]),int(data_tmp[2]),float(data_tmp[16]),float(data_tmp[17]),float(data_tmp[19])]) # 1st entry: HCDN_FID, 2nd entry: hru_id, 3rd entry: Longitude, 4th entry: Latitude, 5th entry: area (in sq. meters) 

intersect_data = np.array(intersect_data)
#intersect_data = intersect_data[intersect_data[:,0].argsort(),:]

## read lat-long data of the centroids of the grids (which intersect with CAMELS basins) with their serial number
fname = 'lat_lon_data.txt'
filename = direc + '/evaporation_data_combined/' + fname
fid = open(filename)
data = fid.readlines()
fid.close()
serial_data = []
for rind in range(1,len(data),1):
    data_tmp = data[rind].split()
    serial_data.append([int(data_tmp[0]),float(data_tmp[1]),float(data_tmp[2])])
serial_data = np.array(serial_data)

## compute areal average for each CAMELS hru_unit
un_hru_fids = np.unique(intersect_data[:,0])
un_hru_fids = np.ndarray.tolist(un_hru_fids) 
for hru_fid in un_hru_fids:
    
    inds = [i for i in range(0,len(intersect_data[:,0])) if (intersect_data[i,0] == float(hru_fid))]  # find indices corresponding to 'hru_fid' in intersect_data

    # identify GLEAM textfile serial numbers for each grid that intersect with the polygon with 'fid = hru_fid' 
    serial = []
    area = []
    for ind_tmp in inds:
        
        hru_id = intersect_data[int(ind_tmp),1]  # hru id of CAMELS_basin
        serial_tmp = [serial_data[i,0] for i in range(0,len(serial_data[:,0])) if (serial_data[i,1] == intersect_data[ind_tmp,2] and serial_data[i,2] == intersect_data[ind_tmp,3])]
        serial.append(serial_tmp[0])
        area.append(intersect_data[ind_tmp,4])

        # read evapotranspiration data for each serial number
        evap = []
        date = []
        for sind in serial:

            
            # read evaporation data
            fname = 'E_' + str(int(sind)) + '_combined.txt'
            filename = direc + '/evaporation_data_combined/' + fname
            file_fid = open(filename,'r')
            data = file_fid.readlines()
            file_fid.close()
            data = [data[i].split() for i in range(1,len(data))]
            evap_tmp = [float(data[i][2]) for i in range(0,len(data))]
            evap.append(evap_tmp)
            date_tmp = np.array(data)[:,0:2]
            date.append(date_tmp)

    date = date[0]
    # computation of areal average evapotranspiration
    area_avg_evap = np.empty(len(evap[0]), dtype = float)
    area = np.array(area)
    for eind in range(0,len(evap)):
        area_avg_evap = np.array(evap[eind])*area[eind] + area_avg_evap
    area_avg_evap = area_avg_evap/np.sum(area)

    # write areal average evaporation to a textfile
    fname = str(int(hru_id)) + '_evap_areal_average.txt'
    filename = direc + '/' + new_folder_name + '/' + fname
    wfid = open(filename,'w')
    wfid.write('Year\tDay\tAreal_average_evaporation(mm/day)\n')
    for wind in range(0,len(area_avg_evap)):
        wfid.write('%d\t%d\t%f\n' %(int(date[wind][0]),int(date[wind][1]),area_avg_evap[wind]))
    wfid.close()