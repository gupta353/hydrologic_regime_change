"""
This script reads rainfall (from CAMELS), evapotranspiration (from GLEAM), snow-water-equivalent (from CAMELS) and streamflow data (from CAMELS) and combined them into one textfile for each CAMELS basin
All the three rainfall and swe products - daymet, maurer, and nldas - are read

Author: Abhinav Gupta (Created: 13 Sep 2021)

"""
import datetime
import numpy as np
import operator
from operator import indexOf

class Wildcard:
    def __eq__(self, anything):
        return True

wc = Wildcard()

direc_CAMELS = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2'
direc_GLEAMS = 'D:/Research/non_staitionarity/data/GLEAMS/'

# directory in which the data will be saved
direc_save = 'D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data'

# read the list of basin hrus
fname = 'GLEAM_grids_HCDN_intersect1_ALber_Equ_area.txt'
filename = direc_GLEAMS + '/USA_grid_lat_long/' + fname
fid = open(filename,'r')
data  = fid.readlines()
fid.close()
hru_ids = [data[i].split()[2] for i in range(1,len(data))]
hru_ids = list(set(hru_ids))
del(data)

# read rainfall, evapotranspiration, swe, and streamflow data for each 'hru_id'
for hru_id_tmp in hru_ids:

    HUC_2 = hru_id_tmp
    if (len(hru_id_tmp) == 7):
        HUC_2 = '0' + HUC_2

    # rainfall and swe from daymet
    fname = HUC_2 + '_lump_cida_forcing_leap.txt'
    filename = direc_CAMELS + '/basin_mean_forcing/daymet/common_directory/' + fname
    fid = open(filename,'r')
    data = fid.readlines()
    fid.close()
    daymet_rainfall_swe = []
    for ind in range(4,len(data)):
        daymet_rainfall_swe_tmp = data[ind].split()
        #daymet_rainfall_swe = [data[i].split() for i in range(4,len(data))]
        daymet_rainfall_swe.append([int(daymet_rainfall_swe_tmp[0]),int(daymet_rainfall_swe_tmp[1]),int(daymet_rainfall_swe_tmp[2]),daymet_rainfall_swe_tmp[5],daymet_rainfall_swe_tmp[7]]) # entries in order: year, month, day, prcp, swe

    # rainfall and swe from maurer
    fname = HUC_2 + '_lump_maurer_forcing_leap.txt'
    filename = direc_CAMELS + '/basin_mean_forcing/maurer/common_directory/' + fname
    fid = open(filename,'r')
    data = fid.readlines()
    fid.close()
    maurer_rainfall_swe = []
    for ind in range(4,len(data)):
        maurer_rainfall_swe_tmp = data[ind].split()
        #daymet_rainfall_swe = [data[i].split() for i in range(4,len(data))]
        maurer_rainfall_swe.append([int(maurer_rainfall_swe_tmp[0]),int(maurer_rainfall_swe_tmp[1]),int(maurer_rainfall_swe_tmp[2]),maurer_rainfall_swe_tmp[5],daymet_rainfall_swe_tmp[7]]) # entries in order: year, month, day, prcp, swe

    # rainfall and swe from nldas
    fname = HUC_2 + '_lump_nldas_forcing_leap.txt'
    filename = direc_CAMELS + '/basin_mean_forcing/nldas/common_directory/' + fname
    fid = open(filename,'r')
    data = fid.readlines()
    fid.close()
    nldas_rainfall_swe = []
    for ind in range(4,len(data)):
        nldas_rainfall_swe_tmp = data[ind].split()
        #daymet_rainfall_swe = [data[i].split() for i in range(4,len(data))]
        nldas_rainfall_swe.append([int(nldas_rainfall_swe_tmp[0]),int(nldas_rainfall_swe_tmp[1]),int(nldas_rainfall_swe_tmp[2]),nldas_rainfall_swe_tmp[5],nldas_rainfall_swe_tmp[7]]) # entries in order: year, month, day, prcp, swe

    # read streamflow data
    fname = HUC_2 + '_streamflow_qc.txt'
    filename = direc_CAMELS + '/usgs_streamflow/common_directory/' + fname
    fid = open(filename,'r')
    data = fid.readlines()
    fid.close()
    strm = []
    for ind in range(0,len(data)):
        strm_tmp = data[ind].split()
        strm.append([int(strm_tmp[1]),int(strm_tmp[2]),int(strm_tmp[3]),strm_tmp[4],strm_tmp[5]]) # entries in order: year, month, day, streamflow, quality flag    

    # read evapotranspiration data
    fname = hru_id_tmp + '_evap_areal_average.txt'
    filename = direc_GLEAMS + '/CAMELS_areal_avg_evaporation/' + fname
    fid = open(filename,'r')
    data = fid.readlines()
    fid.close()
    evap = []
    for ind in range(1,len(data)):
        evap.append(data[ind].split()) # entries in order: Year, day, evapotranspiration

    # fill the missing values with NaNs such that every type of data has entries upto 31 Dec 2020
    # daymet rainfall-swe
    last_date = daymet_rainfall_swe[-1]
    if (int(last_date[0]) != 2020 or int(last_date[1]) != 12 or int(last_date[2]) != 31):
        date_begin = datetime.date(int(last_date[0]),int(last_date[1]),int(last_date[2]))
        date_end = datetime.date(2020,12,31)
        datenum_begin = date_begin.toordinal() + 1
        datenum_end = date_end.toordinal()
        for datenum_tmp in range(datenum_begin,datenum_end+1):
            date_tmp = datetime.date.fromordinal(datenum_tmp)
            daymet_rainfall_swe.append([date_tmp.year,date_tmp.month,date_tmp.day,'nan','nan'])

    # maurer rainfall-swe
    last_date = maurer_rainfall_swe[-1]
    if (int(last_date[0]) != 2020 or int(last_date[1]) != 12 or int(last_date[2]) != 31):
        date_begin = datetime.date(int(last_date[0]),int(last_date[1]),int(last_date[2]))
        date_end = datetime.date(2020,12,31)
        datenum_begin = date_begin.toordinal() + 1
        datenum_end = date_end.toordinal()
        for datenum_tmp in range(datenum_begin,datenum_end+1):
            date_tmp = datetime.date.fromordinal(datenum_tmp)
            maurer_rainfall_swe.append([date_tmp.year,date_tmp.month,date_tmp.day,'nan','nan'])

    # nldas rainfall-swe
    last_date = nldas_rainfall_swe[-1]
    if (int(last_date[0]) != 2020 or int(last_date[1]) != 12 or int(last_date[2]) != 31):
        date_begin = datetime.date(int(last_date[0]),int(last_date[1]),int(last_date[2]))
        date_end = datetime.date(2020,12,31)
        datenum_begin = date_begin.toordinal() + 1
        datenum_end = date_end.toordinal()
        for datenum_tmp in range(datenum_begin,datenum_end+1):
            date_tmp = datetime.date.fromordinal(datenum_tmp)
            nldas_rainfall_swe.append([date_tmp.year,date_tmp.month,date_tmp.day,'nan','nan'])

    # streamflow
    last_date = strm[-1]
    if (int(last_date[0]) != 2020 or int(last_date[1]) != 12 or int(last_date[2]) != 31):
        date_begin = datetime.date(int(last_date[0]),int(last_date[1]),int(last_date[2]))
        date_end = datetime.date(2020,12,31)
        datenum_begin = date_begin.toordinal() + 1
        datenum_end = date_end.toordinal()
        for datenum_tmp in range(datenum_begin,datenum_end+1):
            date_tmp = datetime.date.fromordinal(datenum_tmp)
            strm.append([date_tmp.year,date_tmp.month,date_tmp.day,'nan','nan'])
    
    first_date = strm[0]
    if (int(first_date[0]) != 1980 or int(first_date[1]) != 1 or int(first_date[2]) != 1):
        date_begin = datetime.date(1980,1,1)
        date_end = datetime.date(int(first_date[0]),int(first_date[1]),int(first_date[2]))
        datenum_begin = date_begin.toordinal()
        datenum_end = date_end.toordinal()-1
        strm_initial = []
        for datenum_tmp in range(datenum_begin,datenum_end+1):
            date_tmp = datetime.date.fromordinal(datenum_tmp)
            strm_initial.append([date_tmp.year,date_tmp.month,date_tmp.day,'nan','nan'])
        strm_initial.extend(strm)
        strm = strm_initial

    # write all the data in one list
    write_data = []
    for wind in range(0,len(evap)):

        # identify the different data of same date
        year_evap = evap[wind][0]
        day_evap = evap[wind][1]
        date_evap = datetime.datetime.strptime(year_evap + ' ' + day_evap,"%Y %j")
        year_tmp = date_evap.year
        month_tmp = date_evap.month
        day_tmp = date_evap.day

        # identify rainfall indices with same date as evapotranspiration
        daymet_ind = daymet_rainfall_swe.index([year_tmp,month_tmp,day_tmp,wc,wc])
        maurer_ind = maurer_rainfall_swe.index([year_tmp,month_tmp,day_tmp,wc,wc])
        nldas_ind = nldas_rainfall_swe.index([year_tmp,month_tmp,day_tmp,wc,wc])

        # identify streamflow index with same date as evapotranspiration
        strm_ind = strm.index([year_tmp,month_tmp,day_tmp,wc,wc])

        daymet_ind = daymet_ind
        maurer_ind = maurer_ind
        nldas_ind = nldas_ind
        strm_ind = strm_ind
        write_data.append([year_tmp,month_tmp,day_tmp,daymet_rainfall_swe[daymet_ind][3],maurer_rainfall_swe[maurer_ind][3],nldas_rainfall_swe[nldas_ind][3],daymet_rainfall_swe[daymet_ind][4],maurer_rainfall_swe[maurer_ind][4],nldas_rainfall_swe[nldas_ind][4],strm[strm_ind][3],strm[strm_ind][4],evap[wind][2]])
 
    # write the data to a textfile
    fname = HUC_2 + '_GLEAMS_CAMELS_data.txt'
    filename = direc_save + '/' + fname
    fid = open(filename,'w')
    fid.write('Year\tMonth\tDay\tprcp_daymet(mm/day)\tprcp_maurer(mm/day)\tprcp_nldas(mm/day)\tswe_daymet(mm)\tswe_maurer(mm)\tswe_nldas(mm)\tstreamflow(CFS)\tstreamflow_quality_flag\tevapotranspiration_gleams(mm/day)\n')
    for wind in range(0,len(write_data)):
        fid.write("%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\t%f\n" %(int(write_data[wind][0]),int(write_data[wind][1]),int(write_data[wind][2]),float(write_data[wind][3]),\
            float(write_data[wind][4]),float(write_data[wind][5]),float(write_data[wind][6]),float(write_data[wind][7]),float(write_data[wind][8]),float(write_data[wind][9]),write_data[wind][10],float(write_data[wind][11])))
    fid.close()
