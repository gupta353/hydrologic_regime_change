"""
This script combines GLEAM actual evaporation data such that all the teporal data corresponding to a location is saved to one textfile.

Author: Abhinav Gupta (Created: 8 Sep 2021)
"""
import os

# directory containing daily data
direc = 'D:/Research/non_staitionarity/data/GLEAMS'
years = range(1980,2021,1)

# create a new directory in the GLEAMS folder
new_dir = 'evaporation_data_combined'
os.mkdir(direc + '/' + new_dir)
for count in range(1,2526,1):
    fname = 'E_' + str(count) + '.txt'

    evap = [] # a list containing evaporation data with corresponding day of the year and the year
    for year_tmp in years:
        dir_tmp = direc + '/' + str(year_tmp)
        filename = dir_tmp + '/' + fname
        
        # read textfile
        fid = open(filename,'r')
        data = fid.readlines()
        fid.close()

        # split 'data'
        for rind in range(2,len(data),1):
            evap_tmp = data[rind].split('\n')
            evap.append([year_tmp,rind-1,float(evap_tmp[0])])

    # write data to a textfile
    fname = 'E_' + str(count) + '_combined.txt'
    filename = direc + '/' + 'evaporation_data_combined' + '/' + fname
    fid = open(filename,'w')
    fid.write('Year\tDay\tEvaporation(mm/day)\n')
    for wind in range(0,len(evap)):
        fid.write("%d\t%d\t%f\n"% (evap[wind][0],evap[wind][1],evap[wind][2]))      