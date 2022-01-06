"""
This script reads rainfall_runoff_data.nc files 

Author: Abhinav Gupta (Created: 20 Dec 2021)

"""
import netCDF4 as nc

direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis'

basin = '01013500'
fname = 'rainfall_runoff.mat'
filename = direc + '/' + basin + '/' + fname

ds = nc.Dataset(filename)

numPeriods = ds['numPeriods'][:]

