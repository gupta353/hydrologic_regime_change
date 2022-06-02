"""
This script prepares static dataset for RF regression

Author: Abhinav Gupta (Created: 19 Apr 2021)

"""

direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/camels_attributes_v2.0/camels_attributes_v2.0'

# climatic data
fname = 'camels_clim.txt'
filename = direc + '/' + fname
fid = open(filename)
data = fid.readlines()
fid.close()
clim = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split(';')
    clim.append([data_tmp[0]] + [float(x) for x in data_tmp[1:8] + [float(data_tmp[9]), float(data_tmp[10])]])

# vegetation data
fname = 'camels_vege.txt'
filename = direc  + '/' + fname
fid = open(filename)
data = fid.readlines()
fid.close()
veg = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split(';')
    veg.append([data_tmp[0]] + [float(x) for x in data_tmp[1:6]])

# topographical data
fname = 'camels_topo.txt'
filename = direc + '/' + fname
fid = open(filename)
data = fid.readlines()
fid.close()
topo = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split(';')
    topo.append([data_tmp[0]] + [float(x) for x in data_tmp[3:-1]])

# soil data
fname = 'camels_soil.txt'
filename = direc + '/' + fname
fid = open(filename)
data = fid.readlines()
fid.close()
soil = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split(';')
    soil.append([data_tmp[0]] + [float(x) for x in data_tmp[1:-3]])

# geological properties
fname = 'camels_geol.txt'
filename = direc + '/' + fname
fid = open(filename)
data = fid.readlines()
fid.close()
geo = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split(';')
    geo.append([data_tmp[0], float(data_tmp[5]), float(data_tmp[7])])
    
# arrange all data in one list
write_data= []
for wind in range(0, len(veg)):
    basin = veg[wind][0]
    ind_clim = [i for i in range(0,len(clim)) if clim[i][0]==basin]
    ind_clim = ind_clim[0]
    ind_tp = [i for i in range(0,len(topo)) if topo[i][0]==basin]
    ind_tp = ind_tp[0]
    ind_sl = [i for i in range(0,len(soil)) if soil[i][0]==basin]
    ind_sl = ind_sl[0]
    ind_geo = [i for i in range(0,len(geo)) if geo[i][0]==basin]
    ind_geo = ind_geo[0]

    write_data.append(clim[ind_clim] + veg[wind][1:] + topo[ind_tp][1:] + soil[ind_sl][1:] + geo[ind_geo][1:])

# write data to a textfile
direc_save = 'D:/Research/non_staitionarity/data/RF_static_data'
sname = 'RF_static.txt'
filename = direc_save + '/' + sname
fid = open(filename, 'w')
fid.write('basin\tp_mean\tpet_mean\tp_seasonality\tfrac_snow\taridity\thigh_prcp_freq\thigh_prcp_dur\tlow_prcp_freq\tlow_prcp_dur\tfrac_forest\tlai_max\tlai_diff\tgvf_max\tgvf_diff\telev_mean\tslope_mean\tarea_gages2\tsoil_depth_pelletier\tsoil_depth_statsgo\tsoil_porosity\tsoil_conductivity\tmax_water_content\tsand_frac\tsilt_frac\tclay_frac\tcarbonate_rock_frac\tgeo_permeability_log10\n')
formatspec = '%s\t' + '%f\t'*26 + '%f\n'
for wind in range(0,len(write_data)):
    fid.write(formatspec%tuple(write_data[wind]))
fid.close()