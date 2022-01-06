%{
This script identifies the watersheds for which the hydrograph separation
could not be carrried out using the parameters
rec_length_thresh=4;                % minimum lenght of recession period to be considered (in days)
prcp_strm_ratio_thresh=0.1;         % upper limit for precipitation to streamflow ratio (recommended <=0.1)
evap_strm_ratio_thresh_pos=0.1;     % upper limit for positive evaporation to streamflow ratio (recommended <=0.1)
evap_strm_ratio_thresh_neg=-0.1;    % lower limit for negative evaporation to streamflow ratio (recommended <=-0.1)
length_thresh = 5;
%}

clear all
close all
clc

direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis';
darea_direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata';

% drainage area
filename = fullfile(darea_direc,'gauge_information.txt');
fid = fopen(filename,'r');
data = textscan(fid,'%s%s%s%f%f%f','delimiter','\t','headerlines',1);
fclose(fid);
stations = data{2};
dareas = data{6};

list = dir(direc);

count  = 0;
nancount = 0;
darea_list = [];
station_list = {};
for station_ind = 3:length(list)
    
    station_id = list(station_ind).name;
    direc_tmp = fullfile(direc, station_id);
    if isfolder(direc_tmp)
        fname = ['MRC_',station_id,'.txt'];
        filename = fullfile(direc_tmp,fname);
        if ~isfile(filename)
            count = count+1;
            dind = find(strcmp(stations,station_id));
            darea = dareas(dind);    % in km^{2}
            darea_list(count) = darea;
            station_list{count} = station_id;
        end
        fname = ['NaNError.txt'];
        filename = fullfile(direc_tmp,fname);
        if isfile(filename)
            nancount = nancount+1;
        end
    end
    
end