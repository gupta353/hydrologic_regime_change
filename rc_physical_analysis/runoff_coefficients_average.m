%{
Idenitfy avrege of runoff coefficients and average of 30 largest runoff
coefficients

%}

clear all
close all
clc

% read lat long data
direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata';
fname = 'gauge_information.txt';
filename = fullfile(direc,fname);

fid = fopen(filename,'r');
data = textscan(fid,'%s%s%s%f%f%f','delimiter','\t','headerlines',1);
fclose(fid)

basin_list = data{2};
lats = data{4};
longs = data{5};

%% list of basins
direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis';
list = dir(direc);

fname = 'rainfall_runoff_data.mat';
count = 0;
for dir_ind = 3:length(list)
    
    local_dir = list(dir_ind).name;
    filename = fullfile(direc, local_dir, fname);
    
    if isfolder(fullfile(direc,local_dir)) && isfile(filename)
       
        count = count + 1;
        % read runoff coefficients of each rainfall-runoff hydrograph
        filename = fullfile(direc, local_dir, fname);
        load(filename);

        % Extract runoff coefficients
        for per_ind = 1:length(period)
            rc(per_ind) = period{per_ind}.runoff_coefficient;
        end
        
        % remove infs from the data
        rc(isinf(rc)) = [];
        
        rc_avg(count) = mean(rc); % compute average of runoff coefficients

        % Identify the 40 largest runoff-coefficients
        rc_sort = sort(rc);
        rc_sort_avg(count) = mean(rc_sort(end-39:end));
        
        % number of runoff-coefficients greater than 1
        num_rc_greater_one = sum(rc>1);
        
        ind = strcmp(basin_list, local_dir);
        lat = lats(ind);
        long = longs(ind);
    
        basin_write_data{count} = local_dir;
        write_data(count,:) = [rc_avg(count), rc_sort_avg(count), num_rc_greater_one, lat, long];
    
    end
    
end

% save data
sname = 'avg_rc.txt';
filename = fullfile(direc, sname);
fid = fopen(filename, 'w');
fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\n','gage_id', 'average_runoff_coeff', 'average_runoff_coeff_high_vals', 'Num_rc_greater_1', 'Lat', 'Long');
for wind = 1:size(write_data,1)
    fprintf(fid, '%s\t%f\t%f\t%f\t%f\t%f\n', basin_write_data{wind}, write_data(wind,:));
end
fclose(fid);

