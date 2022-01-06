%{
This script optimizes the parameters of SCS-CN  method and unit hydrograph
for a rainfall runoff event,
inputs: Rainfall, baseflow

Author: Abhinav Gupta (Created: 27 Dec 2021)

%}

clear all
close all
clc

% drainage area
darea_direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata';
filename = fullfile(darea_direc,'gauge_information.txt');
fid = fopen(filename,'r');
data = textscan(fid,'%s%s%s%f%f%f','delimiter','\t','headerlines',1);
fclose(fid);
stations = data{2};
dareas = data{6};
clear data
 
direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis';
input_direc = 'D:/Research/non_staitionarity/codes/rc_physical_analysis/complete_model';
    
list = dir(direc);

% function baseflow = baseflow_separation()
% lambda = 0.2;
% CN = 80;
% S = 25400/CN - 254; % potential maximum retention in mm
% Ia = lambda*S;      % initial abstraction
D = 1;              % timescale (in days) - for D-duration unit hydrograph
kmax = 1000;          % number of time-steps at which D-duration unit hydrograph will be computed
% alpha = 1;
% beta = 1/2;

for list_ind = 398%:length(list)
    
    basin = list(list_ind).name;
    
    %% determine drainage area
    ind = strcmp(stations, basin);
    darea = dareas(ind);        % in km2
    
    %% load data
    fname = 'rainfall_runoff_data.mat';
    filename = fullfile(direc, basin, fname);
    load(filename);
    t0 = tic;
   %% process each rainfall-runoff event in a for loop
    parfor per_ind = 1:100%length(period)
        
        [qest, Pe_est, theta_opt] = complete_model_parProcess(period{per_ind}, D, kmax, darea);
%         % read data
%         P = period{per_ind}.rain;
%         strm = period{per_ind}.completed_streamflow/darea/1000*24*3600;     % conversion from cms to mm/day
%         baseflow = period{per_ind}.baseflow/darea/1000*24*3600;             % conversion from cms to mm/day
%         qobs = strm - baseflow;
%         
%         % identify rainfall-runoff events with the rainfall time-series
%         % identify indices of zeros in rainfall data
%         b_ind = find(P>0, 1);
%         e_ind = find(P>0, 1, 'last');
%         ind_zero = find(P == 0);
%         ind_zero(ind_zero < b_ind) = [];
%         ind_zero(ind_zero > e_ind) = [];
%         
%         % remove indices of continous zeros
%         for a = 2:length(ind_zero)
%             if ind_zero(a)-ind_zero(a-1)==1
%                 ind_zero(a-1)=NaN;
%             end
%         end
%         ind_zero(isnan(ind_zero)) = [];
%         ind_zero = [0; ind_zero; length(P)]; 
%         event_indices = [ind_zero(1:end-1) + 1,ind_zero(2:end)];
%         
%         %% optimize the parameters
%         data.P = P;
%         data.D = D;
%         data.kmax = kmax;
%         data.qobs = qobs;
%         data.event_indices = event_indices;
%         theta_opt = SCSCN_uh_param_est(data);
%         
%         %% compute excess rainfall and corresponding direc runoff hydrograph at optimal parameter set
%         [qest, Pe_est] = SCSCN_uh(P, theta_opt, D, kmax, length(qobs), event_indices);
        
        period{per_ind}.qest = qest;
        period{per_ind}.Pe_est = Pe_est;
        period{per_ind}.theta_opt = theta_opt;
        %% plot data
%         yyaxis left
% %         plot(strm)
% %         hold on;
% %         plot(baseflow,'--')
%         plot(qest,'r')
%         hold on
%         plot(qobs,'-o')
%         hold off
%         
%         yyaxis right
%         bar(P,0.3,'facecolor','b')
%         hold on
%         bar(Pe_est,0.3,'facecolor','r')
%         hold off
%         set(gca,'ydir', 'reverse')
%         
%         pause;
        %}
    end
    dt = toc(t0);
end