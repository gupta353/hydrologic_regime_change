% This routine computes uncertainty bounds over streamflow for each
% rainfall-runoff period
% Ref: Beven (2019). Towards hypothesis testing in inexact sciences

clear all
close all
clc

direc='D:/Research/Thesis_work/Structural_vs_measurement_uncertainty/matlab_codes';

mh_thresh = 0.80;      % (normalized) Mahalanobis distance threshold to select similar rainfall-runoff events
usgs_station = '04178000';
%% load data
fname=['rainfall_runoff_data_',usgs_station,'.mat'];
filename=fullfile(direc,'huc_04100003/MRC',fname);
load(filename);

%% record all runoff-coefficients
for per_ind=1:length(period)
    
    runoff_coeff(per_ind)=period{per_ind}.runoff_coefficient;
    
end
% runoff_coeff(runoff_coeff>3)=[];
runoff_coeff_max=max(runoff_coeff);
runoff_coeff_min=min(runoff_coeff);

%% compute uncertainty bounds by assigning equal membership to all runoff coefficients
%{
for per_ind=1:length(period)
    
    period{per_ind}.ub_streamflow=period{per_ind}.completed_streamflow*runoff_coeff_max/period{per_ind}.runoff_coefficient;
    period{per_ind}.lb_streamflow=period{per_ind}.completed_streamflow*runoff_coeff_min/period{per_ind}.runoff_coefficient;
    
end

% plot data
for per_ind=1:length(period)
    plot(period{per_ind}.lb_streamflow);
    hold on;
    plot(period{per_ind}.ub_streamflow);
    plot(period{per_ind}.completed_streamflow,'r');
    pause;
    hold off;
end
%}
%% compute membership of each runoff coefficient based on rainfall volume and antecendent moisture condition
%
for per_ind = 1:length(period)
    
    tot_rain(per_ind,1) = sum(period{per_ind}.rain);
    int_strm(per_ind,1) = sum(period{per_ind}.streamflow(1));
    
end
tot_rain = (tot_rain-mean(tot_rain))/std(tot_rain);
int_strm = (int_strm-mean(int_strm))/std(int_strm);

% computation of Mahalanobis distance
dist_data = [tot_rain,int_strm];
D = zeros(size(dist_data,1),size(dist_data,1));
for cind = 1:size(dist_data,2)
    
    var_tmp = dist_data(:,cind);
    var_tmp = repmat(var_tmp,1,length(var_tmp));
    D = D + (var_tmp - var_tmp').^2;
    
end
D = D.^0.5;

% normalization of Mahalanobis distance
max_D = max(D,[],2);
Dnorm = D./repmat(max_D,1,size(D,2));

% compute the maximum and minimum runoff coefficients according to a
% Mahalanobis distance threshold
day_sum=[];
for per_ind=1:length(period)
    
    
    ind = find(Dnorm(per_ind,:)<=mh_thresh);
    runoff_coeff_max=max(runoff_coeff(ind));
    runoff_coeff_min=min(runoff_coeff(ind));
    period{per_ind}.ub_streamflow=period{per_ind}.completed_streamflow*runoff_coeff_max/period{per_ind}.runoff_coefficient;
    period{per_ind}.lb_streamflow=period{per_ind}.completed_streamflow*runoff_coeff_min/period{per_ind}.runoff_coefficient;
    day_sum(per_ind) = length(period{per_ind}.streamflow);
    
end
day_sum = cumsum(day_sum);

% plot data
lb_streamflow = [];
ub_streamflow = [];
streamflow = [];
for per_ind=1:length(period)
    lb_streamflow = [lb_streamflow;period{per_ind}.lb_streamflow(1:length(period{per_ind}.streamflow))];
    ub_streamflow = [ub_streamflow;period{per_ind}.ub_streamflow(1:length(period{per_ind}.streamflow))];
    streamflow = [streamflow;period{per_ind}.streamflow];
end

streamflow_reconstruct = join_strms_series(period);
period1 = period;
for per_ind = 1:length(period1)
    period1{per_ind}.completed_streamflow = period1{per_ind}.lb_streamflow;
end
lb_streamflow_reconstruct = join_strms_series(period1);
period1 = period;
for per_ind = 1:length(period1)
    period1{per_ind}.completed_streamflow = period1{per_ind}.ub_streamflow;
end
ub_streamflow_reconstruct = join_strms_series(period1);

% save data
sname = ['runoff_coeff_uncer_dist_thresh_',num2str(mh_thresh),'.mat'];
filename = fullfile(direc,['huc_04100003/results/runoff_coeff_uncertainty_bounds_',usgs_station],sname);
save(filename,'ub_streamflow_reconstruct','lb_streamflow_reconstruct','streamflow');

plot(lb_streamflow_reconstruct);
hold on;
plot(ub_streamflow_reconstruct);
plot(streamflow,'r')
xlim([1465 1525])
%}