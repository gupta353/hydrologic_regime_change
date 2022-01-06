%{
Computation of baseflow using Eckhardt filter

Note BFI_max parameter is estimated using Collischonn and Fan (2013) method

Eckhardt (2005)

%}

clear all
close all
clc

direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis';

list = dir(direc);

a_list = [];
nse_list = [];
BFI_list = [];
for list_ind = 398%:length(list)
    
    basin = list(list_ind).name;
    
    %% find the linear recession parameter
    % read mrc data
    fname = 'mrc_completed.txt';
    filename = fullfile(direc, basin, fname);
    if ~isfile(filename)
        continue
    end
    fid = fopen(filename, 'r');
    data = textscan(fid, '%s%f', 'delimiter', '\t', 'headerlines', 1);
    fclose(fid);
    mrc = data{2};
    
    % find the parameter
    [a, nse] = linear_recession_parameter(mrc);
    a_list = [a_list;a];
    nse_list = [nse_list;nse];
    
    %% compute BFI using the streamflow time-series (Collischonn and Fan, 2013)
    %
    fname = 'raw_time_series.txt';
    filename = fullfile(direc, basin, fname);
    fid = fopen(filename, 'r');
    data = textscan(fid, '%s%f%f%f', 'delimiter', '\t', 'headerlines', 1);
    fclose(fid);
    
    strm = data{2};
    prcp = data{3};
    evap = data{4};
    [rec_period, rec_lengths]= extractRecPeriod(strm,prcp,evap);
    last_ind = rec_period{end}(end,1);
    strm = strm(1:last_ind);
    [BFI,bflow] = computeBFI(strm, a);
    BFI_list = [BFI_list;BFI];
%     plot(strm)
%     hold on
%     plot(bflow,'--')
%     hold off
%     pause;
    %}
    %% estimate baseflow for each rainfall-runoff event
    %
    fname = 'rainfall_runoff_data.mat';
    filename = fullfile(direc, basin, fname);
    load(filename);
    
    for per_ind = 1:length(period)
        strm = period{per_ind}.completed_streamflow;  
        baseflow = Eckhardt_filter(strm, a, BFI);
        period{per_ind}.baseflow = baseflow;
%         plot(strm);
%         hold on
%         plot(baseflow,'--');
%         hold off
%         pause;
    end
    %}
    
    % save rainfall runoff data
    save(filename, 'period')
end


function [BFI,baseflow] = computeBFI(strm, a)
    
    n = length(strm);
    baseflow = zeros(n,1);
    baseflow(end) = strm(end);
    for ind = n-1:-1:1
        baseflow_tmp = baseflow(ind+1)/a;
        baseflow(ind) = min(baseflow_tmp, strm(ind));
    end
    BFI = sum(baseflow)/sum(strm);
end

function baseflow = Eckhardt_filter(strm, a, BFI)
    
    n = length(strm);
    baseflow = zeros(n,1);
    baseflow(1) = strm(1);
    for ind = 2:n
        baseflow_tmp = ((1 - BFI)*a*baseflow(ind-1) + (1 - a)*BFI*strm(ind))/(1 - a*BFI);
        baseflow(ind) = min(strm(ind), baseflow_tmp);
    end

end