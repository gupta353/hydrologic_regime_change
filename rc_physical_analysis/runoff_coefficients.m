% This routine computes runoff coeffcients of a drainage area for all the
% rainfall-runoff events in the user-supplied rainfall-runoff data
% Note: (1) Missing values in rainfall data should be designated as NaNs in
%           input rainfall data
%       (2) Appearance of negative streamflow at the begining of streamflow
%           hydrograph indicates that mrc has over-estimated the streamflows in
%           the previous hydrograph, to correct for over-estimation of
%           streamflow appended streamflows in previous hydrographs is adjusted
%           such that negative streamflows disappear
% Def: Gap period: a time-period with negligible rainfall
clear all
close all
clc

direc='D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data/complete_watersheds_0';
darea_direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata';
save_direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis';

length_thresh = 5;               % minimum separation between the end and begining of two storms (In other words, minimum length of gap period)

% drainage area
filename = fullfile(darea_direc,'gauge_information.txt');
fid = fopen(filename,'r');
data = textscan(fid,'%s%s%s%f%f%f','delimiter','\t','headerlines',1);
fclose(fid);
stations = data{2};
dareas = data{6};

list = dir(save_direc);

for station_ind = 101:length(list)
    
    station_id = list(station_ind).name;
    
    if isfolder(fullfile(save_direc,station_id))
        
        ind = find(strcmp(stations,station_id));
        darea = dareas(ind);    % in km^{2}
        
        %% depth of rainfall which is assumed to be negligible for the separation of hydrographs (in mm/day)
        prcp_thresh_fname = 'prcp_threshold.txt';
        filename = fullfile(save_direc,station_id,prcp_thresh_fname);
        if isfile(filename)
            fid = fopen(filename,'r');
            prcp_thresh = textscan(fid,'%f');
            fclose(fid);
            prcp_thresh = prcp_thresh{1};
        else
            continue
        end
        %% read MRC
        mrc_fname='MRC_completed.txt';
        filename=fullfile(save_direc,station_id,mrc_fname);
        if isfile(filename)   % if mrc_post_processed data is available
            fid=fopen(filename,'r');
            mrc=textscan(fid,'%f%f','delimiter','\t','headerlines',1);
            fclose(fid);
            mrc=mrc{2};           % in cms
        else
            continue
        end
        %% load streamflow data, rainfall data, and evaporation data
        % read data
        fname = 'raw_time_series.txt';
        filename = fullfile(save_direc,station_id,fname);
        fid = fopen(filename,'r');
        data = textscan(fid,'%s%f%f%f','delimiter','\t','headerlines',1);
        fclose(fid);
        
        dates = data{1};
        prcp_vals = data{3};
        strm_vals = data{2};
        evap_vals = data{4};
        
        %{
        yyaxis left
        plot(strm_vals);

        yyaxis right
        bar(prcp_vals);
        set(gca, 'YDir','reverse')
        %pause;
        %}
        %% Identify periods with negligible rainfall (in other words, gap-periods) %
        ind_prcp_thresh=find(prcp_vals<prcp_thresh);
        differ=ind_prcp_thresh(2:end)-ind_prcp_thresh(1:end-1);
        ind_differ=find(differ>1);                                                      % indices of gap time-steps
        eind_gap_periods=[ind_prcp_thresh(ind_differ);ind_prcp_thresh(end)];            % end index of each gap-period
        bind_gap_periods=[ind_prcp_thresh(1);ind_prcp_thresh(ind_differ(1:end)+1)];     % begin index of each gap-period
        gap_period=[bind_gap_periods,eind_gap_periods];
        length_gap_periods=(eind_gap_periods-bind_gap_periods)+1;                       % length of gap-periods
        
        % remove gap periods that are smaller in length than length_thresh
        ind_gp=find(length_gap_periods<length_thresh);
        gap_period(ind_gp,:)=[];
        length_gap_periods(ind_gp)=[];
        %% Identify rainfall periods using gap periods
        pseudo_gap_period=[0,0;gap_period];
        rain_period(:,1)=pseudo_gap_period(:,2)+1;                                      % begin index of rain period
        rain_period(:,2)=[pseudo_gap_period(2:end,1)-1;length(strm_vals)];              % end index of rain period
        if rain_period(end,1)>rain_period(end,2)
            rain_period(end,:)=[];
        end
        
        %% Identify streamflow period corresponding to each rainfall period
        %
        strm_tot=[];
        for strm_ind=1:size(rain_period,1)-1
            
            bind=rain_period(strm_ind,1);                     % begin index of curerent rainfall period
            eind=rain_period(strm_ind+1,1)-1;                 % index just before the start of next rainfall period
            period{strm_ind}.rain=prcp_vals(bind:eind);
            period{strm_ind}.streamflow=strm_vals(bind:eind);
            period{strm_ind}.begin_date = dates{bind};
            period{strm_ind}.end_date = dates{eind};
            
            %     strm_tot=[strm_tot;period{strm_ind}.streamflow];
            %     plot(strm_tot);
            %     pause;
        end
        strm_ind=strm_ind+1;
        period{strm_ind}.rain=prcp_vals(rain_period(strm_ind,1):end);
        period{strm_ind}.streamflow=strm_vals(rain_period(strm_ind,1):end);
        period{strm_ind}.begin_date = dates{rain_period(strm_ind,1)};
        period{strm_ind}.end_date = dates{end};
        %}
        
        %% Complete the recession curve by appending the data from MRC
        %
        period=complete_hydrograph(period,mrc);
        %}
        %{
        figure;
        for per_ind=1:length(period)

            yyaxis left
            plot(period{per_ind}.completed_streamflow);

            yyaxis right
            bar(period{per_ind}.rain);
            set(gca, 'YDir','reverse')
            %pause;
        end
        %}
        
        %% Computation of runoff coefficients
        [runoff_coeff,strm_tmp_vol,prcp_tmp_vol]=runoff_coff_comp(period,darea);
        saveRCPlot(fullfile(save_direc,station_id), 'RC_1', prcp_tmp_vol, runoff_coeff);
        
        for run_ind=1:length(runoff_coeff)
            period{run_ind}.runoff_coefficient=runoff_coeff(run_ind);
        end
        
        % identify indice of runoff-coefficients greater than 1 that are adjacent to each other and reconstruct the rainfall-runoff events
        rind = find(runoff_coeff>1);
        ind_diff = rind(2:end) - rind(1:end-1);
        ind_diff(ind_diff ~= 1) = 0;
        while sum(ind_diff) ~= 0
            period = combinePeriodHighRC(period);
            period=complete_hydrograph(period,mrc);
            [runoff_coeff,strm_tmp_vol,prcp_tmp_vol]=runoff_coff_comp(period,darea);
            
            for run_ind=1:length(runoff_coeff)
                period{run_ind}.runoff_coefficient=runoff_coeff(run_ind);
            end
            
            rind = find(runoff_coeff>1);
            ind_diff = rind(2:end) - rind(1:end-1);
            ind_diff(ind_diff ~= 1) = 0;
        end
        
        [runoff_coeff,strm_tmp_vol,prcp_tmp_vol]=runoff_coff_comp(period,darea);
        saveRCPlot(fullfile(save_direc,station_id), 'RC_2', prcp_tmp_vol, runoff_coeff);
        
        % identify indice of runoff-coefficients greater than 1 that are separated by more than 2 events and reconstruct the rainfall-runoff events
        period = combinePeirodHighRC_1(period);
        period=complete_hydrograph(period,mrc);
        [runoff_coeff,strm_tmp_vol,prcp_tmp_vol]=runoff_coff_comp(period,darea);
        saveRCPlot(fullfile(save_direc,station_id), 'RC_3', prcp_tmp_vol, runoff_coeff);
        
        % identify indice of runoff-coefficients greater than 1 that are separated by less than 2 events and reconstruct the rainfall-runoff events
        period = combinePeriodHighRC_2(period);
        period=complete_hydrograph(period,mrc);
        [runoff_coeff,strm_tmp_vol,prcp_tmp_vol]=runoff_coff_comp(period,darea);
        saveRCPlot(fullfile(save_direc,station_id), 'RC_4', prcp_tmp_vol, runoff_coeff);
        
        %{
        figure;
        for per_ind=1:length(period)

            yyaxis left
            plot(period{per_ind}.completed_streamflow);

            yyaxis right
            bar(period{per_ind}.rain, 'Barwidth',0.1);
            set(gca, 'YDir','reverse')
            title({['Runoff-coefficient = ',num2str(runoff_coeff(per_ind))], ['Time-period = ', period{per_ind}.begin_date,' to ',period{per_ind}.end_date]})
            pause;
        end
        %}
        %
        % save data
        sname = 'rainfall_runoff_data.mat';
        save_filename=fullfile(save_direc,station_id,sname);
        save(save_filename,'period')
        
        close all
        fclose('all');
        clear rain_period gap_period eind_gap_periods bind_gap_periods length_gap_periods pseudo_gap_period rain_period period
        %}
    end
end