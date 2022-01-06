% This routine estimates the master recession curve
% 
% Ref: Lamb and Beven (1997), Tallaksen (1995)

clear all
close all
clc

direc='D:/Research/non_staitionarity/data/CAMELS_GLEAMS_combined_data/complete_watersheds_12';
darea_direc = 'D:/Research/non_staitionarity/data/CAMELS_raw/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata';
save_direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis';

% define parameters
rec_length_thresh=4;                % minimum lenght of recession period to be considered (in days)
prcp_strm_ratio_thresh=0.1;         % upper limit for precipitation to streamflow ratio (recommended <=0.1)
evap_strm_ratio_thresh_pos=0.1;     % upper limit for positive evaporation to streamflow ratio (recommended <=0.1)
evap_strm_ratio_thresh_neg=-0.1;    % lower limit for negative evaporation to streamflow ratio (recommended <=-0.1)
prcp_thresh_max = 0.70;

begin_year = 1985;
end_year = 2013;

list = dir(direc);

for file_ind = 3:length(list)
    
    fname = list(file_ind).name;
    
    prcp_strm_ratio_thresh_tmp = prcp_strm_ratio_thresh;         
    evap_strm_ratio_thresh_pos_tmp = evap_strm_ratio_thresh_pos;
    evap_strm_ratio_thresh_neg_tmp = evap_strm_ratio_thresh_neg;

    station_id = strsplit(fname,'_');
    station_id = station_id{1};
    %station_id = '02315500';
    %fname = [station_id, '_GLEAMS_CAMELS_data.txt'];
    
    % directory where data will be saved
    save_dir_tmp = fullfile(save_direc,station_id);
    mkdir(save_dir_tmp)
    
    % drainage area
    filename = fullfile(darea_direc,'gauge_information.txt');
    fid = fopen(filename,'r');
    data = textscan(fid,'%s%s%s%f%f%f','delimiter','\t','headerlines',1);
    fclose(fid);
    stations = data{2};
    dareas = data{6};

    ind = find(strcmp(stations,station_id));
    darea = dareas(ind);    % in km^{2}

    % read data
    filename = fullfile(direc,fname);
    fid = fopen(filename,'r');
    data = textscan(fid,'%d%d%d%f%f%f%f%f%f%f%s%f','delimiter','\t','headerlines',1);
    fclose(fid);
    
    % daymet data for rainfall and swe
    years = data{1};
    months = data{2};
    days = data{3};
    prcp = data{4};                 % in mm day^{-1}
    swe = data{7};                  % in mm day^{-1}
    strm = data{10}*0.028316847;    % cfs to cms
    evap = data{12};                % in mm day^{-1}

    % extract data in a particulat time-window as defined by begin_year and end_year
    begin_ind = find(years==begin_year & months==10 & days==1);
    end_ind = find(years==end_year & months==9 & days==30);

    prcp_vals = prcp(begin_ind:end_ind);
    swe_vals = swe(begin_ind:end_ind);
    strm_vals = strm(begin_ind:end_ind);
    strm_vals = strm_vals/darea/1000*3600*24; % cms to mm day^{-1}
    plot(strm_vals)
    evap_vals = evap(begin_ind:end_ind);

    if isnan(prcp_vals)
        writeError(save_dir_tmp, 'NaNError.txt', 'Rainfall data contains NaNs');
        continue
    end

    if isnan(evap_vals)
        writeError(save_dir_tmp, 'NaNError.txt', 'Evaporation data contains NaNs');
        continue
    end

    %% extract recession periods
    [rec_period, rec_lengths] = extractRecPeriod(strm_vals,prcp_vals,evap_vals);

    %% Compute statistics to filter the recession periods
    % compute total volumes of streamflow, rainfall, and evaporation during each recession period
    [filt_rec_periods, prcp_vol, strm_vol, evap_vol] = filterRecPeriod(rec_period, rec_lengths, prcp_strm_ratio_thresh_tmp, evap_strm_ratio_thresh_pos_tmp,evap_strm_ratio_thresh_neg_tmp, rec_length_thresh);
    
    figure; plot(strm_vol); hold on; plot(prcp_vol); plot(evap_vol); legend({'strm', 'prcp', 'evap'});
    
    while length(filt_rec_periods)<5 && prcp_strm_ratio_thresh_tmp<prcp_thresh_max
        
        prcp_strm_ratio_thresh_tmp = prcp_strm_ratio_thresh_tmp + 0.05;
        evap_strm_ratio_thresh_pos_tmp = evap_strm_ratio_thresh_pos_tmp + 0.05;
        evap_strm_ratio_thresh_neg_tmp = evap_strm_ratio_thresh_neg_tmp - 0.05;
        
        filt_rec_periods = filterRecPeriod(rec_period, rec_lengths, prcp_strm_ratio_thresh_tmp, evap_strm_ratio_thresh_pos_tmp,evap_strm_ratio_thresh_neg_tmp, rec_length_thresh);
        
    end
    
    if length(filt_rec_periods)<3
        writeError(save_dir_tmp, 'filt_recession_period_error.txt', 'Not able to find suitable recession periods that satisfied the imposed constraints')
        continue
    end

    %% create mrc based on filtered recession periods
    mrcf = createMRC(filt_rec_periods);

    %% plot mrc
    close all
    figure; plot(mrcf*darea*1000/24/3600, linewidth = 2)
    xlabel('Arbitrary time steps (days)','fontname','arial','fontsize',10);
    ylabel('Streamflow (m^{3} s^{-1})','fontname','arial','fontsize',10);
    title('Master recession curve','fontname','arial','fontsize',10);
    set(gca,'fontname','arial','fontsize',10);

    % save plot
    sname = 'mrc.png';
    filename=fullfile(save_dir_tmp,sname);
    saveas(gcf,filename,'png');

    sname = 'mrc.svg';
    filename=fullfile(save_dir_tmp,sname);
    saveas(gcf,filename,'svg');

    %% save mrc
    sname=['MRC_',station_id,'.txt'];
    filename=fullfile(save_dir_tmp,sname);
    fid=fopen(filename,'w');
    fprintf(fid,'%s\t%s\n','time-step(days)','Streamflow(cms)');
    for write_ind=1:length(mrcf)

        fprintf(fid,'%f\t%f\n',write_ind-1,mrcf(write_ind)*darea*1000/24/3600); % coversion from mm day^{-1} to m^3 s^{-1}

    end
    fclose(fid);

    %% compute rainfall threshold for hydrograph separation
    for ind = 1:length(rec_period)
        prcp_max(ind) = max(rec_period{ind}(:,2));
    end
    prcp_max(prcp_max==0) = [];
    prcp_thresh = prctile(prcp_max,75);

    % save precipitation threshold value to a text file
    fname = 'prcp_threshold.txt';
    filename = fullfile(save_dir_tmp,fname);
    fid = fopen(filename,'w');
    fprintf(fid,'%f',prcp_thresh);
    fclose(fid);
    
    %% write the computed thresholds value to  textfile
    fname = 'mrc_parameters.txt';
    filename = fullfile(save_dir_tmp,fname);
    fid = fopen(filename,'w');
    string = ['prcp_strm_ratio_thresh = ', num2str(prcp_strm_ratio_thresh_tmp),'\nevap_strm_ratio_thresh_pos = ', num2str(evap_strm_ratio_thresh_pos_tmp), '\nevap_strm_ratio_thresh_neg = ' , num2str(evap_strm_ratio_thresh_neg_tmp),...
        '\nrec_length_thresh = ', num2str(rec_length_thresh)];
    fprintf(fid,string);
    fclose(fid);
    
    %% write prcp, strm and evaporation data
    writeRawData(strm_vals*darea*1000/3600/24, prcp_vals, evap_vals, begin_year, end_year, save_dir_tmp,'raw_time_series.txt');
     
    clear prcp swe strm evap prcp_vals swe_vals strm_vals evap_vals rec_period filt_rec_periods  mrc mrcf prcp_max prcp_thresh ind
end