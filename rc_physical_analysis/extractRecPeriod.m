%{
This function extracts recession periods from a streamflow time-series
inputs: strm_vals = streamflow time series
        prcp_vals = rainfall time series
        evap_vals = evaporation tgime series
outputs: rec_period = a cell-array such that each cell contains for columns
                      1. time steps of the recession period in the given
                         streamflow time series
                      2. rainfall during the time steps
                      3. evaporation during the time steps
                      4. streamflow during the time steps
        rec_lengths = length of each recession period as number of time
                      steps
%}

function [rec_period, rec_lengths]= extractRecPeriod(strm_vals,prcp_vals,evap_vals)

    strm_shifted=strm_vals(1:end-1);
    differ=strm_vals(2:end)-strm_shifted;
    ind=find(differ<0);
    rec_times=ind+1;

    %
    rec_time_shifted=rec_times(1:end-1);
    differ=rec_times(2:end)-rec_time_shifted;
    ind=find(differ>1);
    rec_lengths=[ind(1);ind(2:end)-ind(1:end-1)];       % lengths of observed recession periods
    break_times=rec_times(ind+1);                       % starting indices of recession periods

    %figure; hold on;
    time_count=1;
    for rec_ind=1:length(rec_lengths)

        rec_length_tmp=rec_lengths(rec_ind);
        plot_times=rec_times(time_count:time_count+rec_length_tmp-1);
        strm_vals_tmp=strm_vals(plot_times);

        rec_period{rec_ind}(:,1)=plot_times;
        rec_period{rec_ind}(:,2)=prcp_vals(plot_times);
        rec_period{rec_ind}(:,3)=evap_vals(plot_times);
        rec_period{rec_ind}(:,4)=strm_vals(plot_times);

        plot(plot_times,strm_vals_tmp);

    %     pause(1);
        time_count=time_count+rec_length_tmp;

    end
%     hold off;
%     xlabel('Time-steps (days)','fontname','arial','fontsize',12);
%     ylabel('Streamflow (mm day^{-1})','fontname','arial','fontsize',12);
%     title('Recession periods','fontname','arial','fontsize',12);
%     set(gca,'fontname','arial','fontsize',12);

end