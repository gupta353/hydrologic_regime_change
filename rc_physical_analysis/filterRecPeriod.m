%{
This function filters out the recession periods that do not satisfy some
constraints
%}

function [filt_rec_periods,prcp_vol, strm_vol, evap_vol]= filterRecPeriod(rec_period, rec_lengths, prcp_strm_ratio_thresh, evap_strm_ratio_thresh_pos,evap_strm_ratio_thresh_neg, rec_length_thresh)

    for rec_ind=1:length(rec_period)

        prcp_vol(rec_ind)=sum(rec_period{rec_ind}(:,2));
        evap_vol(rec_ind)=sum(rec_period{rec_ind}(:,3));
        strm_vol(rec_ind)=sum(rec_period{rec_ind}(:,4));

    end

    prcp_strm_ratios=prcp_vol./strm_vol;
    evap_strm_ratios=evap_vol./strm_vol;

    % plot histograms of ratios
    %{
    figure; hist(prcp_strm_ratios);
    xlabel('Precipitation streamflow ratio','fontname','arial','fontsize',12);
    ylabel('Number of samples in the bin','fontname','arial','fontsize',12);
    set(gca,'fontname','arial','fontsize',12);

    figure; hist(evap_strm_ratios);
    xlabel('Evaporation streamflow ratio','fontname','arial','fontsize',12);
    ylabel('Number of samples in the bin','fontname','arial','fontsize',12);
    set(gca,'fontname','arial','fontsize',12);
    %}
    %% filter the recession periods
    ind_prcp=find(prcp_strm_ratios<=prcp_strm_ratio_thresh);        % precipitation threshold
    ind_evap=find(evap_strm_ratios<=evap_strm_ratio_thresh_pos...
        & evap_strm_ratios>=evap_strm_ratio_thresh_neg);        % evaporation threshold
    ind_length=find(rec_lengths>=rec_length_thresh);                % length threshold
    ind_final1=intersect(ind_prcp,ind_evap);
    ind_final=intersect(ind_final1,ind_length);

    filt_rec_periods=rec_period(ind_final);                         % filtered recession periods

end