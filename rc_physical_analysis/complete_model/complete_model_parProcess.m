%{
this script is written to facilitate parallel processing
%}

function [qest, Pe_est, theta_opt] = complete_model_parProcess(period_tmp, D, kmax, darea)
        
        % read data
        P = period_tmp.rain;
        strm = period_tmp.completed_streamflow/darea/1000*24*3600;     % conversion from cms to mm/day
        baseflow = period_tmp.baseflow/darea/1000*24*3600;             % conversion from cms to mm/day
        qobs = strm - baseflow;       
        
        % check if all preciptation values are zero
        ind = find(P == 0);
        if length(ind) == length(P)
            qest = zeros(length(strm),1);
            Pe_est = zeros(length(strm),1);
            theta_opt = NaN;
        else
            % identify rainfall-runoff events with the rainfall time-series
            % identify indices of zeros in rainfall data
            b_ind = find(P>0, 1);
            e_ind = find(P>0, 1, 'last');
            ind_zero = find(P == 0);
            ind_zero(ind_zero < b_ind) = [];
            ind_zero(ind_zero > e_ind) = [];

            % remove indices of continous zeros
            for a = 2:length(ind_zero)
                if ind_zero(a)-ind_zero(a-1)==1
                    ind_zero(a-1)=NaN;
                end
            end
            ind_zero(isnan(ind_zero)) = [];
            ind_zero = [0; ind_zero; length(P)]; 
            event_indices = [ind_zero(1:end-1) + 1,ind_zero(2:end)];

            %% optimize the parameters
            data.P = P;
            data.D = D;
            data.kmax = kmax;
            data.qobs = qobs;
            data.event_indices = event_indices;
            theta_opt = SCSCN_uh_param_est(data);

            %% compute excess rainfall and corresponding direc runoff hydrograph at optimal parameter set
            [qest, Pe_est] = SCSCN_uh(P, theta_opt, D, kmax, length(qobs), event_indices);

            qest = qest*darea*1000/24/3600;
        end
        
end