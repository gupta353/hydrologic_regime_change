%{
This function identifies the periods of high runoff-coefficients (>1),
combines the events of high runoff runoff-coefficients that are not adjacent 
to other high runoff-coefficient events, and reconstructs the cell-array of structures period. 
For example, if the kth event (in 'period') have high runoff coefficient
this events will be merged with (k-1)th and (k+1)th event, if (k-2)th and
(k+2)th events are not high runoff-coefficient events.

inputs: period = a cell-array of structures containing all the events
                  created
outputs: period = reconstructed event (including the ones not
reconstructed)
%}

function period = combinePeirodHighRC_1(period)

    runoff_coeff = [];
    for ind = 1:length(period)
        runoff_coeff = [runoff_coeff,period{ind}.runoff_coefficient];
    end

    rind = find(runoff_coeff>1);
    if length(rind)>1                       % check if there are more than 1 events with high RC
        cind = [];
        for ind = 2:length(rind)-1

            if (rind(ind)-rind(ind-1) > 2 && rind(ind+1)-rind(ind) > 2)
                cind = [cind,rind(ind)];
            end

        end

        if (rind(2)-rind(1) > 2 )
                cind = [rind(1),cind];
        end

        if (rind(end)-rind(end-1) > 2)
            cind = [cind,rind(end)];
        end

        %
        for ind_tmp  = cind
            if (ind_tmp>1) && ind_tmp<length(period)
                period{ind_tmp}.rain = [period{ind_tmp-1}.rain; period{ind_tmp}.rain; period{ind_tmp+1}.rain];
                period{ind_tmp}.streamflow = [period{ind_tmp-1}.streamflow; period{ind_tmp}.streamflow; period{ind_tmp+1}.streamflow];
                period{ind_tmp}.begin_date = period{ind_tmp-1}.begin_date;
                period{ind_tmp}.end_date = period{ind_tmp+1}.end_date;
                period{ind_tmp-1} = [];
                period{ind_tmp+1} = [];
            elseif ind_tmp==1
                period{ind_tmp}.rain = [period{ind_tmp}.rain; period{ind_tmp+1}.rain];
                period{ind_tmp}.streamflow = [period{ind_tmp}.streamflow; period{ind_tmp+1}.streamflow];
                period{ind_tmp}.end_date = period{ind_tmp+1}.end_date;
                period{ind_tmp+1} = [];
            elseif ind_tmp==length(period)
                period{ind_tmp}.rain = [period{ind_tmp-1}.rain; period{ind_tmp}.rain];
                period{ind_tmp}.streamflow = [period{ind_tmp-1}.streamflow; period{ind_tmp}.streamflow];
                period{ind_tmp}.begin_date = period{ind_tmp-1}.begin_date;
                period{ind_tmp-1} = [];
            end
        end
        %}
        period = period(~cellfun('isempty',period));
        % remove 'completed_hydrograph' corresponding to each event
        for ind = 1:length(period)
            period{ind} = rmfield(period{ind},'completed_streamflow');
        end
    else
        period = period;
    end

end