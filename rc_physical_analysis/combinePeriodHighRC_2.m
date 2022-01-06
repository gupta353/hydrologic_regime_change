%{
This function identifies the periods of high runoff-coefficients (>1),
combines the continous in-time events of high runoff runoff-coefficients,
and reconstructs the cell-array of structures period. For example, if the
kth, (k+2)th and (k+4)th events (in 'period') have high runoff coefficient,
kth to (k+4)th events will be merged to form one kth event. Also, the
(k+1)th to (k+4)th events will be removed.

inputs: period = a cell-array of structures containing all the events
                  created
outputs: period = reconstructed event (including the ones not
reconstructed)
%}

function period = combinePeriodHighRC_2(period)
    
    runoff_coeff = [];
    for ind = 1:length(period)
        runoff_coeff = [runoff_coeff,period{ind}.runoff_coefficient];
    end
    
    rind = find(runoff_coeff>1);
    ind_diff = rind(2:end) - rind(1:end-1);
    ind_diff(ind_diff ~= 2) = 0;
    two_ind = find(ind_diff==2);
    rind1 = rind(two_ind);

    groups = {};
    count = 0;
    i = 1;
    while i <= length(rind1)
        count = count + 1;
        j = i;
        while j+1 <= length(rind1) && (rind1(j+1) - rind1(j) == 2)
            j = j + 1;
        end
        groups{count} = [rind1(i:j),rind1(j)+2];
        i = j+1;
    end

    for gind = 1:length(groups)
              
        pinds = groups{gind}(1)-1:groups{gind}(end)+1;
        pinds(pinds==0) = [];
        pinds(pinds>length(period)) = [];
        while_ind = 1;
        while isempty(period{pinds(while_ind)})
            pinds(while_ind) = [];
        end
        
        for ind = 2:length(pinds)
            period{pinds(1)}.rain = [period{pinds(1)}.rain; period{pinds(ind)}.rain];
            period{pinds(1)}.streamflow = [period{pinds(1)}.streamflow; period{pinds(ind)}.streamflow];
            period{pinds(1)}.end_date = period{pinds(ind)}.end_date;
            period{pinds(ind)} = [];
        end
        
    end
    period = period(~cellfun('isempty',period));
    
    % remove 'completed_hydrograph' corresponding to each event
    for ind = 1:length(period)
        period{ind} = rmfield(period{ind},'completed_streamflow');
    end

end