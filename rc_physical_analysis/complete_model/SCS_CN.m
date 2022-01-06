%{
Computation of excess rainfall using SCS-CN method

Author: Abhinav Gupta (Created: 30 Dec 2021)
%}

function Pe = SCS_CN(P, Ia, S, event_indices) 
    
    Pe = [];
    for ind = 1:size(event_indices,1)       % process each sub-event in a for loop

        begin_ind = event_indices(ind, 1);
        end_ind = event_indices(ind, 2);
        
        Ia_tmp = Ia(ind);
        S_tmp = S(ind);
        
        P_tmp = [0;P(begin_ind:end_ind)];
        n = length(P_tmp);

        cum_P = cumsum(P_tmp);
        ind_Ia = find(cum_P>Ia_tmp, 1);     % when cumulative rainfall becomes greater than initial abstraction
        if isempty(ind_Ia)                  % if cumulative rainfall is always less than initial abstraction
            cum_Pe_tmp = zeros(size(P_tmp));
        else
            cum_Ia = cum_P(1:ind_Ia-1,:);
            cum_Ia = [cum_Ia;Ia_tmp*ones(n-ind_Ia+1,1)];
            cum_Pe_tmp = (cum_P - cum_Ia).^2./(cum_P - cum_Ia + S_tmp);

            % Force Pe values to zero corresponding to those time-steps during which cumulative P is less than cumulative Ia
            ind_cons = cum_P < cum_Ia;
            cum_Pe_tmp(ind_cons) = 0;
        end
        
        Pe_tmp = cum_Pe_tmp(2:end) - cum_Pe_tmp(1:end-1);
        Pe = [Pe;Pe_tmp];
        
    end

end