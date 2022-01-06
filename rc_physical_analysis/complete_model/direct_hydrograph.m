%{
Computation of streamflow response due to excess rainfall

Author: Abhinav Gupta (Created: 30 Dec 2021)
%}

function q = direct_hydrograph(Pe, alphas, betas, D, kmax, L, event_indices)
    
    q = zeros(10000,1);     % initialize direct runoff
    for ind = 1:size(event_indices,1)
        
        alpha = alphas(ind);
        beta = betas(ind);
        
        begin_ind = event_indices(ind,1);
        end_ind = event_indices(ind,2);
        Pe_tmp = Pe(begin_ind:end_ind);
        
%         k = floor(alpha/beta + 10*alpha/beta^2);
%         disp(k);
        % compute D-duration unit hydrograph
        u_D = uh_D(alpha, beta, D, kmax);

        % compute direc runoff hydrograph
        q_tmp = conv(u_D, Pe_tmp);
        
        % super-impose direc-runoff hydrograph from this sub-event with that of previous sub-events
        q(begin_ind : begin_ind+length(q_tmp)-1) = q(begin_ind : begin_ind+length(q_tmp)-1) + q_tmp;
    end

    L = min(L, length(q));
    q = q(1:L);
end