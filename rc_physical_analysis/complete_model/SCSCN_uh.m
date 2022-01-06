%{
Computation of excess rainfall and streamflow response due to excess runoff

Author: Abhinav Gupta (Created: 30 Dec 2021)
%}

function [qest, Pe_est] = SCSCN_uh(P, params, D, kmax ,L, event_indices)
        
        % extract parameters
        d = size(event_indices,1);
        lambdas = params(1:d);
        CNs = params(d + 1 : 2*d);
        alphas = params(2*d+1 : 3*d);
        betas = params(3*d + 1 : 4*d);
        
        % computations
        S = 25400./CNs - 254;                                                       % potential maximum retention in mm
        Ia = lambdas.*S;                                                            % initial abstraction
        Pe_est = SCS_CN(P, Ia, S, event_indices);                                   % excess rainfall
        qest = direct_hydrograph(Pe_est, alphas, betas, D, kmax, L, event_indices); % hydrograph corresponding to excess rainfall
        
end