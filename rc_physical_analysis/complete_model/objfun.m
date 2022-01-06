%{
Objective function to be minimized

Author: Abhinav Gupta (Created: 30 Dec 2021)
%}
function sss = objfun(x, data)
    
    P = data.P;
    D = data.D;
    kmax = data.kmax;
    qobs = data.qobs;
    event_indices = data.event_indices;
    
    %% using simulated annealing algorithm
    %{
    % number of lambda and CN parameters
    d_CN = size(event_indices,1);
    
    % check parameter bounds
    lb = [0.001*ones(1,d_CN), 0.001*ones(1,d_CN), 0.01*ones(1,d_CN), 0.01*ones(1,d_CN)];
    ub = [0.30*ones(1,d_CN), 100*ones(1,d_CN), inf*ones(1,d_CN), inf*ones(1,d_CN)];
    ind_lb = x>=lb;
    ind_ub = x<=ub;
    
    alphas = x(2*d_CN+1 : 3*d_CN);
    betas = x(3*d_CN+1 : 4*d_CN);
    mu_t = alphas./betas;
    
    mu_ind = mu_t < 50;
    
    if sum(ind_lb) + sum(ind_ub) == 2*4*d_CN && sum(mu_ind) == d_CN
        [qest, Pe_est] = SCSCN_uh(P, x, D, kmax, length(qobs), event_indices);
        L = min(length(qest),length(qobs));
        sss = sum((qest(1:L) - qobs(1:L)).^2);
    else
        sss = inf;
    end
    %}
    
    %% using DDS algorithm
    [qest, Pe_est] = SCSCN_uh(P, x, D, kmax, length(qobs), event_indices);
    L = min(length(qest),length(qobs));
    sss = sum((qest(1:L) - qobs(1:L)).^2);
end