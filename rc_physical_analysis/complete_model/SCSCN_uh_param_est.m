%{
SCS-CN-UH parameter optimization

Author: Abhinav Gupta (Created: 29 Dec 2021)
%}

function theta_opt = SCSCN_uh_param_est(data)

    event_indices = data.event_indices;
    
    d = size(event_indices,1);       % number of sub-events
    lb = [0.001*ones(1,d), 0.001*ones(1,d), 0.01*ones(1,d), 0.01*ones(1,d)]; % lower bound
    ub = [0.30*ones(1,d), 100*ones(1,d), 100*ones(1,d), 100*ones(1,d)];      % upper bound
    %% using simulated annealing
    %{
    fun = @(x)objfun(x,P,D,kmax,qobs, event_indices);
    options.InitTemp = 1;
    options.MaxTries = 2000;
    options.MaxSuccess = 300;
    theta_opt = anneal(fun, [0.20*ones(1,d_CN),80*ones(1,d_CN),1*ones(1,d_CN),1*ones(1,d_CN)], options);%, [0.01, 0.01, 0.01, 0.01], [0.30, 100, inf, inf]);
    %}
    
    %% using DDS algorithm
    objfun_name = @(x)objfun(x,data);
    objfunc_bounds_name = 'objfun';
    S_name = [1:length(lb)]'; 
    S_min = lb';
    S_max = ub';
    Discrete_flag = zeros(length(lb),1);
    theta_opt = MainDDS(objfun_name, objfunc_bounds_name, S_name, S_min, S_max, Discrete_flag);
    
end