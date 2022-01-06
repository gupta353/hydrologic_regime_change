%{ 
Parameter estimtion: Boussinesq 1904 recession relationship
%}
function [mdl, diagnostics] = Boussinesq_1904_est(mrc)

    Qt_Q0 = mrc/mrc(1);
    y = log(Qt_Q0);
    t = [0:length(y)-1]';
    
    % local optimum by fmincon
    obj = @(a3)sum(y + 2*log(1 + a3*t)).^2;
    options = optimoptions('fmincon', 'MaxFunctionEvaluations', 10^5, 'MaxIterations', 10^4);
    a3_opt = fmincon(obj, 0.02, [], [], [], [], 0, inf, [], options);
    t_tmp = -2*log(1 + a3_opt*t);
    
    % brute force optimization
    %{
    a3_list = 0.01:0.01:10;

    for count = 1:length(a3_list)

        a3 = a3_list(count);
        t_tmp = -2*log(1 + a3*t);
        mse_list(count) = computeMSE(y, t_tmp);

    end
    ind = find(mse_list == min(mse_list));
    a3_opt = a3_list(ind);
    t_tmp = -2*log(1 + a3_opt*t);
    %}
    
    mse = computeMSE(y, t_tmp);
    nse = computeNSE(y, t_tmp);
    aic = length(t_tmp)*log(mse) + 2;  % number of parameters = 1

    diagnostics.aic = aic;
    diagnostics.mse = mse;
    diagnostics.nse = nse;

    mdl.a3 = a3_opt;
    mdl.obs = y;
    mdl.Fitted = t_tmp;
    
end

function nse = computeNSE(obs, pred)
    nse = 1 - sum((obs - pred).^2)/sum((obs - mean(obs)).^2);
end

function mse = computeMSE(obs, pred)
    mse = mean((obs - pred).^2);
end