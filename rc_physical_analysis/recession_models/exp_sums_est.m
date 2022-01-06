%{ 
Parameter estimtion: sum of n exponentials (maximum value of n is set to 3 ) to model recession
curves
%}
function [mdl, diagnostics] = exp_sums_est(mrc)
    
    Qt_Q0 = mrc/mrc(1);
    y = log(Qt_Q0);
    t = [0:length(y)-1]';
    
    %% n = 1
    obj = @(theta)sum((y - log(theta(1)) + theta(2)*t).^2);
    options = optimoptions('fmincon', 'MaxFunctionEvaluations', 10^5, 'MaxIterations', 10^4);
    theta_opt = fmincon(obj, [1,0.2], [], [], [], [], [0, 0], [inf,inf], [], options);
    b1 = theta_opt(1);
    a1 = theta_opt(2);
    Fitted = log(b1) - a1*t;
    mse = computeMSE(y, Fitted);
    nse = computeNSE(y, Fitted);
    aic = length(y)*log(mse) + 2*2;
    
    mdl1.Fitted = Fitted;
    mdl1.a1 = a1;
    mdl1.b1 = b1;
    mdl1.n = 1;
    
    diagnostics1.mse = mse;
    diagnostics1.nse = nse;
    diagnostics1.aic = aic;
    
    %% n = 2
    obj = @(theta)sum((y - ...
        log(theta(1)*exp(-theta(2)*t) + theta(3)*exp(-theta(4)*t))...
        ).^2);
    options = optimoptions('fmincon', 'MaxFunctionEvaluations', 10^5, 'MaxIterations', 10^4);
    theta_opt = fmincon(obj, [0.5,0.2, 0.5, 0.2], [], [], [], [], [0, 0, 0, 0], [inf, inf, inf, inf], [], options);
    b1 = theta_opt(1); b2 = theta_opt(3);
    a1 = theta_opt(2); a2 = theta_opt(4);
    Fitted = log(b1*exp(-a1*t) + b2*exp(-a2*t));
    mse = computeMSE(y, Fitted);
    nse = computeNSE(y, Fitted);
    aic = length(y)*log(mse) + 2*4;
    
    mdl2.Fitted = Fitted;
    mdl2.a1 = a1;
    mdl2.b1 = b1;
    mdl2.a2 = a2;
    mdl2.b2 = b2;
    mdl2.n = 2;
    
    diagnostics2.mse = mse;
    diagnostics2.nse = nse;
    diagnostics2.aic = aic;
    
    %% n = 3
    obj = @(theta)sum((y - ...
        log(theta(1)*exp(-theta(2)*t) + theta(3)*exp(-theta(4)*t) + theta(5)*exp(-theta(6)*t))...
        ).^2);
    options = optimoptions('fmincon', 'MaxFunctionEvaluations', 10^5, 'MaxIterations', 10^4);
    theta_opt = fmincon(obj, [0.34,0.2, 0.33, 0.2, 0.33, 0.2], [], [], [], [], [0, 0, 0, 0, 0, 0], [inf, inf, inf, inf, inf, inf], [], options);
    b1 = theta_opt(1); b2 = theta_opt(3); b3 = theta_opt(5);
    a1 = theta_opt(2); a2 = theta_opt(4); a3 = theta_opt(6);
    Fitted = log(b1*exp(-a1*t) + b2*exp(-a2*t) + b3*exp(-a3*t));
    mse = computeMSE(y, Fitted);
    nse = computeNSE(y, Fitted);
    aic = length(y)*log(mse) + 2*6;
    
    mdl3.Fitted = Fitted;
    mdl3.a1 = a1;
    mdl3.b1 = b1;
    mdl3.a2 = a2;
    mdl3.b2 = b2;
    mdl3.a3 = a3;
    mdl3.b3 = b3;
    mdl3.n = 3;
    
    diagnostics3.mse = mse;
    diagnostics3.nse = nse;
    diagnostics3.aic = aic;
    
    aic_list = [diagnostics1.aic, diagnostics2.aic, diagnostics3.aic];
    ind = find(aic_list == min(aic_list));
    ind = ind(1);
    if ind == 1
        mdl = mdl1;
        diagnostics = diagnostics1;
    elseif ind ==2
        mdl = mdl2;
        diagnostics = diagnostics2;
    else
        mdl = mdl3;
        diagnostics = diagnostics2;
    end
        
end
%}

function nse = computeNSE(obs, pred)
    nse = 1 - sum((obs - pred).^2)/sum((obs - mean(obs)).^2);
end

function mse = computeMSE(obs, pred)
    mse = mean((obs - pred).^2);
end