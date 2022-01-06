%{
Parameter estimtion: power law recession relationship (Brutsaert and Nieber, 1976)
%}
function [mdl, diagnostics] = power_law_est(mrc)

    Qt_Q0 = mrc/mrc(1);
    y = log(Qt_Q0);
    t = [0:length(y)-1]';
    
    %% b = 1
    mdl1 = fitlm(t,y, 'Intercept', false);
    mse = computeMSE(y, mdl1.Fitted);
    nse = computeNSE(y, mdl1.Fitted);
    aic_1 = length(y)*log(mse) + 2;
    
    diagnostics1.mse = mse;
    diagnostics1.nse = nse;
    diagnostics1.aic = aic_1;
    
    %% b ~= 1
    Q0 = mrc(1);
    obj = @(theta)sum((mrc.^(1-theta(1)) - (...
        Q0^(1-theta(1)) - (1-theta(1))*theta(2)*t...
        )).^2);
    options = optimoptions('fmincon', 'MaxFunctionEvaluations', 10^5, 'MaxIterations', 10^4);
    theta_opt = fmincon(obj, [1.5,0.5], [], [], [], [], [1.0001,0.000001], [10, inf], [], options);
    b = theta_opt(1);
    b_minus = 1 - b;
    a = theta_opt(2);
    Fitted_vals_2 = (Q0^b_minus - b_minus*a*t).^(1/b_minus);
    Fitted_vals_2 = log(Fitted_vals_2/Fitted_vals_2(1));
    mse = computeMSE(y, Fitted_vals_2);
    nse = computeNSE(y, Fitted_vals_2);
    aic_2 = length(y)*log(mse) + 2*2;
    
    diagnostics2.mse = mse;
    diagnostics2.nse = nse;
    diagnostics2.aic = aic_2;
    
    if aic_1 <= aic_2
        diagnostics  = diagnostics1;
        mdl.b = 1;
        mdl.a = -mdl1.Coefficients.Estimate(1);
        mdl.Fitted = mdl1.Fitted;
    else
        diagnostics  = diagnostics2;
        mdl.b = b;
        mdl.a = a;
        mdl.Fitted = Fitted_vals_2;
    end
    

end

function nse = computeNSE(obs, pred)
    nse = 1 - sum((obs - pred).^2)/sum((obs - mean(obs)).^2);
end

function mse = computeMSE(obs, pred)
    mse = mean((obs - pred).^2);
end