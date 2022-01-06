%{
Estimation of linear storage parameter a such that dQ_dt = -kQ,
 and a = exp(-k)

Note: This script also implements piece wise linear model. The model with
lowest aic is selected. if the piecewise linear model is selected, the
parameter 'a' is computed using the lower recession limb

inputs: mrc = master recession curve
outputs: a = linear recession parameter
         nse = nash Sutcliff efficiency of fit between observed and
         simulated mrc
%}

function [a, nse] = linear_recession_parameter(mrc)

    % fit linear model to mrc
    [mdl_pol, diagnostics_pol] = power_law_est_local(mrc);
    
    % fit piecewise linear model
    [mdl_pl, diagnostics_pl] = piecewise_linear_est(mrc);
    
    % determine a
    if diagnostics_pol.aic <= diagnostics_pl.aic
        a = exp(mdl_pol.Coefficients.Estimate(1));
        nse = diagnostics_pol.nse;
    else
        a = exp(mdl_pl.mdl2.Coefficients.Estimate(2));
        nse = diagnostics_pl.nse;
    end
    
end

%%
function [mdl, diagnostics] = power_law_est_local(mrc)

    Qt_Q0 = mrc/mrc(1);
    y = log(Qt_Q0);
    t = [0:length(y)-1]';
    
    %% b = 1
    mdl = fitlm(t,y, 'Intercept', false);
    mse = computeMSE(y, mdl.Fitted);
    nse = computeNSE(y, mdl.Fitted);
    aic = length(y)*log(mse) + 2;
    
    diagnostics.mse = mse;
    diagnostics.nse = nse;
    diagnostics.aic = aic;
    
end

%%
function nse = computeNSE(obs, pred)
    nse = 1 - sum((obs - pred).^2)/sum((obs - mean(obs)).^2);
end

function mse = computeMSE(obs, pred)
    mse = mean((obs - pred).^2);
end