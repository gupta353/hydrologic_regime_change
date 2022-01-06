%{
Parameter estimtion: Horton's double exponential recession relationship
%}
function [mdl, diagnostics] = Hortons_relationship_est(mrc)
    
    Qt_Q0 = mrc/mrc(1);
    y = log(Qt_Q0);
    t = [0:length(y)-1]';
    
    %% manual optimization method
    %{
    m_list = 0:0.05:10;
    for count = 1:length(m_list)

        m = m_list(count);
        t_tmp = t.^m;
        mdl = fitlm(t_tmp,y, 'Intercept', false);
        mse_list(count) = computeMSE(y, mdl.Fitted);

    end
    ind = find(mse_list == min(mse_list));
    m_opt = m_list(ind);
    t_tmp = t.^m_opt;
    mdl = fitlm(t_tmp,y, 'Intercept', false);
    
    nse = computeNSE(y, mdl.Fitted);
    mse = computeMSE(y, mdl.Fitted);
    aic = length(y)*log(mse) + 2*2;     % number of parameters = 2
    %}
    
    %% fmincon method
    obj = @(theta)sum((y + theta(2)*t.^theta(1)).^2);
    theta_opt = fmincon(obj, [0.5,1], [], [], [], [], [-inf,0], [inf,inf]);
    m_opt = theta_opt(1);
    a2_opt = theta_opt(2);
    
    Fitted = -a2_opt*t.^m_opt;
    
    mse = computeMSE(y, Fitted);
    nse = computeNSE(y, Fitted);
    aic = length(y)*log(mse) + 2*2;
    
    diagnostics.aic = aic;
    diagnostics.mse = mse;
    diagnostics.nse = nse;
    
    mdl.Fitted = Fitted;
    mdl.m_opt = m_opt;
    mdl.a2_opt = a2_opt;
 
end

function nse = computeNSE(obs, pred)
    nse = 1 - sum((obs - pred).^2)/sum((obs - mean(obs)).^2);
end

function mse = computeMSE(obs, pred)
    mse = mean((obs - pred).^2);
end