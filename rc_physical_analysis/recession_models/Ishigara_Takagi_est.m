%{
Parameter estimtion: Ishigara and Takagi (1965) recession relationship 
%}
function [mdl1, diagnostics] = Ishigara_Takagi_est(mrc)

    Qt_Q0 = mrc/mrc(1);
    y = log(Qt_Q0);
    t = [0:length(y)-1]';
       
    for t_ind = 2:length(t)

        t_tmp1 = t(1:t_ind);
        y_tmp1 = y(1:t_ind);
        t_tmp2 = t(t_ind+1:end);
        y_tmp2 = y(t_ind+1:end);
        
        mdl = fitlm(t_tmp1, y_tmp1, 'Intercept', false);
        if ~isempty(t_tmp2)         % if the two components are actually used
            [a3_opt, y_fitted2] = Boussinesq_1904_paramOpt(t_tmp2,y_tmp2);
            residuals_comb = [mdl.Residuals.Raw; y_tmp2-y_fitted2];

            % check for continuity
            a1 = -mdl.Coefficients.Estimate(1);
            fun = @(x)((1+a3_opt*x)^2/exp(a1*x)-1);
            t_cont = fsolve(fun,(t_tmp1(end) + t_tmp2(1))/2);
            if (t_cont>=t_tmp1(end) && t_cont<=t_tmp2(1)) cont_sat = 1; else cont_sat = 0; end 
        
        else
            residuals_comb = mdl.Residuals.Raw;
            cont_sat = 1;
        end

        mse_comb = mean(residuals_comb.^2);
        mse_list(t_ind-1,:) = [t(t_ind), mse_comb, cont_sat];
    end
    
    ind_cont_sat  = find(mse_list(:,3) == 1);
    if ~isempty(ind_cont_sat)       % if continuity is satisfied at some point
        
        mdl1.cont_sat = 'True';
        
        mse_list_cont_sat = mse_list(ind_cont_sat,:);
        ind  = find(mse_list_cont_sat(:,2) == min(mse_list_cont_sat(:,2)));
        t_ind = find(t == mse_list_cont_sat(ind,1));
        t_tmp1 = t(1:t_ind);
        y_tmp1 = y(1:t_ind);
        t_tmp2 = t(t_ind+1:end);
        y_tmp2 = y(t_ind+1:end);

        mdl = fitlm(t_tmp1, y_tmp1, 'Intercept', false);
        % output variable
        mdl1.Fitted1 = mdl.Fitted;
        mdl1.a1 = -mdl.Coefficients.Estimate(1);
        if ~isempty(t_tmp2)
            [a3_opt, y_fitted2] = Boussinesq_1904_paramOpt(t_tmp2,y_tmp2);
            fitted_vals = [mdl.Fitted;y_fitted2];
            % output variable
            mdl1.Fitted2 = y_fitted2;
            mdl1.a3 = a3_opt;
            mdl1.fitted = fitted_vals;
        else
        a3_opt = 0;
        fitted_vals = mdl.Fitted;
        
        % output variable
        mdl1.Fitted2 = [];
        mdl1.fitted = fitted_vals;
        end

        % output variable
        mdl1.t_tmp1 = t_tmp1;
        mdl1.t_tmp2 = t_tmp2;
        mdl1.y_tmp1 = y_tmp1;
        mdl1.y_tmp2 = y_tmp2;
        
    else % if continuty is not satisfied at any point
        
        mdl1.cont_sat = 'False';
        
        ind  = find(mse_list(:,2) == min(mse_list(:,2)));
        t_ind = find(t == mse_list(ind,1));
        t_tmp1 = t(1:t_ind);
        y_tmp1 = y(1:t_ind);
        t_tmp2 = t(t_ind+1:end);
        y_tmp2 = y(t_ind+1:end);

        mdl = fitlm(t_tmp1, y_tmp1, 'Intercept', false);
        
        % output variable
        mdl1.Fitted1 = mdl.Fitted;  % output variable
        mdl1.a1 = -mdl.Coefficients.Estimates(1);
        if ~isempty(t_tmp2)
            [a3_opt, y_fitted2] = Boussinesq_1904_paramOpt(t_tmp2,y_tmp2);
            fitted_vals = [mdl.Fitted;y_fitted2];

            % output variable
            mdl1.Fitted2 = y_fitted2;
            mdl1.a3 = a3_opt;
            mdl1.fitted = fitted_vals;
        else
            a3_opt = 0;
            fitted_vals = mdl.Fitted;

            % output variable
            mdl1.Fitted2 = [];
            mdl1.fitted = fitted_vals;
        end

        % output variable
        mdl1.t_tmp1 = t_tmp1;
        mdl1.t_tmp2 = t_tmp2;
        mdl1.y_tmp1 = y_tmp1;
        mdl1.y_tmp2 = y_tmp2;

    end
    
    mse_fit = computeMSE(y, fitted_vals);
    nse_fit = computeNSE(y, fitted_vals);
    aic_fit =  length(y)*log(mse_fit) + 2*(mdl.NumCoefficients + 1);
    
    diagnostics.aic = aic_fit;
    diagnostics.mse = mse_fit;
    diagnostics.nse = nse_fit;
   
end

function nse = computeNSE(obs, pred)
    nse = 1 - sum((obs - pred).^2)/sum((obs - mean(obs)).^2);
end

function mse = computeMSE(obs, pred)
    mse = mean((obs - pred).^2);
end

% parameter optimization of the Boussinesq 1904 equation
function [a3_opt, y_fitted] = Boussinesq_1904_paramOpt(t,y)
    
    % manual optimization
    %{
    mse_list = [];
    a3_list = 0:0.01:10;
    for count = 1:length(a3_list)

        a3 = a3_list(count);
        y_fitted = -2*log(1 + a3*t);
        residuals = y - y_fitted;
        mse_list(count) = sum((residuals).^2);
    end
    a3_opt = a3_list(mse_list == min(mse_list));
    y_fitted = -2*log(1 + a3_opt*t);
    %}
    
    % fmincon method
    obj = @(a3)sum((y + 2*log(1 + a3*t)).^2);
    options = optimoptions('fmincon', 'MaxFunctionEvaluations', 10^5, 'MaxIterations', 10^4);
    a3_opt = fmincon(obj, 0.02, [], [], [], [], 0, inf, [], options);
    y_fitted = -2*log(1 + a3_opt*t);
    
end
