%{ 
Parameter estimtion: piecewise linear recession relationship
%}
function [mdl, diagnostics] = piecewise_linear_est(mrc)

    Qt_Q0 = mrc/mrc(1);
    y = log(Qt_Q0);
    t = [0:length(y)-1]';
    
    for t_ind = 2:length(t)-1

        t_tmp1 = t(1:t_ind);
        y_tmp1 = y(1:t_ind);
        t_tmp2 = t(t_ind+1:end);
        y_tmp2 = y(t_ind+1:end);

        mdl1 = fitlm(t_tmp1, y_tmp1, 'Intercept', false);
        mdl2 = fitlm(t_tmp2, y_tmp2);
        
        intercept1 = 0;
        intercept2 = mdl2.Coefficients.Estimate(1);
        slope1 = mdl1.Coefficients.Estimate(1);
        slope2 = mdl2.Coefficients.Estimate(2);
        
        point_of_continutity = (intercept1 - intercept2)/(slope2 - slope1);
        
        residuals_comb = [mdl1.Residuals.Raw; mdl2.Residuals.Raw];
        mse_comb = mean(residuals_comb.^2);
        
        if point_of_continutity <= t_tmp2(1) && point_of_continutity >= t_tmp1(end)
            mse_list(t_ind-1,:) = [t(t_ind),mdl1.MSE,mdl2.MSE,mse_comb,1];
        else
            mse_list(t_ind-1,:) = [t(t_ind),mdl1.MSE,mdl2.MSE,mse_comb,0];
        end
        
    end
    
    ind_cont_sat = find(mse_list(:,5) == 1);
    mse_list_cont_sat = mse_list(ind_cont_sat, :);
    ind  = find(mse_list_cont_sat(:,4)  == min(mse_list_cont_sat(:,4)));
    
    if ~isempty(ind)            % if continutity condition is satisfied at some optimal partition 
        
        t_ind = find(t == mse_list_cont_sat(ind,1));
        t_tmp1 = t(1:t_ind);
        y_tmp1 = y(1:t_ind);
        t_tmp2 = t(t_ind+1:end);
        y_tmp2 = y(t_ind+1:end);

        mdl1 = fitlm(t_tmp1, y_tmp1, 'Intercept', false);
        mdl2 = fitlm(t_tmp2, y_tmp2);
        
        mdl.cont_satisfied = 'True';
        mdl.t_pl1 = t_tmp1;
        mdl.t_pl2 = t_tmp2;
        
    else                       % if continuity condition is not satisfied

        ind  = find(mse_list(:,4)==min(mse_list(:,4)));
        
        t_ind = find(t == mse_list(ind,1));
        t_tmp1 = t(1:t_ind);
        y_tmp1 = y(1:t_ind);
        t_tmp2 = t(t_ind+1:end);
        y_tmp2 = y(t_ind+1:end);

        mdl1 = fitlm(t_tmp1, y_tmp1, 'Intercept', false);
        mdl2 = fitlm(t_tmp2, y_tmp2);
        
        mdl.cont_satisfied = 'False';
        mdl.t_pl1 = t_tmp1;
        mdl.t_pl2 = t_tmp2;

    end
    
    fitted_vals = [mdl1.Fitted;mdl2.Fitted];
    mse_fit = computeMSE(y, fitted_vals);
    nse_fit = computeNSE(y, fitted_vals);
    aic_fit =  length(y)*log(mse_fit) + 2*(mdl1.NumCoefficients + mdl2.NumCoefficients);

    diagnostics.aic = aic_fit;
    diagnostics.mse = mse_fit;
    diagnostics.nse = nse_fit;
    
    mdl.mdl1 = mdl1;
    mdl.mdl2 = mdl2;

end

function nse = computeNSE(obs, pred)
    nse = 1 - sum((obs - pred).^2)/sum((obs - mean(obs)).^2);
end

function mse = computeMSE(obs, pred)
    mse = mean((obs - pred).^2);
end