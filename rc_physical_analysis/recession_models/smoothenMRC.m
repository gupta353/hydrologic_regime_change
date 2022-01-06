%{
The script to test the smoothening of MRC

%}
clear all
close all
clc

direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis';
%local_dir = '08109700';
%fname = 'MRC_08109700.txt';


list = dir(direc);

for list_ind = 1:length(list)
    
    local_dir = list(list_ind).name;
    fname = ['MRC_',local_dir,'.txt'];
    
    
    filename = fullfile(direc, local_dir, fname);
    
    if isfile(filename)
        
        % read data
        fid = fopen(filename, 'r');
        data = textscan(fid,'%f%f', 'headerlines', 1, 'delimiter', '\t');
        fclose(fid);
        
        mrc = data{2};
        mrc_smoothed = irwsm(mrc,1,0.8);
        % plot mrc data
        subplot(2,3,1)
        plot(0:length(mrc_smoothed)-1,mrc_smoothed)
        hold on
        scatter(0:length(mrc)-1,mrc,'filled')
        hold off
        mrc = mrc_smoothed;
        
        %% Linear relationship
        %
        [mdl, Fitted_vals_2, diagnostics1, diagnostics2, b, a] = power_law(mrc);
        Qt_Q0 = mrc/mrc(1);
        y = log(Qt_Q0);
        t = [0:length(y)-1]';
        subplot(2,3,2)
        scatter(t, y, 'filled')
        hold on
        plot(t, mdl.Fitted, 'color', 'black')
        plot(t, Fitted_vals_2, 'color', 'red')
        hold off
        legend({'Observations', 'b = 1', ['b = ',num2str(b)]})
        title({['NSE (b = 1) = ', num2str(diagnostics1.nse)],['NSE (b != 1) = ', num2str(diagnostics2.nse)], ['aic = ', num2str(diagnostics2.aic)]})
        legend('boxoff')
        %}
        
        %% piecewise linear
        [mdl1, mdl2, diagnostics, mse_list, cont_satisfied, t_pl1, t_pl2, y_pl1, y_pl2] = piecewise_linear(mrc);
        
        %
        subplot(2,3,3)
        scatter(t, y, 'filled')
        hold on
        plot(t_pl1, mdl1.Fitted, 'color', 'black')
        plot(t_pl2, mdl2.Fitted, 'color', 'black')
        hold off
        title({['NSE = ', num2str(diagnostics.nse)], ['Continuity satisfed = ', cont_satisfied], ['aic = ', num2str(diagnostics.aic)]})
        %}
        %% Horton's relationship
        [mdl, diagnostics, t_h, y_h, m_opt] = Hortons_relationship(mrc);
        %
        subplot(2,3,4)
        scatter(t, y, 'filled')
        hold on
        plot(t, mdl.Fitted, 'color', 'black')
        hold off
        title(['NSE = ',num2str(diagnostics.nse)], ['aic = ', num2str(diagnostics.aic)])
        %}
        %% Boussinesq 1904 relationship
        [diagnostics, t_tmp, y, a3_opt] = Boussinesq_1904(mrc);
        %
        subplot(2,3,5)
        scatter(t, y, 'filled')
        hold on
        plot(t, t_tmp, 'color', 'black')   % t_tmp is fitted value of y in this case
        hold off        
        title(['NSE = ',num2str(diagnostics.nse)])
        %}
        
        %% Ishigara and Takagi (1965) method
        [mdl, a3_opt, diagnostics, fitted_vals, mse_list, cont_sat, t_tmp1, t_tmp2, y_tmp1, y_tmp2] = Ishigara_Takagi(mrc);
        %
        subplot(2,3,6)
        scatter(t, y, 'filled')
        hold on
        plot(t_tmp1, fitted_vals(1:length(t_tmp1)), 'color', 'black')
        plot(t_tmp2, fitted_vals(length(t_tmp1)+1:end), 'color', 'black')
        hold off        
        title({['NSE = ',num2str(diagnostics.nse)], ['Continuity satisfied = ', cont_sat]})
        pause;
        %}
    end
    
end

% power law relationship (Brutsaert and Nieber, 1976)
function [mdl, Fitted_vals_2, diagnostics1, diagnostics2, b, a] = power_law(mrc)

    Qt_Q0 = mrc/mrc(1);
    y = log(Qt_Q0);
    t = [0:length(y)-1]';
    
    % b = 1
    mdl = fitlm(t,y, 'Intercept', false);
    mse = computeMSE(y, mdl.Fitted);
    nse = computeNSE(y, mdl.Fitted);
    aic_1 = length(y)*log(mse) + 2;
    
    diagnostics1.mse = mse;
    diagnostics1.nse = nse;
    diagnostics1.aic = aic_1;
    
    % b ~= 1
    Q0 = mrc(1);
    obj = @(theta)sum((mrc.^(1-theta(1)) - (...
        Q0^(1-theta(1)) - (1-theta(1))*theta(2)*t...
        )).^2);
    theta_opt = fmincon(obj, [1.5,0.5], [], [], [], [], [1.0001,0.000001], [10, inf]);
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
    

end

% piecewise linear relationship
function [mdl1, mdl2, diagnostics, mse_list, cont_satisfied, t_tmp1, t_tmp2, y_tmp1, y_tmp2] = piecewise_linear(mrc)

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
        cont_satisfied = 'True';
        
    else                       % if continuity condition is not satisfied

        ind  = find(mse_list(:,4)==min(mse_list(:,4)));
        
        t_ind = find(t == mse_list(ind,1));
        t_tmp1 = t(1:t_ind);
        y_tmp1 = y(1:t_ind);
        t_tmp2 = t(t_ind+1:end);
        y_tmp2 = y(t_ind+1:end);

        mdl1 = fitlm(t_tmp1, y_tmp1, 'Intercept', false);
        mdl2 = fitlm(t_tmp2, y_tmp2);
        
        cont_satisfied = 'False';

    end
    
    fitted_vals = [mdl1.Fitted;mdl2.Fitted];
    mse_fit = computeMSE(y, fitted_vals);
    nse_fit = computeNSE(y, fitted_vals);
    aic_fit =  length(y)*log(mse_fit) + 2*(mdl1.NumCoefficients + mdl2.NumCoefficients);

    diagnostics.aic = aic_fit;
    diagnostics.mse = mse_fit;
    diagnostics.nse = nse_fit;
    
        
end

% Horton's double exponential relationship
function [mdl, diagnostics, t_tmp, y, m_opt] = Hortons_relationship(mrc)
    
    Qt_Q0 = mrc/mrc(1);
    y = log(Qt_Q0);
    t = [0:length(y)-1]';

    m_list = 0:0.05:10;
    for count = 1:length(m_list)

        m = m_list(count);
        t_tmp = t.^m;
        mdl = fitlm(t_tmp,y, 'Intercept', false);
        mse_list(count) = mdl.MSE;

    end
    ind = find(mse_list == min(mse_list));
    m_opt = m_list(ind);
    t_tmp = t.^m_opt;
    mdl = fitlm(t_tmp,y, 'Intercept', false);
    
    nse = computeNSE(y, mdl.Fitted);
    mse = computeMSE(y, mdl.Fitted);
    aic = length(y)*log(mse) + 2*2;     % number of parameters = 2
    
    
    diagnostics.aic = aic;
    diagnostics.mse = mse;
    diagnostics.nse = nse;
 
end

% Boussinesq 1904 relationship
function [diagnostics, t_tmp, y, a3_opt] = Boussinesq_1904(mrc)

    Qt_Q0 = mrc/mrc(1);
    y = log(Qt_Q0);
    t = [0:length(y)-1]';
    
    % local optimum by fmincon
    obj = @(a3)sum(y + 2*log(1 + a3*t)).^2;
    a3_opt = fmincon(obj, 0.02, [], [], [], [], 0, inf);
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
    
end

% Ishigara and Takagi (1965) relationship
function [mdl, a3_opt, diagnostics, fitted_vals, mse_list, cont_sat,t_tmp1, t_tmp2, y_tmp1, y_tmp2] = Ishigara_Takagi(mrc)

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
        
        cont_sat = 'True';
        
        mse_list_cont_sat = mse_list(ind_cont_sat,:);
        ind  = find(mse_list_cont_sat(:,2) == min(mse_list_cont_sat(:,2)));
        t_ind = find(t == mse_list_cont_sat(ind,1));
        t_tmp1 = t(1:t_ind);
        y_tmp1 = y(1:t_ind);
        t_tmp2 = t(t_ind+1:end);
        y_tmp2 = y(t_ind+1:end);

        mdl = fitlm(t_tmp1, y_tmp1, 'Intercept', false);
        if ~isempty(t_tmp2)
            [a3_opt, y_fitted2] = Boussinesq_1904_paramOpt(t_tmp2,y_tmp2);
            fitted_vals = [mdl.Fitted;y_fitted2];
        else
        a3_opt = 0;
        fitted_vals = mdl.Fitted;
        end
        
    else % if continuty is not satisfied at any point
        
        cont_sat = 'False';
        
        ind  = find(mse_list(:,2) == min(mse_list(:,2)));
        t_ind = find(t == mse_list(ind,1));
        t_tmp1 = t(1:t_ind);
        y_tmp1 = y(1:t_ind);
        t_tmp2 = t(t_ind+1:end);
        y_tmp2 = y(t_ind+1:end);

        mdl = fitlm(t_tmp1, y_tmp1, 'Intercept', false);
        if ~isempty(t_tmp2)
            [a3_opt, y_fitted2] = Boussinesq_1904_paramOpt(t_tmp2,y_tmp2);
            fitted_vals = [mdl.Fitted;y_fitted2];
        else
        a3_opt = 0;
        fitted_vals = mdl.Fitted;
        end
    end
    
    mse_fit = computeMSE(y, fitted_vals);
    nse_fit = computeNSE(y, fitted_vals);
    aic_fit =  length(y)*log(mse_fit) + 2*(mdl.NumCoefficients + 1);
    
    diagnostics.aic = aic_fit;
    diagnostics.mse = mse_fit;
    diagnostics.nse = nse_fit;
    
end

% sum of n exponentials (maximum value of n is set to 3 )
%{
function exp_sums(t,y)
    
    % n = 1
    obj = @(theta)sum((y - log(theta(1)) - theta(2)*t).^2);
    theta_opt = fmincon(obj, [1,0.2], [], [], [], [], [0, 0], [inf,inf]);
    
    % n = 2
    
end
%}

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
    a3_opt = fmincon(obj, 0.02, [], [], [], [], 0, inf);
    y_fitted = -2*log(1 + a3_opt*t);
    
end