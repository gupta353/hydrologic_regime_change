%{
this script tests the fmincon method and manual optimization method for
Horton's recession relationship

Author: Abhinav Gupta (Created: 13 Dec 2021)

%}

clear all
close all
clc

direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis';
%local_dir = '08109700';
%fname = 'MRC_08109700.txt';


list = dir(direc);
lc = 0;
for list_ind = 1:length(list)
    local_dir = list(list_ind).name;
    fname = ['MRC_',local_dir,'.txt'];
    
    filename = fullfile(direc, local_dir, fname);
    
    if isfile(filename)
        lc = lc + 1;
        
        % read data
        fid = fopen(filename, 'r');
        data = textscan(fid,'%f%f', 'headerlines', 1, 'delimiter', '\t');
        fclose(fid);
        
        mrc = data{2};
        
        %
        Qt_Q0 = mrc/mrc(1);
        y = log(Qt_Q0);
        t = [0:length(y)-1]';
        
        %% manual method
        m_list = 0:0.05:10;
        for count = 1:length(m_list)
            
            m = m_list(count);
            t_tmp = t.^m;
            mdl = fitlm(t_tmp,y, 'Intercept', false);
            mse_list(count) = computeMSE(y, mdl.Fitted);
            
        end
        ind = find(mse_list == min(mse_list));
        m_opt_manual(lc) = m_list(ind);
        t_tmp = t.^m_opt_manual(lc);
        mdl = fitlm(t_tmp,y, 'Intercept', false);
        a2_opt_manual(lc) = -mdl.Coefficients.Estimate(1);
        
        %% fmincon method
        obj = @(theta)sum((y + theta(2)*t.^theta(1)).^2);
        theta_opt = fmincon(obj, [0.5,1], [], [], [], [], [-inf,0], [inf,inf]);
        m_opt_fmincon(lc) = theta_opt(1);
        a2_opt_fmincon(lc) = theta_opt(2);
    end
end

function mse = computeMSE(obs, pred)
    mse = mean((obs - pred).^2);
end