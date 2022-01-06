%{
this script tests the fmincon method and manual optimization method for
Boussinesq 1904 recession relationship

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
for list_ind = 3:length(list)
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
        
        % manual method
        a3_list = 0.01:0.01:10;
        
        for count = 1:length(a3_list)
            
            a3 = a3_list(count);
            t_tmp = -2*log(1 + a3*t);
            mse_list(count) = computeMSE(y, t_tmp);
            
        end
        ind = find(mse_list == min(mse_list));
        a3_opt_manual(lc) = a3_list(ind);
        t_tmp_manual = -2*log(1 + a3_opt_manual(lc)*t);
        
        obj = @(a3)sum(y + 2*log(1 + a3*t)).^2;
        a3_opt_fmincon(lc) = fmincon(obj, 0.02, [], [], [], [], 0, inf);
        t_tmp_fmincon = -2*log(1 + a3_opt_fmincon(lc)*t);
    end
end

function mse = computeMSE(obs, pred)
mse = mean((obs - pred).^2);
end