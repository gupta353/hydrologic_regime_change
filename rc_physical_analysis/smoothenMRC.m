%{
Smoothening of mrc and extension of mrc by using recession relationships

%}
clear all
close all
clc

direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis';
%local_dir = '08109700';
%fname = 'MRC_08109700.txt';


list = dir(direc);

for list_ind = 497:length(list)
    
    local_dir = list(list_ind).name;
    fname = ['MRC__post_processed','.txt'];
    
    direc_tmp = fullfile(direc, local_dir);
    filename = fullfile(direc_tmp, fname);
    
    if isfile(filename)
        aic_list = [];
        % read data
        fid = fopen(filename, 'r');
        data = textscan(fid,'%f%f', 'headerlines', 1, 'delimiter', '\t');
        fclose(fid);
        
        mrc = data{2};
        % plot mrc data
        subplot(3,3,1)
        plot(0:length(mrc)-1,mrc,'-o')
        
        if isempty(mrc)   % if mrc data is not available go to the next iteration
            continue
        end
        %% Linear relationship (1)
        %
        [mdl_pol, diagnostics_pol] = power_law_est(mrc);
        aic_list = [aic_list, diagnostics_pol.aic];
        
        
        Qt_Q0 = mrc/mrc(1);
        y = log(Qt_Q0);
        t = [0:length(y)-1]';
        subplot(3,3,2)
        scatter(t, y, 'filled')
        hold on
        plot(t, mdl_pol.Fitted, 'color', 'black')
        hold off
        legend({'Observations', ['Fitted, b = ',num2str(mdl_pol.b)]})
        title({['NSE = ', num2str(diagnostics_pol.nse)], ['aic = ', num2str(diagnostics_pol.aic)],})
        legend('boxoff')
        %}
        
        %% piecewise linear (2)
        [mdl_pl, diagnostics_pl] = piecewise_linear_est(mrc);
        aic_list = [aic_list, diagnostics_pl.aic];
        %
        subplot(3,3,3)
        scatter(t, y, 'filled')
        hold on
        plot(mdl_pl.t_pl1, mdl_pl.mdl1.Fitted, 'color', 'black')
        plot(mdl_pl.t_pl2, mdl_pl.mdl2.Fitted, 'color', 'black')
        hold off
        title({['NSE = ', num2str(diagnostics_pl.nse)], ['Continuity satisfed = ', mdl_pl.cont_satisfied], ['aic = ', num2str(diagnostics_pl.aic)]})
        %}
        %% Horton's relationship (3)
        [mdl_h, diagnostics_h] = Hortons_relationship_est(mrc);
        aic_list = [aic_list, diagnostics_h.aic];
        %
        subplot(3,3,4)
        scatter(t, y, 'filled')
        hold on
        plot(t, mdl_h.Fitted, 'color', 'black')
        hold off
        title(['NSE = ',num2str(diagnostics_h.nse)], ['aic = ', num2str(diagnostics_h.aic)])
        %}
        %% Boussinesq 1904 relationship (4)
        [mdl_B1904, diagnostics_B1904] = Boussinesq_1904_est(mrc);
        aic_list = [aic_list, diagnostics_B1904.aic];
        %
        subplot(3,3,5)
        scatter(t, mdl_B1904.obs, 'filled')
        hold on
        plot(t, mdl_B1904.Fitted, 'color', 'black')   % t_tmp is fitted value of y in this case
        hold off        
        title(['NSE = ',num2str(diagnostics_B1904.nse)])
        %}
        
        %% Ishigara and Takagi (1965) method (5)
        [mdl_IT, diagnostics_IT] = Ishigara_Takagi_est(mrc);
        aic_list = [aic_list, diagnostics_IT.aic];
        %
        subplot(3,3,6)
        scatter(t, y, 'filled')
        hold on
        plot(mdl_IT.t_tmp1, mdl_IT.Fitted1, 'color', 'black')
        plot(mdl_IT.t_tmp2, mdl_IT.Fitted2, 'color', 'black')
        hold off        
        title({['NSE = ',num2str(diagnostics_IT.nse)], ['Continuity satisfied = ', mdl_IT.cont_sat]})
        %}
        
        %% Sum of n exponentials (6)
        [mdl_ne, diagnostics_ne] = exp_sums_est(mrc);
        aic_list = [aic_list, diagnostics_ne.aic];
        
        %
        subplot(3,3,7)
        scatter(t, y, 'filled')
        hold on
        plot(t, mdl_ne.Fitted, 'color', 'black')
        legend({'Observations',['Fitted, n = ', num2str(mdl_ne.n)]})
        legend('boxoff')
        hold off
        title(['R^2 = ', num2str(diagnostics_ne.nse)]);
        %}
        %% Idenitfy the model with minimum AIC
        %
        ind = find(aic_list == min(aic_list));
        if ind == 1
            mdl_final = mdl_pol;
            mdl_final.model = 'Power law (linear on log-log scale)';
            diagnostics_final = diagnostics_pol;
            mrc_comp = power_law(mrc, mdl_final.b, mdl_final.a);
        elseif ind ==2
            mdl_final = mdl_pl;
            mdl_final.model = 'Piecewise linear on log-log scale';
            diagnostics_final = diagnostics_pl;
            mrc_comp = piecewise_linear(mrc, mdl_final.mdl1, mdl_final.mdl2);
        elseif ind == 3
            mdl_final = mdl_h;
            mdl_final.model = 'Horton''s double exponential';
            diagnostics_final = diagnostics_h;
            mrc_comp = Hortons_relationship(mrc, mdl_final.m, mdl_final.a2);
        elseif ind == 4
            mdl_final = mdl_B1904;
            mdl_final.model = 'Boussinesq 1904';
            diagnostics_final = diagnostics_B1904;
            mrc_comp  = Boussinesq_1904(mrc, mdl_final.a3);
        elseif ind == 5
            mdl_final = mdl_IT;
            mdl_final.model = 'Ishigara and Takagi';
            diagnostics_final = diagnostics_IT;
            mrc_comp  = Ishigara_Takagi(mrc, mdl_final.a1, mdl_final.a3);
        elseif ind == 6
            mdl_final = mdl_ne;
            mdl_final.model = 'sum of n exponentials';
            diagnostics_final = diagnostics_ne;
            mrc_comp = exp_sums(mrc, mdl_final, mdl_final.n);
        end
        
        subplot(3,3,8)
        scatter(1:length(mrc_comp), mrc_comp);
        pause;
        %}
        %% write completed mrc data to textfile
        %
        sname = 'MRC_completed.txt';
        filename = fullfile(direc_tmp,sname);
        fid = fopen(filename,'w');
        fprintf(fid,'%s\t%s\n','time-step(days)','Streamflow(cms)');
        for write_ind = 1:length(mrc_comp)
            fprintf(fid,'%f\t%f\n',write_ind-1,mrc_comp(write_ind));
        end
        fclose(fid);
        %}
    end
    
end