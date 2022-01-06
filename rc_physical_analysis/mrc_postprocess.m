%{
post-processing of MRC
Ref: Cheng, Zhang, and Brutsaert (2016)

%}

direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis';
%local_dir = '08109700';
%fname = 'MRC_08109700.txt';


list = dir(direc);

for list_ind = 588:length(list)
    local_dir = list(list_ind).name;
    fname = ['MRC_',local_dir,'.txt'];
    
    direc_tmp = fullfile(direc, local_dir);
    filename = fullfile(direc_tmp, fname);
    
    if isfile(filename)
        % read data
        fid = fopen(filename, 'r');
        data = textscan(fid,'%f%f', 'headerlines', 1, 'delimiter', '\t');
        fclose(fid);
        
        mrc = data{2};
        mrc_smoothed = irwsm(mrc,1,0.8);
        % plot mrc data
        subplot(2,3,1)
        scatter(0:length(mrc)-1,mrc,'filled')
        
        %% remove first two points of mrc which are likely to contain surfaceflow
        mrc(1) = [];
        
        %% remove streamflow values below the threshold of 0.0028 cms
        ind = find(mrc<0.0028, 1);
        mrc(ind:end) = [];
        subplot(2,3,2)
        plot(0:length(mrc)-1,mrc,'-o')
        
        %% remove all the Q values after the positive value of dQ_dt
        dQ_dt = compute_dQ_dt(mrc);
        subplot(2,3,3)
        plot(1:length(dQ_dt),dQ_dt,'-o')
        title('dQ/dt');
      
        ind = find(dQ_dt>0);
        if length(ind) > 1
            diff = ind(2:end) - ind(1:end-1);
            ind_diff = find(diff==1);
            if ~isempty(ind_diff)
                ind = ind(ind_diff(1));
                mrc(ind-1:end) = [];
            end
        end
        
        subplot(2,3,4)
        plot(0:length(mrc)-1,mrc,'-o')
        %% remove very high -dQ_dt values
%         dQ_dt = compute_dQ_dt(mrc);
%         dQ_dt_neg = -dQ_dt;
%         TF = ischange(dQ_dt_neg, 'variance', 'MaxNumChanges',1);
%         ind = find(TF == 1);
%         mrc = mrc(ind:end);
%         subplot(3,3,5)
%         plot(0:length(mrc)-1,mrc,'-o')
%         pause;

        %% smoothen the mrc
        mrc_smoothed = irwsm(mrc,1,0.6);
        % plot mrc data
        subplot(2,3,5)
        plot(0:length(mrc_smoothed)-1,mrc_smoothed)
        hold on
        scatter(0:length(mrc)-1,mrc,'filled')
        hold off
        mrc = mrc_smoothed;
        
         %% user input for the part of the MRC to be kept
        t_begin = input('first time-step: ');
        t_end = input('second time-step: ');
        mrc = mrc(t_begin+1:t_end+1);
        subplot(2,3,6)
        plot(0:length(mrc)-1,mrc)
        pause;
        
        % write mrc data to a textfile
        sname=['MRC_','_post_processed.txt'];
        filename=fullfile(direc_tmp,sname);
        fid=fopen(filename,'w');
        fprintf(fid,'%s\t%s\n','time-step(days)','Streamflow(cms)');
        for write_ind=1:length(mrc)
            
            fprintf(fid,'%f\t%f\n',write_ind-1,mrc(write_ind));
            
        end
        fclose(fid);
    end

    
end

%%
function dQ_dt = compute_dQ_dt(mrc)

    Q_i_minus_1 = mrc(1:end-2);
    Q_i_plus_1 = mrc(3:end);

    dQ_dt = (Q_i_plus_1 - Q_i_minus_1)/2;

end