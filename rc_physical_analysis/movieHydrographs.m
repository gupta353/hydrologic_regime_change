%{
This script creates a video of streamflow hydrographs created using the MRC
method

Author: Abhinav Gupta (Created: 30 Nov 2021)
%}

clear all
close all
clc

direc = 'D:/Research/non_staitionarity/codes/results/rc_physical_analysis/';
basin = '07068000';
fname = 'rainfall_runoff_data.mat';

% read data
filename = fullfile(direc, basin, fname);
load(filename);

% create movie
% vidfile_name = fullfile(direc, basin, 'hydrograph_movie.mp4');
% vidfile = VideoWriter(vidfile_name,'MPEG-4');
% vidfile.FrameRate = 1/2;
% open(vidfile);
for per_ind=1:length(period)
    
    yyaxis left
    plot(period{per_ind}.completed_streamflow);
    hold on
    plot(period{per_ind}.streamflow);
    hold off
    
    yyaxis right
    bar(period{per_ind}.rain, 'Barwidth',0.1);
    set(gca, 'YDir','reverse')
    title({['Runoff-coefficient = ',num2str(period{per_ind}.runoff_coefficient)], ['Time-period = ', period{per_ind}.begin_date,' to ',period{per_ind}.end_date]})
    pause;
    %F(per_ind) = getframe(gcf); 
    %writeVideo(vidfile,F(per_ind));
    
end
%close(vidfile)