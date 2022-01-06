%{
This script test the definition of the incomplete gamma function

%}

clear all
close all
clc

N = 100; % number of terms to be used

beta = 10;
alpha = 10.5;
t = 0.1:0.1:3;

%% using MATLAB function
Y = gammainc(beta*t,alpha);
plot(t,Y)
pause;

%% manual computation
in_gam_func = 0;
for k = 0:N
    T = ((beta*t).^(alpha+k)).*exp(-beta*t)/gamma(alpha+k+1);
    in_gam_func = in_gam_func + T;
    %plot(t,in_gam_func);
    %pause
end
hold on
plot(t,in_gam_func,'o')


