%{
Recession  model: Ishigara and Takagi relationship
%}

function mrc_comp = Ishigara_Takagi(mrc, a1, a3)

    t_pot = [1:200]';
    mrc_ext = mrc(end)*(1 + a3*t_pot).^(-2);
    
    % eliminate the part of mrc that is below 1 percent of peak value
    mrc_comp = [mrc;mrc_ext];
    ratio = mrc_comp/mrc(1);
    ind = find(ratio<=0.01, 1);
    mrc_comp = mrc_comp(1:ind);

end