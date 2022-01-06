%{
Recession model: Boussinesq 1904 relationship model
%}

function mrc_comp = Boussinesq_1904(mrc, a3)
    
    t_pot = length(mrc) + [1:200]';
    mrc_ext = mrc(1)*(1 + a3*t_pot).^(-2);
    
    % eliminate the part of mrc that is below 1 percent of peak value
    mrc_comp = [mrc;mrc_ext];
    ratio = mrc_comp/mrc(1);
    ind = find(ratio<=0.01, 1);
    if ~isempty(ind)
        mrc_comp = mrc_comp(1:ind);
    end
    
end