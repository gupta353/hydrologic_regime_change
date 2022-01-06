%{
Computation of master recession using power law model
%}

function mrc_comp = power_law(mrc, b, a)
    
    t_pot = length(mrc) + [1:200]';
    if b == 1
        mrc_ext = mrc(1)*exp(-a*t_pot);
    else
        b_minus = 1 - b;
        mrc_ext = (mrc(1)^(b_minus) - b_minus*a*t_pot).^(1/b_minus);
    end
    
    % eliminate the part of mrc that is below 1 percent of peak value
    mrc_comp = [mrc;mrc_ext];
    ratio = mrc_comp/mrc(1);
    ind = find(ratio<=0.01, 1);
    if ~isempty(ind)
        mrc_comp = mrc_comp(1:ind);
    end
    
end