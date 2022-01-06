%{
Recession model: sum of n exponentials
%}

function mrc_comp = exp_sums(mrc, mdl, n)
    
    t_pot = length(mrc) + [1:200]';
    if n == 1
        
        a1 = mdl.a1;
        b1 = mdl.b1;
        mrc_ext = mrc(1)*b1*exp(-a1*t_pot);
        
    elseif n == 2
        
        a1 = mdl.a1;
        b1 = mdl.b1;
        a2 = mdl.a2;
        b2 = mdl.b2;
        mrc_ext = mrc(1)*(b1*exp(-a1*t_pot) + b2*exp(-a2*t_pot));
        
    else
        
        a1 = mdl.a1;
        b1 = mdl.b1;
        a2 = mdl.a2;
        b2 = mdl.b2;
        a3 = mdl.a3;
        b3 = mdl.b3;
        mrc_ext = mrc(1)*(b1*exp(-a1*t_pot) + b2*exp(-a2*t_pot) + b3*exp(-a3*t_pot));
        
    end
    
    % eliminate the part of mrc that is below 1 percent of peak value
    mrc_comp = [mrc;mrc_ext];
    ratio = mrc_comp/mrc(1);
    ind = find(ratio<=0.01, 1);
    if ~isempty(ind)
        mrc_comp = mrc_comp(1:ind);
    end
end