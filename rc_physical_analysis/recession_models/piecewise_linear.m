%{
recession model: piecewise linear
%}
function mrc_comp = piecewise_linear(mrc, mdl1, mdl2)
    
    length_t1 = length(mdl1.Fitted);
    length_t2 = length(mdl2.Fitted);
    t_pot = length_t1 + length_t2 + [1:200]';
    intercept = mdl2.Coefficients.Estimate(1);
    slope = mdl2.Coefficients.Estimate(2);
    mrc_ext = mrc(1)*exp(intercept + slope*t_pot );

    % eliminate the part of mrc that is below 1 percent of peak value
    mrc_comp = [mrc;mrc_ext];
    ratio = mrc_comp/mrc(1);
    ind = find(ratio<=0.01, 1);
    if ~isempty(ind)
        mrc_comp = mrc_comp(1:ind);
    end
    
end