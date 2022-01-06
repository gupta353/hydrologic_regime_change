% This routine adjusts MRC

function adj_mrc=adjust_mrc(mrc,strm)
    
    min_neg_strm=min(strm);
    min_mrc=min(mrc);
    
    if abs(min_neg_strm)<min_mrc
        1
        adj_mrc=mrc-min_neg_strm;
    else
        2
        adj_mrc=mrc-min_mrc;
    end
    
end