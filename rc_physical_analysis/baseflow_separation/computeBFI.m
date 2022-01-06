%{
Computation of baseflow index = total-baseflow/total-streamflow
inputs: strm = streamflow time-series (this code is written for daily timescale series but would work for finer timescales also)
        a = linear recession constant (b_k = a*b_(k-1); a = exp(-k))
outputs: BFI  = baseflwo index (an estimate)
         baseflow = baseflow value

Ref: (Collischonn and Fan, 2013)

%}

function [BFI,baseflow] = computeBFI(strm, a)
    
    n = length(strm);
    baseflow = zeros(n,1);
    baseflow(end) = strm(end);
    for ind = n-1:-1:1
        baseflow_tmp = baseflow(ind+1)/a;
        baseflow(ind) = min(baseflow_tmp, strm(ind));
    end
    BFI = sum(baseflow)/sum(strm);
    
end