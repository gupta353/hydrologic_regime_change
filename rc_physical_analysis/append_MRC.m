% This routine appends MRC to the tail os streamflow hydrograph
% input: strm = streamflow hydrograph
%        mrc = master recession curve in same units of measurement as of
%        strm
% output: astrm = streamflow hydrograph with mrc appended (units: same as that of strm)
%         added_strm = part of mrc that is appended to streamflow
%         hydrograph

function [astrm,added_strm]=append_MRC(strm,mrc)
    
    tail_strm=strm(end);
    ind_min=find(mrc<=tail_strm,1);
    
    if ~isempty(ind_min)
        added_strm=mrc(ind_min+1:end);
        astrm=[strm;added_strm];
    else
        added_strm=[];
        astrm=strm;
    end
    
    % correction for negative strm
    neg_astrm=min(astrm(astrm<0));
    if ~isempty(neg_astrm)
        astrm(astrm<0)=0;
        disp(['Streamfloe correction = ',num2str(abs(neg_astrm))]);
    end
    
end