% This routine completes the runoff coefficints for each rainfall-runoff
% event
% inputs: period = a cell array of structures such that each structure 
%                  contains the rainfall (in mm at daily time-scale) and complete 
%                  streamflow (with MRC appended) data (in cms at daily time-scale)
%         darea = total drainage area of streamflow stations (in km2)
% output: runoff_coeff = runoff-coefficient of each rainfall-runoff event
%         strm_tmp_vol = total volume of streamflow in each rainfall-runoff
%         event
%         prcp_tmp_vol = total volume of rainfall in each rainfall-runoff
%         event

function [runoff_coeff,strm_tmp_vol,prcp_tmp_vol]=runoff_coff_comp(period,darea)
    
    
    for per_ind=1:length(period);
        
        strm_tmp=period{per_ind}.completed_streamflow;
        strm_tmp_vol(per_ind)=sum(strm_tmp)*3600*24/darea/1000;         % total volume of streamflow in mm
        prcp_tmp=period{per_ind}.rain;
        prcp_tmp_vol(per_ind)=sum(prcp_tmp);
        runoff_coeff(per_ind)=strm_tmp_vol(per_ind)/prcp_tmp_vol(per_ind);
        
    end

end