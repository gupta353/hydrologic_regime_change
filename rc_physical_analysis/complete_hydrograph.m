% This routine takes incomplete streamflow hydrographs as input and
% completes the hydrograph by appropriately appending MRC to the hydrograph
% inputs: period = a cell array of structures such that each structure
%                 contains rainfall and corresponding streamflow
%                 (incomplete) data (in m^3 s^-1)
%         mrc = master recession curve (in cms)
% output: periodc = a cell array of structures such that each structure
%                 contains rainfall, corresponding incomplete streamflow
%                 and complete streamflow (MRC appened) (in m^3 s^-1)


function periodc=complete_hydrograph(period,mrc)
    
    periodc=period;
    strm_ind=1;
    strm1=period{strm_ind}.streamflow;
    [strm,added_strm]=append_MRC(strm1,mrc);
    periodc{strm_ind}.completed_streamflow=strm;
    
    for strm_ind=2:length(period)
        
        strm1=period{strm_ind}.streamflow;
        
        % subtract the stremflow due to previous events
        [strm,residual_added_strm]=subtract_columns(strm1,added_strm);
        
        % complete the streamflow by appending the MRC
        [strm,added_strm]=append_MRC(strm,mrc);
        
        % add added_strm and residual_added_strm
        added_strm=add_columns(added_strm,residual_added_strm);
        
        periodc{strm_ind}.completed_streamflow=strm;
    end
end