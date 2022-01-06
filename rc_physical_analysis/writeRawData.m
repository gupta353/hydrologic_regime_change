%{
this script writes a textfile to given directory containing streamflow,
rainfall, and evaporation data
%}

function writeRawData(strm_vals, prcp_vals, evap_vals, begin_year, end_year, save_dir,fname)
 
    begin_datenum = datenum([num2str(begin_year),'-10-1'],'yyyy-mm-dd');
    end_datenum = datenum([num2str(end_year),'-9-30'],'yyyy-mm-dd');
    datenums = begin_datenum:end_datenum;
    
    filename = fullfile(save_dir,fname);
    fid = fopen(filename,'w');
    fprintf(fid, '%s\t%s\t%s\t%s\n', 'Date','Streamflow(CMS)','Rainfall(mm/day)','Evaporation(mm/day)');
    
    for wind = 1:length(strm_vals)
        fprintf(fid, '%s\t%f\t%f\t%f\n', datestr(datenums(wind),'yyyy-mm-dd'),strm_vals(wind),prcp_vals(wind),evap_vals(wind));
    end
   fclose(fid);
   
end