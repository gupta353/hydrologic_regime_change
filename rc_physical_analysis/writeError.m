%{
this script writes a textfile to given directory with the error message
that the [data] contains NaNs
%}
function writeError(save_dir, fname, string)

    filename = fullfile(save_dir,fname);
    fid = fopen(filename,'w');
    fprintf(fid,'%s',string);
    fclose(fid);

end