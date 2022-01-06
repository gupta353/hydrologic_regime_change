%{
This script creates and saves runoff coefficients plots

%}

function saveRCPlot(save_dir,sname,prcp_tmp_vol,runoff_coeff)

    figure; 
    scatter(prcp_tmp_vol,runoff_coeff,5,'filled')
    xlabel('Total rainfall volume (mm)', 'fontname', 'arial', 'fontsize', 10)
    ylabel('Runoff coefficient', 'fontname', 'arial', 'fontsize', 10)
    box on
    set(gca, 'fontname', 'arial', 'fontsize', 10)
    
    filename = fullfile(save_dir,[sname,'.png']);
    saveas(gcf,filename,'png')

end