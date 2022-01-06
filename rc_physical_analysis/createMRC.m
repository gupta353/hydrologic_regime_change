%{
This function creates Master Recession Curve (MRC)
inputs: filt_rec_periods = a cell-array such that each cell contains a recession period that have been filtered through some conditions
outputs: mrcf = MRC
%}
function  mrcf = createMRC(filt_rec_periods)
    
    % sort the filtered recession periods
    figure; hold on
    for rec_ind=1:length(filt_rec_periods)

        comp_val(rec_ind,1)=min(filt_rec_periods{rec_ind}(:,4));
        plot(filt_rec_periods{rec_ind}(:,4));

    end
    hold off;
    xlabel('Arbitrary time-steps (days)','fontname','arial','fontsize',12);
    ylabel('Streamflow (mm day^{-1})','fontname','arial','fontsize',12);
    title('Filtered recession periods','fontname','arial','fontsize',12);
    set(gca,'fontname','arial','fontsize',12);

    comp_val=[comp_val,(1:length(comp_val))'];
    comp_val=sortrows(comp_val);

    % create MRC
    mrcf=[];
    for mrcf_ind=2:size(comp_val,1)

        % pick the lower and upper recession curve
        low_rec_ind=comp_val(mrcf_ind-1,2);
        up_rec_ind=comp_val(mrcf_ind,2);
        lower_rec=sort(filt_rec_periods{low_rec_ind}(:,4),'descend');
        upper_rec=sort(filt_rec_periods{up_rec_ind}(:,4),'descend');

        min_upper_rec=min(upper_rec);
        differ=abs(lower_rec-min_upper_rec);
        ind=find(differ==min(differ));
        mrcf=[lower_rec(ind:end);mrcf];

    end
    mrcf=[upper_rec;mrcf];

end