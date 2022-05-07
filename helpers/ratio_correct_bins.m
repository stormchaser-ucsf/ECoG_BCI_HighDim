function [res] = ratio_correct_bins(files)
%function [res] = ratio_correct_bins(files)


D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
idx=[];
data=[];
for ii=1:length(files)
    %disp(ii)
    file_loaded=1;
    try
        load(files{ii});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        temp = cell2mat(features(kinax));
        clicker_state = TrialData.FilteredClickerState;
        %clicker_state = TrialData.ClickerState;
        correct_bins = sum(clicker_state == TrialData.TargetID);
        acc  = correct_bins/(length(clicker_state)-sum(clicker_state==0));
        temp=acc;
        if TrialData.TargetID == 1
            D1 = [D1 temp];
        elseif TrialData.TargetID == 2
            D2 = [D2 temp];
        elseif TrialData.TargetID == 3
            D3 = [D3 temp];
        elseif TrialData.TargetID == 4
            D4 = [D4 temp];
        elseif TrialData.TargetID == 5
            D5 = [D5 temp];
        elseif TrialData.TargetID == 6
            D6 = [D6 temp];
        elseif TrialData.TargetID == 7
            D7 = [D7 temp];
        end
        data=[data temp];
        idx=[idx TrialData.TargetID];
        %end
    end
end
res=[];
%res.data=[D1 D2 D3 D4 D5 D6 D7];
res.data=data;
res.idx=idx;
end




