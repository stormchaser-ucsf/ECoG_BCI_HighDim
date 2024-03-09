function [D1,D2,D3,D4] = load_erp_data_B2_imag(files)


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
for i=1:length(files)
    load(files{i});
    disp([i ])
    features  = TrialData.SmoothedNeuralFeatures;
    features = cell2mat(features);
    features = features(769:end,:);
    fs = TrialData.Params.UpdateRate;
    data = features;

    if size(data,2) < 47
        l = 47 - size(data,2);
        data(:,end+1:47) = repmat(data(:,end),1,l);
    end

    % z-score to first few bins
    m=mean(data(:,1:8),2);
    s=std(data(:,1:8)',1)';
    data = ((data'-m')./s')';

    % now get the ERPs

    if TrialData.TargetID == 1
        D1 = cat(3,D1,data);
    elseif TrialData.TargetID == 2
        D2 = cat(3,D2,data);
    elseif TrialData.TargetID == 3
        D3 = cat(3,D3,data);
    elseif TrialData.TargetID == 4
        D4 = cat(3,D4,data);    
    end

end

end