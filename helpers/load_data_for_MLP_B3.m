function [condn_data] = load_data_for_MLP_B3(files,ecog_grid)
%function [condn_data] = load_data_for_MLP(files)


D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for ii=1:length(files)
    disp(ii)
    file_loaded=1;
    try
        load(files{ii});
    catch
        file_loaded=0;
    end
    if file_loaded

        features  = TrialData.SmoothedNeuralFeatures;
        temp = cell2mat(features);
        kinax = find(TrialData.TaskState==3);
        temp = cell2mat(features(kinax));

        % get delta, beta and hG removing bad channels
        temp = temp([257:512 1025:1280 1537:1792],:);
        bad_ch = [108 113 118];
        good_ch = ones(size(temp,1),1);
        for iii=1:length(bad_ch)
            bad_ch_tmp = bad_ch(iii)*[1 2 3];
            good_ch(bad_ch_tmp)=0;
        end
        temp = temp(logical(good_ch),:);

        %if TrialData.TargetID  ==  TrialData.SelectedTargetID
        % temp=temp(:,end-5:end);
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
        %end
    end
end


clear condn_data
idx = 1:size(D1,1);
condn_data{1}=[D1(idx,:) ]'; % right thumb
condn_data{2}= [D2(idx,:)]'; % left leg
condn_data{3}=[D3(idx,:)]'; % left thumb
condn_data{4}=[D4(idx,:)]'; % head
condn_data{5}=[D5(idx,:)]'; % lips
condn_data{6}=[D6(idx,:)]'; % tong
condn_data{7}=[D7(idx,:)]'; % BMF

% 2norm
for i=1:length(condn_data)
    tmp = condn_data{i};
    for j=1:size(tmp,1)
        tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
    end
    condn_data{i}=tmp;
end

end




