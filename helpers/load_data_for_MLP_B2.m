function [condn_data] = load_data_for_MLP_B2(files,ecog_grid)
%function [condn_data] = load_data_for_MLP(files)


D1=[];
D2=[];
D3=[];
D4=[];
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
        
        % get the pooled data
        new_temp=[];
        TrialData.Params.ChMap = ecog_grid;
        [xx yy] = size(TrialData.Params.ChMap);
        for k=1:size(temp,2)
            tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
            tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
            tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
            pooled_data=[];
            for i=1:2:xx
                for j=1:2:yy
                    delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                    beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                    hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                    pooled_data = [pooled_data; delta; beta ;hg];
                end
            end
            new_temp= [new_temp pooled_data];
        end
        temp=new_temp;
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
        end
        %end
    end
end


clear condn_data
idx = [1:96];
condn_data{1}=[D1(idx,:) ]'; % right thumb
condn_data{2}= [D2(idx,:)]'; % left leg
condn_data{3}=[D3(idx,:)]'; % left thumb
condn_data{4}=[D4(idx,:)]'; % head

% 2norm
for i=1:length(condn_data)
    tmp = condn_data{i};
    for j=1:size(tmp,1)
        tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
    end
    condn_data{i}=tmp;
end

end




