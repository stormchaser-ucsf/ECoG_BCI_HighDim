function [trial_data] = load_data_for_MLP_TrialLevel(files,trial_type,pooling)
%function [condn_data] = load_data_for_MLP(files)


trial_data=[];
kk=1;
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
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        temp = cell2mat(features(kinax));

        if pooling

            % get the pooled data
            new_temp=[];
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

            % get unpooled data all features
            %temp=temp(129:end,:);

        else

            % get unpooled data main 4 features
            temp=temp([129:256 513:640 769:896],:);

        end

        for i=1:size(temp,2)
            temp(:,i) = temp(:,i)./norm(temp(:,i));
        end

        % get decoding accuracy of online
        if trial_type>0
            decodes=TrialData.ClickerState;
        end


        trial_data(kk).neural = temp;
        trial_data(kk).targetID = TrialData.TargetID;
        trial_data(kk).trial_type = trial_type;
        if trial_type>0
            trial_data(kk).decodes = decodes;
        end        
        kk=kk+1;
    end
end

end




