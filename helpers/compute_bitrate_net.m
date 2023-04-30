function [acc] = compute_bitrate_net(files,num_target,net)


% look at the decodes per direction to get a max vote
acc=zeros(num_target);
for ii=1:length(files)
    %disp(ii)
    indicator=1;
    try
        load(files{ii});
    catch ME
        warning('Not able to load file, skipping to next')
        indicator = 0;
    end
    if indicator
        kinax = TrialData.TaskState;
        kinax = find(kinax==3);
        temp = cell2mat(TrialData.SmoothedNeuralFeatures(kinax));

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

        % 2-norm
        for i=1:size(temp,2)
            temp(:,i) = temp(:,i)./norm(temp(:,i));
        end

        out = net(temp);
        out(out<0.4)=0; % thresholding
        [prob,idx] = max(out); % getting the decodes
        decodes=idx;
        %decodes = mode_filter(idx); % running it through a 5 sample mode filter

        decodes_sum=[];
        for i=1:7
            decodes_sum(i) = sum(decodes==i);
        end
        [aa bb]=max(decodes_sum);

        acc(TrialData.TargetID,bb) = acc(TrialData.TargetID,bb)+1;
    end
end

for i=1:size(acc,1)
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end


end