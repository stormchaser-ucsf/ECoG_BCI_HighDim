function [acc,acc_bin,trial_len] = accuracy_online_data_Hand(files,targets)

acc=zeros(targets);
acc_bin=zeros(targets);
trial_len=[];
for i=1:length(files)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end

    if file_loaded
        out = TrialData.ClickerState;
        tid = TrialData.TargetID;
        decodes=[];
        for ii=1:targets
            decodes(ii) = sum(out==ii);
        end

        %bin level
        decodes1 = TrialData.ClickerState;
        for j=1:length(decodes1)
            if decodes1(j)~=0
                acc_bin(tid,decodes1(j)) = acc_bin(tid,decodes1(j))+1;
            end
        end

        % trial length
        trial_len = [trial_len length(out)];

        % trial level
        [aa bb]=max(decodes);
        if sum(decodes==aa)==1
            acc(tid,bb) = acc(tid,bb)+1;
        else
            idx = find(decodes~=0);
            prob_thresh=[];
            for j=1:length(idx)
                prob_thresh(j) = sum(TrialData.ClickerDistance(out == idx(j)));
            end
            [aa bb]=max(prob_thresh);
            bb=idx(bb);
            acc(tid,bb) = acc(tid,bb)+1;
        end

    end
end

for i=1:targets
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

for i=1:targets
    acc_bin(i,:) = acc_bin(i,:)/sum(acc_bin(i,:));
end

end

