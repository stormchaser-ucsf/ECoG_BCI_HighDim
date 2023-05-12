function [acc] = accuracy_online_data_9DOF(files)

acc=zeros(9);
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
        for ii=1:9
            decodes(ii) = sum(out==ii);
        end
        [aa bb]=max(decodes);
        acc(tid,bb) = acc(tid,bb)+1;
    end
end

for i=1:9
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

end

