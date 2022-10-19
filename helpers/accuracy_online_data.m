function [acc] = accuracy_online_data(files)

acc=zeros(7);
for i=1:length(files)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end

    if file_loaded
        out = TrialData.FilteredClickerState;
        tid = TrialData.TargetID;
        decodes=[];
        for ii=1:7
            decodes(ii) = sum(out==ii);
        end
        [aa bb]=max(decodes);
        acc(tid,bb) = acc(tid,bb)+1;
    end
end

for i=1:7
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

end

