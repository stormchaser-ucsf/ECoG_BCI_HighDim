function [acc,acc1] = accuracy_online_data(files)

acc=zeros(7);
acc1=zeros(7);
for i=1:length(files)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end

    if file_loaded
        out = TrialData.ClickerState;
        out1 = TrialData.FilteredClickerState;
        tid = TrialData.TargetID;
        decodes=[];
        for ii=1:7
            decodes(ii) = sum(out==ii);
        end
        [aa bb]=max(decodes);
        acc(tid,bb) = acc(tid,bb)+1; % trial level
        for j=1:length(out)
            if out(j)>0
                acc1(tid,out(j)) = acc1(tid,out(j))+1; % bin level
            end
        end
    end
end

for i=1:7
    acc(i,:) = acc(i,:)/sum(acc(i,:));
    acc1(i,:) = acc1(i,:)/sum(acc1(i,:));
end

end

