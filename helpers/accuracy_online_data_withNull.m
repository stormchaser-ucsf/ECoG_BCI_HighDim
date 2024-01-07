function [acc,acc1] = accuracy_online_data_withNull(files)

acc=zeros(7);
acc1=zeros(7,8);
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
        for j=1:length(out1)
            if out1(j)==0
                acc1(tid,8) = acc1(tid,8)+1; % bin level, null decode
            else
                acc1(tid,out1(j)) = acc1(tid,out1(j))+1; % bin level
            end
        end
    end
end

for i=1:7
    acc(i,:) = acc(i,:)/sum(acc(i,:));
    acc1(i,1:end) = acc1(i,1:end)/sum(acc1(i,1:end));
end

end

