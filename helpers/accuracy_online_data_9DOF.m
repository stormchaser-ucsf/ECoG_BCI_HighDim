function [acc,acc1,bino_pdf] = accuracy_online_data_9DOF(files)


if nargin<2
    num_targets=9;
end

acc=zeros(num_targets);
acc1=zeros(num_targets);
bino_pdf={};
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
        for ii=1:num_targets
            decodes(ii) = sum(out==ii);
        end
        [aa bb]=max(decodes);

        if sum(aa==decodes)==1
            acc(tid,bb) = acc(tid,bb)+1; % trial level
        else
            disp(['error trial ' num2str(i)])
            % get the actions that have same number of max decodes
            idx=find(decodes==aa);
            prob_sum=[];
            prob_val = TrialData.ClickerDistance;
            for j=1:length(idx)
                xx =  (idx(j) == out);
                prob_sum(j) = sum(prob_val(xx));
            end

            [~, bb]=max(prob_sum);
            bb = idx(bb);
            acc(tid,bb) = acc(tid,bb)+1; % trial level
        end
        
        for j=1:length(out)
            if out(j)>0
                acc1(tid,out(j)) = acc1(tid,out(j))+1; % bin level
            end
        end
    end
end
bino_pdf.n = length(files);
bino_pdf.succ = sum(diag(acc));
bino_pdf.pval = binopdf(sum(diag(acc)),length(files),(1/num_targets));

for i=1:num_targets
    acc(i,:) = acc(i,:)/sum(acc(i,:));
    acc1(i,:) = acc1(i,:)/sum(acc1(i,:));
end



% 
% acc=zeros(9);
% for i=1:length(files)
%     file_loaded=1;
%     try
%         load(files{i});
%     catch
%         file_loaded=0;
%     end
% 
%     if file_loaded
%         out = TrialData.ClickerState;
%         tid = TrialData.TargetID;
%         decodes=[];
%         for ii=1:9
%             decodes(ii) = sum(out==ii);
%         end
%         [aa bb]=max(decodes);
%         if sum(decodes==aa)==1
%             acc(tid,bb) = acc(tid,bb)+1;
%         else
%             idx = find(decodes~=0);
%             prob_thresh=[];
%             for j=1:length(idx)                 
%                  prob_thresh(j) = sum(TrialData.ClickerDistance(out == idx(j)));
%             end
%             [aa bb]=max(prob_thresh);
%             bb=idx(bb);
%             acc(tid,bb) = acc(tid,bb)+1;            
%         end
%         
%     end
% end
% 
% for i=1:9
%     acc(i,:) = acc(i,:)/sum(acc(i,:));
% end

end

