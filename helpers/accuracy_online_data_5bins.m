function [acc,acc1,bino_pdf] = accuracy_online_data_5bins(files,num_targets)

if nargin<2
    num_targets=7;
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
%         if tid==3
%             disp(i)
%         end
%         decodes=[];

        % based on which movement got 5 consecutive bins
        bins_req=TrialData.Params.ClickCounter;
        %bins_req=4;

        % now get the decodes or till time out
        out1=out1(out1>0);
        c = 1;
        st = out1(1);
        decoded_op = 0;
        for ii = 2:length(out1)
            if out1(ii) == st
                c=c+1;
            else
                c=1;
                st=out1(ii);
            end
            if c==bins_req
                decoded_op = st;
                break;
            end
        end

        bb =  decoded_op;
        if bb~=tid
            disp(i)
        end
        if bb>0
            acc(tid,bb) = acc(tid,bb)+1; % trial level
        end

        
%         
%         %old method
%         for ii=1:7
%             decodes(ii) = sum(out==ii);
%         end
%         [aa bb]=max(decodes);
% 
%         if sum(aa==decodes)==1
%             acc(tid,bb) = acc(tid,bb)+1; % trial level
%         else
%             disp(['error trial ' num2str(i)])
%             get the actions that have same number of max decodes
%             idx=find(decodes==aa);
%             prob_sum=[];
%             prob_val = TrialData.ClickerDistance;
%             for j=1:length(idx)
%                 xx =  (idx(j) == out);
%                 prob_sum(j) = sum(prob_val(xx));
%             end
% 
%             [~, bb]=max(prob_sum);
%             bb = idx(bb);
%             acc(tid,bb) = acc(tid,bb)+1; % trial level
%         end
% 
%         for j=1:length(out)
%             if out(j)>0
%                 acc1(tid,out(j)) = acc1(tid,out(j))+1; % bin level
%             end
%         end
%     end
end
bino_pdf.n = length(files);
bino_pdf.succ = sum(diag(acc));
bino_pdf.pval = binopdf(sum(diag(acc)),length(files),(1/num_targets));

for i=1:num_targets
    acc(i,:) = acc(i,:)/sum(acc(i,:));
    acc1(i,:) = acc1(i,:)/sum(acc1(i,:));
end

end

