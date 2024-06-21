function [acc,acc1,bino_pdf] = get_accuracy_online_anotherDecoder(condn_data,decoder)


num_targets=7;


acc=zeros(num_targets);
acc1=zeros(num_targets);
bino_pdf={};

for i=1:length(condn_data)

    neural = condn_data(i).neural;
    out = predict(decoder,neural')';
    [prob_thresh,decodes] = max(out);
    idx=find(prob_thresh<=0.45);
    decodes(idx)=0;
    prob_thresh(idx)=0;
    out=decodes;
    decodes=decodes(decodes>0);
    prob_thresh=prob_thresh(prob_thresh>0);
    [aa bb]=mode(decodes);
    tid = condn_data(i).targetID;

    %if sum(aa==decodes)==1
    if length(decodes)>0
        acc(tid,aa) = acc(tid,aa)+1; % trial level
    end
    %     else
    %         disp(['error trial ' num2str(i)])
    %         % get the actions that have same number of max decodes
    %         idx=find(decodes==aa);
    %         prob_sum=[];
    %         prob_val = prob_thresh;
    %         for j=1:length(idx)
    %             xx =  (idx(j) == out);
    %             prob_sum(j) = sum(prob_val(xx));
    %         end
    %
    %         [~, bb]=max(prob_sum);
    %         bb = idx(bb);
    %         acc(tid,bb) = acc(tid,bb)+1; % trial level
    % end

    for j=1:length(out)
        if out(j)>0
            acc1(tid,out(j)) = acc1(tid,out(j))+1; % bin level
        end
    end

end
bino_pdf.n = length(condn_data);
bino_pdf.succ = sum(diag(acc));
bino_pdf.pval = binopdf(sum(diag(acc)),length(files),(1/num_targets));

for i=1:num_targets
    acc(i,:) = acc(i,:)/sum(acc(i,:));
    acc1(i,:) = acc1(i,:)/sum(acc1(i,:));
end

end

