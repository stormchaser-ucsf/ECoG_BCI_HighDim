function [conf_matrix] = test_network_trialLevel(net,condn_data_overall,test_idx,num_classes)


if nargin<4
    num_classes=7;
end

XTest=[];
YTest=[];
conf_matrix=zeros(num_classes);
for i=1:length(test_idx)
    xtest=condn_data_overall(test_idx(i)).neural;    
    ytest = condn_data_overall(test_idx(i)).targetID;
    out=predict(net,xtest');
    [aa bb]=max(out');
    
    decodes=[];
    for j=1:(num_classes)
        decodes(j) = sum(bb==j);
    end
    [aa bb]=max(decodes);
    if sum(decodes==aa) == 1
        decode = bb;
    elseif sum(decodes==aa) > 1
        disp(['error ' num2str(i) ' trial'])
        xx=mean(out);
        [~, dec2]=max(xx);
        decode = dec2;
    end

    conf_matrix(ytest,decode) = conf_matrix(ytest,decode)+1;    
end

for i=1:size(conf_matrix,2)
    conf_matrix(i,:) = conf_matrix(i,:)./sum(conf_matrix(i,:));
end


end