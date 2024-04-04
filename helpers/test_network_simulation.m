function [cv_perf,conf_matrix] = test_network_simulation(net,condn_data_overall,test_idx,num_classes)


if nargin<4
    num_classes=7;
end

XTest=[];
YTest=[];
for i=1:length(test_idx)
    tmp=condn_data_overall(test_idx(i)).neural;
    XTest = [XTest;tmp];
    tmp1 = condn_data_overall(test_idx(i)).targetID;
    YTest = [YTest;tmp1];
end
YTest=categorical((YTest));

out=predict(net,XTest);
decodes=[];
conf_matrix=zeros(num_classes);
for i=1:size(out,1)
    [aa bb]=max(out(i,:));
    decodes=[decodes;bb];
    conf_matrix(double(YTest(i)),bb)=conf_matrix(double(YTest(i)),bb)+1;
end
decodes=categorical(decodes);
cv_perf = sum(decodes==YTest)/length(decodes);
for i=1:size(conf_matrix,2)
    conf_matrix(i,:) = conf_matrix(i,:)./sum(conf_matrix(i,:));
end

end