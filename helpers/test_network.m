function [cv_perf] = test_network(net,condn_data_overall,test_idx)


XTest=[];
YTest=[];
for i=1:length(test_idx)
    tmp=condn_data_overall(test_idx(i)).neural;
    XTest = [XTest;tmp'];
    tmp1 = condn_data_overall(test_idx(i)).targetID;
    YTest = [YTest;repmat(tmp1,size(tmp,2),1)];
end
YTest=categorical((YTest));

out=predict(net,XTest);
decodes=[];
for i=1:size(out,1)
    [aa bb]=max(out(i,:));
    decodes=[decodes;bb];
end
decodes=categorical(decodes);
cv_perf = sum(decodes==YTest)/length(decodes);
end