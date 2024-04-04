function [cv_perf,conf_matrix,net] = run_classifer_simulation(condn_data_overall,prop)
%function [cv_perf,conf_matrix,net] = run_classifer_simulation(condn_data_overall,prop)

if nargin==1
    prop=0.15;
end

%split into training and validation sets
num_classes = length(unique([condn_data_overall.targetID]));
test_idx = randperm(length(condn_data_overall),round(prop*length(condn_data_overall)));
test_idx=test_idx(:);
I = ones(length(condn_data_overall),1);
I(test_idx)=0;
train_val_idx = find(I~=0);
prop1 = (0.82/(1-prop));
tmp_idx = randperm(length(train_val_idx),round(prop1*length(train_val_idx)));
train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
I = ones(length(condn_data_overall),1);
I([train_idx;test_idx])=0;
val_idx = find(I~=0);val_idx=val_idx(:);

% build a classifier between the two
%layers = get_layers1_simulation(20,2,4);
layers = get_layers2_simulation(2,30,2,4);

% training options for NN
[options,XTrain,YTrain] = ...
    get_options_simulating(condn_data_overall,val_idx,train_idx);


% design the neural net
aa=condn_data_overall(1).neural;
s=size(aa,2);
layers = get_layers1(64,s,num_classes);

% train the network
net = trainNetwork(XTrain,YTrain,layers,options);

% test the network
if prop>0
    [cv_perf,conf_matrix] = test_network_simulation(net,condn_data_overall,test_idx,num_classes);
else
    cv_perf=[];
    conf_matrix=[];
end