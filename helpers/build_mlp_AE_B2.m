function [net,Xtrain,Ytrain] = build_mlp_AE_B2(condn_data)
%function [net,Xtrain,YTrain] = build_mlp_AE(condn_data)



idx = [1:96];
for i=1:length(condn_data)
   tmp = condn_data{i};
   tmp=tmp(:,1:96);
   condn_data{i}=tmp;    
end


%2norm
for i=1:length(condn_data)
   tmp = condn_data{i}; 
   for j=1:size(tmp,1)
       tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
   end
   condn_data{i}=tmp;
end

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};

clear N
N = [A' B' C' D'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1)];
T = zeros(size(T1,1),4);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;


% using custom layers
layers = [ ...
    featureInputLayer(96)    
    fullyConnectedLayer(32)
    eluLayer    
    fullyConnectedLayer(8)
    eluLayer    
    fullyConnectedLayer(3)    
    eluLayer('Name','autoencoder')    
    fullyConnectedLayer(8)
    eluLayer
    fullyConnectedLayer(32)
    eluLayer    
    fullyConnectedLayer(96)    
    regressionLayer
    ];


% 
% % using custom layers
% layers = [ ...
%     featureInputLayer(96)    
%     fullyConnectedLayer(32)
%     eluLayer    
%     fullyConnectedLayer(8)
%     eluLayer('Name','autoencoder')        
%     fullyConnectedLayer(32)
%     eluLayer        
%     fullyConnectedLayer(96)
%     regressionLayer
%     ];

X = N;
Y=categorical(T1);
idx = randperm(length(Y),round(0.8*length(Y)));
Xtrain = X(:,idx);
Ytrain = Y(idx);
I = ones(length(Y),1);
I(idx)=0;
idx1 = find(I~=0);
Xtest = X(:,idx1);
Ytest = Y(idx1);

batch_size=64;
val_freq = floor(size(Xtrain,2)/batch_size);
options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',batch_size,...
    'ValidationFrequency',val_freq,...
    'ExecutionEnvironment','gpu',...
    'ValidationPatience',6,...
    'LearnRateDropFactor',0.1,...
    'OutputNetwork','best-validation-loss',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',25,...
    'ValidationData',{Xtest',Xtest'});




% %'ValidationData',{XTest,YTest},...
% options = trainingOptions('adam', ...
%     'InitialLearnRate',0.01, ...
%     'MaxEpochs',25, ...
%     'Verbose',true, ...
%     'Plots','training-progress',...
%     'MiniBatchSize',32,...
%     'ValidationFrequency',8,...
%     'ExecutionEnvironment','gpu',...
%     'ValidationPatience',5,...
%     'ValidationData',{Xtest',Xtest'});


% build the classifier
clear net
net = trainNetwork(Xtrain',Xtrain',layers,options);


end