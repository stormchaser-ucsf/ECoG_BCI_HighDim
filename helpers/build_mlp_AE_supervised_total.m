function [net,Xtrain,Ytrain] = build_mlp_AE_supervised_total(condn_data)
%function [net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data)



idx = [1:96];
for i=1:length(condn_data)
    tmp = condn_data{i};
    tmp=tmp(:,1:96);
    condn_data{i}=tmp;
end


%2norm and get the labels
Y=[];
for i=1:length(condn_data)
    tmp = condn_data{i};
    for j=1:size(tmp,1)
        tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
    end
    condn_data{i}=tmp;
    Y=[Y;i*ones(size(tmp,1),1)];
end

% % data augmentation, add random offsets
% for i=1:length(condn_data)
%    tmp0 = condn_data{i};
%    noise = randn(size(tmp0))*(std(tmp0(:))/4);
%    tmp=tmp0+noise;
%
%     for j=1:size(tmp,1)
%        tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
%     end
%    condn_data{i}=[tmp0;tmp];
%    Y=[Y;i*ones(size(tmp,1),1)];
% end



A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};

clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];
T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;


% using custom layers
layers = [ ...
    featureInputLayer(96)
    fullyConnectedLayer(32)
    eluLayer
    batchNormalizationLayer
    fullyConnectedLayer(8)
    eluLayer
    batchNormalizationLayer
    fullyConnectedLayer(3)
    eluLayer('Name','autoencoder')
    fullyConnectedLayer(8)
    eluLayer
    batchNormalizationLayer
    fullyConnectedLayer(32)
    eluLayer
    batchNormalizationLayer
    fullyConnectedLayer(96)    
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
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
Y=categorical(Y);
idx = randperm(length(Y),round(0.8*length(Y)));
Xtrain = X(:,idx);
Ytrain = Y(idx);
I = ones(length(Y),1);
I(idx)=0;
idx1 = find(I~=0);
Xtest = X(:,idx1);
Ytest = Y(idx1);


%'ValidationData',{XTest,YTest},...
batch_size=64;
val_freq = floor(size(Xtrain,2)/batch_size);
options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',batch_size,...
    'ValidationFrequency',val_freq,...
    'ExecutionEnvironment','gpu',...
    'ValidationPatience',6,...
    'LearnRateDropFactor',0.1,...
    'OutputNetwork','best-validation-loss',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',20,...
    'ValidationData',{Xtest',Ytest});
%'InitialLearnRate',0.01, ...

% build the classifier
clear net
net = trainNetwork(Xtrain',Ytrain,layers,options);


end