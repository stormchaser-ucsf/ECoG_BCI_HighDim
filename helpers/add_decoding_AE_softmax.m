function [net1] =  add_decoding_AE_softmax(net,condn_data)


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


% get the data at the bottlneck layer
X=N;
X=X(1:96,:);
X = activations(net,X','layer_2');


% create a new classification layer
layers = [featureInputLayer(size(X,1))
    net.Layers(6)
    eluLayer
    fullyConnectedLayer(7)
    softmaxLayer('Name','Classif')
    classificationLayer];

% splitting training and testing
Y=categorical(T1);
idx = randperm(length(Y),round(0.8*length(Y)));
Xtrain = X(:,idx);
Ytrain = Y(idx);
I = ones(length(Y),1);
I(idx)=0;
idx1 = find(I~=0);
Xtest = X(:,idx1);
Ytest = Y(idx1);


%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',25, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',64,...
    'ValidationFrequency',16,...
    'ExecutionEnvironment','gpu',...
    'ValidationPatience',5,...
    'ValidationData',{Xtest',Ytest'});


% build the classifier
clear net1
net1 = trainNetwork(Xtrain',Ytrain',layers,options);

% visualizing what the new latent space looks like
Z = activations(net1,X','layer');
[c,s,l]=pca(Z');
idx=T1;
Z=s';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    %if i==1||i==6||i==7||i==4||i==3
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:),'MarkerSize',20);
    %end
end
title('imagined AE+classif.')
%title('Proj. Online Data through Imagined Latent Space')
set(gcf,'Color','w')




% now reupdating the weights of the decoder of the AE to minimize MSE




end