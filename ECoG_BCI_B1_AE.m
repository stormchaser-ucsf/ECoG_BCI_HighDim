% autoencoder analysis B1


clc;clear
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

load 7DoF_96Dim_Data_Training_Days


% looping data across days
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];

for i=1:length(online_days)
    D1t = online_days(i).Days.D1;
    D2t = online_days(i).Days.D2;
    D3t = online_days(i).Days.D3;
    D4t = online_days(i).Days.D4;
    D5t = online_days(i).Days.D5;
    D6t = online_days(i).Days.D6;
    D7t = online_days(i).Days.D7;
    
    D = [D1t D2t D3t D4t D5t D6t D7t];
    m = mean(D,2);
    s = std(D',1)';
    D = (D-m)./s;
    
    sz = [size(D1t,2) size(D2t,2) size(D3t,2) size(D4t,2) size(D5t,2)...
        size(D6t,2) size(D7t,2)];
    sz = [0 cumsum(sz)];
    
    D1t = D(:,sz(1)+1:sz(2));
    D2t = D(:,sz(2)+1:sz(3));
    D3t = D(:,sz(3)+1:sz(4));
    D4t = D(:,sz(4)+1:sz(5));
    D5t = D(:,sz(5)+1:sz(6));
    D6t = D(:,sz(6)+1:sz(7));
    D7t = D(:,sz(7)+1:sz(8));
    
    %store
    D1 = [D1 D1t];
    D2 = [D2 D2t];
    D3 = [D3 D3t];
    D4 = [D4 D4t];
    D5 = [D5 D5t];
    D6 = [D6 D6t];
    D7 = [D7 D7t];
    
end
% 
% 
% D1= online_days(7).Days.D1;
% D2= online_days(7).Days.D2;
% D3= online_days(7).Days.D3;
% D4= online_days(7).Days.D4;
% D5= online_days(7).Days.D5;
% D6= online_days(7).Days.D6;
% D7= online_days(7).Days.D7;


%XTrain = cat(2,D1,D2,D3,D4,D5,D6,D7);
XTrain = cat(2,D1,D2,D6);
XTrain = XTrain(65:end,:);
sz = [size(D1,2) size(D2,2) size(D3,2) size(D4,2) size(D5,2)  size(D6,2) size(D7,2)];
sz=[0 cumsum(sz)];

% pass the data onto a convolutional autoencoder format
layers = [featureInputLayer([32])
          fullyConnectedLayer(16)
          sigmoidLayer
          batchNormalizationLayer
          fullyConnectedLayer(8)
          sigmoidLayer
          
%           fullyConnectedLayer(4)
%           reluLayer          
          batchNormalizationLayer
          fullyConnectedLayer(3,'name','autoencoder')
          sigmoidLayer
%           layerNormalizationLayer
%           fullyConnectedLayer(4)          
%           reluLayer          
          batchNormalizationLayer
          fullyConnectedLayer(8)          
          sigmoidLayer        
          batchNormalizationLayer
          fullyConnectedLayer(16)
          sigmoidLayer
          fullyConnectedLayer(32)          
          regressionLayer];
% 
options = trainingOptions('adam', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',15, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','gpu',...
    'Verbose',true);
% 
net = trainNetwork(XTrain',XTrain',layers,options);
Z = activations(net,XTrain','autoencoder');

figure;plot3(Z(1,:),Z(2,:),Z(3,:),'.')
% 
 idx = [ones(size(D1,2),1);2*ones(size(D2,2),1);3*ones(size(D3,2),1);...
     4*ones(size(D4,2),1);5*ones(size(D5,2),1);6*ones(size(D6,2),1);...
     7*ones(size(D7,2),1)];


%idx = [ones(size(D1,2),1);2*ones(size(D7,2),1)];


figure;hold on

cmap=turbo(7);
%cmap=[1 0 0;0 0 1];
for i=1:length(idx)
    c = cmap(idx(i),:);
    plot3(Z(1,i),Z(2,i),Z(3,i),'.','MarkerSize',10,'Color',c);
    %plot(Z(1,i),Z(2,i),'.','MarkerSize',10,'Color',c);
end


figure;plot3(Z(1,1:4844),Z(2,1:4844),Z(3,1:4844),'.b')
hold on
plot3(Z(1,4845:11454),Z(2,4845:11454),Z(3,4845:11454),'.r')
plot3(Z(1,11455:end),Z(2,11455:end),Z(3,11455:end),'.k')

%% TRAINING AN CONVOUTIONAL AUTOENCODER TO GET THE CLASS SPECIFIC SPATIAL FILTERS


clc;clear
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

load 7DoF_Data_Training_Days


% looping data across days
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];

for i=1:length(online_days)
    disp(i)
    D1t = online_days(i).Days.D1;
    D2t = online_days(i).Days.D2;
    D3t = online_days(i).Days.D3;
    D4t = online_days(i).Days.D4;
    D5t = online_days(i).Days.D5;
    D6t = online_days(i).Days.D6;
    D7t = online_days(i).Days.D7;
    
%     % standardize
%     for j=1:3 % over the three frequency bands
%         tmp1 = squeeze(D1t(:,:,:,j));tmp1 = tmp1(:,:);
%         tmp2 = squeeze(D2t(:,:,:,j));tmp2 = tmp2(:,:);
%         tmp3 = squeeze(D3t(:,:,:,j));tmp3 = tmp3(:,:);
%         tmp4 = squeeze(D4t(:,:,:,j));tmp4 = tmp4(:,:);
%         tmp5 = squeeze(D5t(:,:,:,j));tmp5 = tmp5(:,:);
%         tmp6 = squeeze(D6t(:,:,:,j));tmp6 = tmp6(:,:);
%         tmp7 = squeeze(D7t(:,:,:,j));tmp7 = tmp7(:,:);
%         
%         clear D
%         D = [tmp1;tmp2;tmp3;tmp4;tmp5;tmp6;tmp7];
%         m = mean(D);
%         s = std(D,1);
%         D = (D-m)./s;
%         
%         sz = [size(tmp1,1) size(tmp2,1) size(tmp3,1) size(tmp4,1) size(tmp5,1)...
%             size(tmp6,1) size(tmp7,1)];
%         sz = [0 cumsum(sz)];
%         
%         tmp1 = D(sz(1)+1:sz(2),:);tmp1 = reshape(tmp1,size(tmp1,1),8,16);
%         tmp2 = D(sz(2)+1:sz(3),:);tmp2 = reshape(tmp2,size(tmp2,1),8,16);
%         tmp3 = D(sz(3)+1:sz(4),:);tmp3 = reshape(tmp3,size(tmp3,1),8,16);
%         tmp4 = D(sz(4)+1:sz(5),:);tmp4 = reshape(tmp4,size(tmp4,1),8,16);
%         tmp5 = D(sz(5)+1:sz(6),:);tmp5 = reshape(tmp5,size(tmp5,1),8,16);
%         tmp6 = D(sz(6)+1:sz(7),:);tmp6 = reshape(tmp6,size(tmp6,1),8,16);
%         tmp7 = D(sz(7)+1:sz(8),:);tmp7 = reshape(tmp7,size(tmp7,1),8,16);
%         
%         D1t(:,:,:,j) = tmp1;
%         D2t(:,:,:,j) = tmp2;
%         D3t(:,:,:,j) = tmp3;
%         D4t(:,:,:,j) = tmp4;
%         D5t(:,:,:,j) = tmp5;
%         D6t(:,:,:,j) = tmp6;
%         D7t(:,:,:,j) = tmp7;
%     end
   
    %store
    D1 = cat(1,D1,D1t);
    D2 = cat(1,D2,D2t);
    D3 = cat(1,D3,D3t);
    D4 = cat(1,D4,D4t);
    D5 = cat(1,D5,D5t);
    D6 = cat(1,D6,D6t);
    D7 = cat(1,D7,D7t);
    
end
% 

% run it through a class-specific convolutional autoencoder as it is
% already in the format of an image

%%%%%% CNN construction %%%%%
layers = [
    imageInputLayer([8 16 3])
    
    convolution2dLayer(2,4,'padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(2,4,'padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(2,4,'padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(3)
    batchNormalizationLayer
    reluLayer
    %
    transposedConv2dLayer([2,4],4,'Stride',2)
    batchNormalizationLayer
    reluLayer
    
    transposedConv2dLayer([4,8],4,'Stride',2,'Cropping','same')
    batchNormalizationLayer
    reluLayer
    
    transposedConv2dLayer([8, 16],3,'Stride',2,'Cropping','same')
    batchNormalizationLayer
    reluLayer
    
    %transposedConv2dLayer([8, 16],3,'Stride',1,'Cropping','same')
    regressionLayer
    
    ];

%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',64,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...    
    'ExecutionEnvironment','auto');

%'ValidationData',{XTest,YTest},...
%%%%%% CNN construction %%%%%

XTrain = D1;
XTrain = permute(XTrain,[2,3,4,1]);
% build the classifier
net = trainNetwork(XTrain,XTrain,layers,options);
%analyzeNetwork(net)

%%%%%% STACKED AUTOENCODER %%%%%%
% using a stacked autoencoder 
layers = [featureInputLayer([128])
    
    fullyConnectedLayer(64)
    sigmoidLayer
    batchNormalizationLayer
    
    fullyConnectedLayer(16)
    sigmoidLayer    
    batchNormalizationLayer
    
    fullyConnectedLayer(4,'name','autoencoder')
    sigmoidLayer    
    batchNormalizationLayer
    
    fullyConnectedLayer(16)
    sigmoidLayer    
    batchNormalizationLayer
    
    fullyConnectedLayer(64)
    sigmoidLayer
    batchNormalizationLayer
        
    fullyConnectedLayer(128)
    regressionLayer];

analyzeNetwork(layers)

options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',64,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...    
    'ExecutionEnvironment','auto');

XTrain = D4;
XTrain = squeeze(XTrain(:,:,:,3));
XTrain = XTrain(:,:);
%XTrain = zscore(XTrain);
% build the classifier
net = trainNetwork(XTrain,XTrain,layers,options);

figure;
XTrain = D1;
XTrain = squeeze(XTrain(:,:,:,3));
XTrain = XTrain(:,:);
%XTrain = zscore(XTrain);
out = predict(net,XTrain);
out = reshape(out,size(out,1),8,16);
subplot(4,2,1);imagesc(squeeze(mean(out,1)))
caxis([-0.6 2])

% class specific filters
XTrain = D2;
XTrain = squeeze(XTrain(:,:,:,3));
XTrain = XTrain(:,:);
%XTrain = zscore(XTrain);
out = predict(net,XTrain);
out = reshape(out,size(out,1),8,16);
subplot(4,2,2);imagesc(squeeze(mean(out,1)))
caxis([-0.6 2])

% class specific filters
XTrain = D3;
XTrain = squeeze(XTrain(:,:,:,3));
XTrain = XTrain(:,:);
%XTrain = zscore(XTrain);
out = predict(net,XTrain);
out = reshape(out,size(out,1),8,16);
subplot(4,2,3);imagesc(squeeze(mean(out,1)))
caxis([-0.6 2])


% class specific filters
XTrain = D4;
XTrain = squeeze(XTrain(:,:,:,3));
XTrain = XTrain(:,:);
%XTrain = zscore(XTrain);
out = predict(net,XTrain);
out = reshape(out,size(out,1),8,16);
subplot(4,2,4);imagesc(squeeze(mean(out,1)))
caxis([-0.6 2])

% class specific filters
XTrain = D5;
XTrain = squeeze(XTrain(:,:,:,3));
XTrain = XTrain(:,:);
%XTrain = zscore(XTrain);
out = predict(net,XTrain);
out = reshape(out,size(out,1),8,16);
subplot(4,2,5);imagesc(squeeze(mean(out,1)))
caxis([-0.6 2])

% class specific filters
XTrain = D6;
XTrain = squeeze(XTrain(:,:,:,3));
XTrain = XTrain(:,:);
%XTrain = zscore(XTrain);
out = predict(net,XTrain);
out = reshape(out,size(out,1),8,16);
subplot(4,2,6);imagesc(squeeze(mean(out,1)))
caxis([-0.6 2])

% class specific filters
XTrain = D7;
XTrain = squeeze(XTrain(:,:,:,3));
XTrain = XTrain(:,:);
%XTrain = zscore(XTrain);
out = predict(net,XTrain);
out = reshape(out,size(out,1),8,16);
subplot(4,2,7);imagesc(squeeze(mean(out,1)))
caxis([-0.6 2])

