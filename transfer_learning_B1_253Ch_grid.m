
clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3
addpath 'C:\Users\nikic\Documents\MATLAB'

%% load B3's Data from 11 days of daily 7DoF initialization
load condn_data_overall_B3

%% build a PnP decoder from B3
% split into training and testing trials, 20% val, 80% train
xx=1;xx1=1;yy=0;
prop = 1;
while xx<7 || xx1<7
    I = ones(length(condn_data_overall),1);
    train_val_idx = find(I~=0);
    tmp_idx = randperm(length(train_val_idx),round(prop*length(train_val_idx)));
    train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
    I([train_idx])=0;
    val_idx = find(I~=0);val_idx=val_idx(:);
    xx = length(unique([condn_data_overall(train_idx).targetID]));
    xx1 = length(unique([condn_data_overall(val_idx).targetID]));
    yy=yy+1;
end

% training options for NN
[options,XTrain,YTrain] = ...
    get_options(condn_data_overall,val_idx,train_idx);

% build MLP
aa=condn_data_overall(1).neural;
s=size(aa,1);
layers = get_layers1(120,s);
net_B3_PnP_trfLearn = trainNetwork(XTrain,YTrain,layers,options);

% save the weights
save net_B3_PnP_trfLearn net_B3_PnP_trfLearn -v7.3

%% load B1's data from daily 7DoF initialization with new 253 grid

cd 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker'
root_path=pwd;
foldernames={'20240614', '20240517', '20240515', '20240508','20240619'};
files=[];file_type=[];
for i=1:length(foldernames)
    fullpath = fullfile(root_path,foldernames{i},'Robot3DArrow');
    tmp_files = findfiles('.mat',fullpath,1)';
    tmp_files1=[];k=1;
    for j=1:length(tmp_files)
        if length(regexp(tmp_files{j},'kf_params'))==0
            tmp_files1 = [tmp_files1;tmp_files(j)];
        end
    end
    files=[files;tmp_files1];
end

% load the data
condn_data = load_data_for_MLP_TrialLevel_B3(files);

condn_data_overall = condn_data;

% prune to only the first 7 actions
condn_data_overall1={};kk=1;
for ii=1:length(condn_data_overall)
    if ~isempty(condn_data_overall(ii).neural) && condn_data_overall(ii).targetID <=7
        condn_data_overall1(kk).neural = condn_data_overall(ii).neural;
        condn_data_overall1(kk).targetID = condn_data_overall(ii).targetID;
        condn_data_overall1(kk).trial_type = condn_data_overall(ii).trial_type;
        kk=kk+1;
    end
end
condn_data_overall = condn_data_overall1;


%% Fine tune the weights of B3 PnP decoder with B1's data

xx=1;xx1=1;yy=0;
prop = 0.8;
while xx<7 || xx1<7
    I = ones(length(condn_data_overall),1);
    train_val_idx = find(I~=0);
    tmp_idx = randperm(length(train_val_idx),round(prop*length(train_val_idx)));
    train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
    I([train_idx])=0;
    val_idx = find(I~=0);val_idx=val_idx(:);
    xx = length(unique([condn_data_overall(train_idx).targetID]));
    xx1 = length(unique([condn_data_overall(val_idx).targetID]));
    yy=yy+1;
end

% training options for NN
[options,XTrain,YTrain] = ...
    get_options(condn_data_overall,val_idx,train_idx);

% data augmentation of the training data
condn_data_overall_data_aug=condn_data_overall(train_idx);
for i=1:7
    idx = find([condn_data_overall_data_aug(1:end).targetID]==i);
    open_idx=[];closed_idx=[];
    for j=1:length(idx)
        if size(condn_data_overall_data_aug(idx(j)).neural,2) >20
            open_idx=[open_idx;idx(j)];
        else
            closed_idx=[closed_idx;idx(j)];
        end
    end

    %%%%% open loop data
    tmp_aug=[];     len=[];
    for j=1:length(open_idx)
        tmp = condn_data_overall_data_aug(open_idx(j)).neural;
        len = [len size(tmp,2)];
        tmp_aug = [tmp_aug tmp];
    end

    % augmentation
    total_aug=[];
    for j=1:253:759
        tmp_feat  = tmp_aug(j:j+253-1,:)';
        C = cov(tmp_feat);
        if rank(C)<253
            C = C + 1e-6*eye(size(C));
        end
        C12 = chol(C);
        m = mean(tmp_feat,1);
        x = randn(size(tmp_feat,1)*2,size(C,1));
        aug_feat = x*C12+m;
        mm = 0.02*randn(size(aug_feat,1),1);
        mm = repmat(mm,1,253);
        aug_feat = aug_feat+mm;
        total_aug = [total_aug aug_feat];
    end
    %total_aug = total_aug + 0.02*randn(size(total_aug));
    XTrain = [XTrain; total_aug];
    YTrain = [YTrain; categorical(i*ones(size(total_aug,1),1))];

    %%%% closed loop data
    tmp_aug=[];    len=[];
    for j=1:length(closed_idx)
        tmp = condn_data_overall_data_aug(closed_idx(j)).neural;
        len = [len size(tmp,2)];
        tmp_aug = [tmp_aug tmp];
    end

    % augmentation
    total_aug=[];
    for j=1:253:759
        tmp_feat  = tmp_aug(j:j+253-1,:)';
        C = cov(tmp_feat);
        if rank(C)<253
            C = C + 1e-6*eye(size(C));
        end
        C12 = chol(C);
        m = mean(tmp_feat,1);
        x = randn(size(tmp_feat,1)*2,size(C,1));
        aug_feat = x*C12+m;
        mm = 0.02*randn(size(aug_feat,1),1);
        mm = repmat(mm,1,253);
        aug_feat = aug_feat+mm;
        total_aug = [total_aug aug_feat];
    end
    %total_aug = total_aug + 0.02*randn(size(total_aug));
    XTrain = [XTrain; total_aug];
    YTrain = [YTrain; categorical(i*ones(size(total_aug,1),1))];
end


% build MLP
%aa=condn_data_overall(1).neural;
%s=size(aa,1);
%layers = get_layers1(120,s);
layers = net_B3_PnP_trfLearn.Layers;
net_B1_253_TrfLearn = trainNetwork(XTrain,YTrain,layers,options);

% save the weights
save net_B1_253_TrfLearn net_B1_253_TrfLearn -v7.3


%% USE PATTERNET FUNCTIONS TO BUILD PNP DECODER FOR B1 WITH NEW GRID USING
%% TRANSFER LEARNING FROM B3

%%

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3
addpath 'C:\Users\nikic\Documents\MATLAB'

%% load B3's Data from 11 days of daily 7DoF initialization
load condn_data_overall_B3

%% build a PnP decoder from B3
% split into training and testing trials, 20% val, 80% train



xx=1;xx1=1;yy=0;
prop = 0.80;
while xx<7 || xx1<7
    I = ones(length(condn_data_overall),1);
    train_val_idx = find(I~=0);
    tmp_idx = randperm(length(train_val_idx),round(prop*length(train_val_idx)));
    train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
    I([train_idx])=0;
    val_idx = find(I~=0);val_idx=val_idx(:);
    xx = length(unique([condn_data_overall(train_idx).targetID]));
    xx1 = length(unique([condn_data_overall(val_idx).targetID]));
    yy=yy+1;
end

% get the training data
[N,T,condn_data] = get_training_samples_mlp(condn_data_overall,train_idx);
[N1,T1,condn_data1] = get_training_samples_mlp(condn_data_overall,val_idx);
train_idx = 1:size(N,2);
val_idx = (train_idx(end)+1):(size(N,2) + size(N1,2));
N = [N N1];
T = [T; T1];

% train 
net=patternnet(120);
net.divideFcn = 'divideind'; % to make into training and testing indices
net.divideParam.trainInd=train_idx;
net.divideParam.valInd=val_idx;
net.divideParam.testInd=[];
net.performParam.regularization=0.2;
net_B3_PnP_trfLearn_patternet = train(net,N,T','UseGPU','yes');
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
genFunction(net_B3_PnP_trfLearn_patternet,'net_B3_PnP_trfLearn_patternet')

% save the weights
save net_B3_PnP_trfLearn_patternet net_B3_PnP_trfLearn_patternet -v7.3

%% load B1's data from daily 7DoF initialization with new 253 grid

clear
cd 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker'
root_path=pwd;
foldernames={'20240515','20240517','20240614','20240619','20240621','20240626','20240710',...
    '20240712'};
files=[];file_type=[];
for i=1:length(foldernames)
    fullpath = fullfile(root_path,foldernames{i},'Robot3DArrow');
    tmp_files = findfiles('.mat',fullpath,1)';
    tmp_files1=[];k=1;
    for j=1:length(tmp_files)
        if length(regexp(tmp_files{j},'kf_params'))==0
            tmp_files1 = [tmp_files1;tmp_files(j)];
        end
    end
    files=[files;tmp_files1];
end

% load the data
condn_data = load_data_for_MLP_TrialLevel_B3(files);

condn_data_overall = condn_data;

% prune to only the first 7 actions
condn_data_overall1={};kk=1;
for ii=1:length(condn_data_overall)
    if ~isempty(condn_data_overall(ii).neural) && condn_data_overall(ii).targetID <=7
        condn_data_overall1(kk).neural = condn_data_overall(ii).neural;
        condn_data_overall1(kk).targetID = condn_data_overall(ii).targetID;
        condn_data_overall1(kk).trial_type = condn_data_overall(ii).trial_type;
        kk=kk+1;
    end
end
condn_data_overall = condn_data_overall1;


%% Fine tune the weights of B3 PnP decoder with B1's data

xx=1;xx1=1;yy=0;
prop = 0.80;
while xx<7 || xx1<7
    I = ones(length(condn_data_overall),1);
    train_val_idx = find(I~=0);
    tmp_idx = randperm(length(train_val_idx),round(prop*length(train_val_idx)));
    train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
    I([train_idx])=0;
    val_idx = find(I~=0);val_idx=val_idx(:);
    xx = length(unique([condn_data_overall(train_idx).targetID]));
    xx1 = length(unique([condn_data_overall(val_idx).targetID]));
    yy=yy+1;
end

% get the training data
[N,T,condn_data] = get_training_samples_mlp(condn_data_overall,train_idx);
[N1,T1,condn_data1] = get_training_samples_mlp(condn_data_overall,val_idx);
train_idx = 1:size(N,2);
val_idx = (train_idx(end)+1):(size(N,2) + size(N1,2));
N = [N N1];
T = [T; T1];

load net_B3_PnP_trfLearn_patternet


% train 
%net=patternnet(120);
%net_B3_PnP_trfLearn_patternet.divideFcn = 'divideind'; % to make into training and testing indices
net_B3_PnP_trfLearn_patternet.divideParam.trainInd=train_idx;
net_B3_PnP_trfLearn_patternet.divideParam.valInd=val_idx;
net_B3_PnP_trfLearn_patternet.divideParam.testInd=[];
%net_B3_PnP_trfLearn_patternet.performParam.regularization=0.2;
%net_B3_PnP_trfLearn_patternet = train(net,N,T','UseGPU','yes');
net_B3B1_PnP_trfLearn_patternet = train(net_B3_PnP_trfLearn_patternet,N,T','UseGPU','yes');


cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
genFunction(net_B3B1_PnP_trfLearn_patternet,'net_B3B1_PnP_trfLearn_patternet')

% save the weights
save net_B3B1_PnP_trfLearn_patternet net_B3B1_PnP_trfLearn_patternet -v7.3

