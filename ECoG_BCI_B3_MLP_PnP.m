% plug and play MLP decoder with B3
% parameters are a single layer and with 120 hidden units


%% build a mlp with all historical data
clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3
addpath 'C:\Users\nikic\Documents\MATLAB'
load('ECOG_Grid_8596_000067_B3.mat')
condn_data={};
for i=1:11%all the original data without covert mime
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);


    %%%%%% load imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end    
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0)];

    %%%%%% load online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,1) ];

    %%%%%% load batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,2) ];
end

% make them all into one giant struct
tmp=cell2mat(condn_data(1));
condn_data_overall=tmp;
for i=2:length(condn_data)
    tmp=cell2mat(condn_data(i));
    for k=1:length(tmp)
        condn_data_overall(end+1) =tmp(k);
    end
end

% split the data into validation and training sets
test_idx = randperm(length(condn_data_overall),round(0.2*length(condn_data_overall)));
test_idx=test_idx(:);
I = ones(length(condn_data_overall),1);
I(test_idx)=0;
train_idx = find(I~=0);train_idx=train_idx(:);

% training options for NN MLP
[options,XTrain,YTrain] = ...
    get_options(condn_data_overall,test_idx,train_idx);
options.Plots='training-progress';

% build a MLP decoder, single hidden layer and 120 units
%layers = get_layers1(120,size(XTrain,2));
%layers = get_layers2(120,64,759);
layers = get_layers(120,64,64,759);

% train the MLP
net = trainNetwork(XTrain,YTrain,layers,options);

% save the network
net_mlp_v0 = net;
save net_mlp_v0 net_mlp_v0

% save data temporarily
condn_data_first11=condn_data;

 layers=[layers(1:5)
           fullyConnectedLayer(6)
           softmaxLayer
          classificationLayer]

%% batch update with more recent data collected on covert miming


condn_data={};
load('ECOG_Grid_8596_000067_B3.mat')
for i=12:19%all the original data without covert mime
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);


    %%%%% load imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end    
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0)];

    %%%%%% load online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,1) ];

    %%%%%% load batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,2) ];
end

% make them all into one giant struct
tmp=cell2mat(condn_data(1));
condn_data_overall=tmp;
for i=2:length(condn_data)
    tmp=cell2mat(condn_data(i));
    for k=1:length(tmp)
        condn_data_overall(end+1) =tmp(k);
    end
end

% split the data into validation and training sets
test_idx = randperm(length(condn_data_overall),round(0.2*length(condn_data_overall)));
test_idx=test_idx(:);
I = ones(length(condn_data_overall),1);
I(test_idx)=0;
train_idx = find(I~=0);train_idx=train_idx(:);

% training options for NN MLP
[options,XTrain,YTrain] = ...
    get_options(condn_data_overall,test_idx,train_idx,7.5e-4);
options.Plots='training-progress';

% batch update the MLP
clear net_mlp_v0
load net_mlp_v0
layers = net_mlp_v0.Layers;
net_mlp_v0 = trainNetwork(XTrain,YTrain,layers,options);

% net_PnP = net_mlp_v0;
% save net_PnP net_PnP

%% test performance on held out days


condn_data={};
load('ECOG_Grid_8596_000067_B3.mat')
for i=20%all the original data without covert mime
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);


%     %%%%%% load imagined data
%     folders = session_data(i).folders(imag_idx);
%     day_date = session_data(i).Day;
%     files=[];
%     for ii=1:length(folders)
%         folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
%         %cd(folderpath)
%         files = [files;findfiles('',folderpath)'];
%     end
%     load('ECOG_Grid_8596_000067_B3.mat')
%     condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0)];

    %%%%%% load online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,1) ];
    [acc_online_trial,acc_online] = accuracy_online_data(files);

    %%%%%% load batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,2) ];
    [acc_batch_trial,acc_batch] = accuracy_online_data(files);
end


% make them all into one giant struct
tmp=cell2mat(condn_data(1));
condn_data_overall=tmp;
for i=2:length(condn_data)
    tmp=cell2mat(condn_data(i));
    for k=1:length(tmp)
        condn_data_overall(end+1) =tmp(k);
    end
end
test_idx=1:length(condn_data_overall);
[cv_perf,conf_matrix] = test_network(net_mlp_v0,condn_data_overall,test_idx);
%[cv_perf,conf_matrix] = test_network(net_PnP,condn_data_overall,test_idx);
aa=nanmean([diag(acc_batch);diag(acc_online)]);

disp([aa cv_perf])

figure;imagesc(conf_matrix)



%% GETTING BIN LEVEL DECODING PERFORMANCE FOR ALL DAYS WITH DAILY MLP


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3
addpath 'C:\Users\nikic\Documents\MATLAB'
condn_data={};
load('ECOG_Grid_8596_000067_B3.mat')
acc_bin_batch=[];
acc_bin_online=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);




    %     %%%%%% load imagined data
    %     folders = session_data(i).folders(imag_idx);
    %     day_date = session_data(i).Day;
    %     files=[];
    %     for ii=1:length(folders)
    %         folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
    %         %cd(folderpath)
    %         files = [files;findfiles('',folderpath)'];
    %     end
    %     load('ECOG_Grid_8596_000067_B3.mat')
    %     condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0)];

    %%%%%% load online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,1) ];
    [acc_online_trial,acc_online] = accuracy_online_data(files);
    acc_bin_online(i,:,:) = acc_online;

    %%%%%% load batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,2) ];
    [acc_batch_trial,acc_batch] = accuracy_online_data(files);
    acc_bin_batch(i,:,:)  = acc_batch;
end

tmp=squeeze(mean(acc_bin_batch,1));
figure;imagesc(tmp)
colormap bone
caxis([0 1])
mean(diag(tmp))
xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
title(['Average Bin Level CL2 Acc with Daily Decoder:  ' num2str(mean(diag(tmp)))])
set(gcf,'Color','w')
set(gca,'FontSize',12)

% plotting overall
a=[];
for i=1:size(acc_bin_batch,1)
    tmp = squeeze(acc_bin_batch(i,:,:));
    a(i) = mean(diag(tmp));
end
figure;plot(a,'LineWidth',1)
xticks(1:length(session_data))
xlim([0.5 18.5])
ylim([0.0 1])
ylabel('Bin Level Decoding Accuracy')
xlabel('Days')
vline(12,'r')
title('CL2')


% plotting by movement
figure;
hold on
indiv_mvmt=[]
for j=1:7
    a=[];
    for i=12:size(acc_bin_batch,1)
        tmp = squeeze(acc_bin_batch(i,:,:));
        tmp = diag(tmp);
        a(i) = tmp(j);
    end
    indiv_mvmt(j) = mean(a(a>0));
    plot(a,'LineWidth',1)
end
xticks(1:length(session_data))
xlim([0.5 18.5])
ylim([0.0 1])
ylabel('Bin Level Decoding Accuracy')
xlabel('Days')
vline(12,'r')
title('CL2')
legend({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})

figure;bar(indiv_mvmt)
xticks(1:7)
xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
ylabel('Bin Level Decoding Accuracy, no mime')
ylim([0 1])

%% LOADING ALL DATA FROM A PNP EXPERIMENT AND LOOKING AT DECODING PROB
% the idea is to check if the raw decoder o/p at sample levels are
% different, or if the temporal pattern or trajectory is different

clc;clear
close all

addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
load('ECOG_Grid_8596_000067_B3.mat')
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20231129\Robot3DArrow';
folders = {'143141', '143636', '143936'};
cd(root_path)

% load the neural features
files=[];
for ii=1:length(folders)
    folderpath = fullfile(root_path, folders{ii},'BCI_Fixed');     
    files = [files;findfiles('',folderpath)'];
end
condn_data = [load_data_for_MLP_TrialLevel_B3(files,ecog_grid,1) ];

% pass it through the PnP decoder
load('C:\Users\nikic\Documents\GitHub\bci\clicker\net_PnP.mat')
decodes=[];
decodes_idx=[];
neural_feat=[];
trial_idx=[];
len=[];
for i=1:length(condn_data)
    tmp = condn_data(i).neural;
    out = predict(net_PnP,tmp')';
    condn_data(i).prob = out;
    decodes = [decodes out];
    neural_feat =[neural_feat condn_data(i).neural];
    decodes_idx = [decodes_idx condn_data(i).targetID*ones(size(out,2),1)'];
    trial_idx =[trial_idx i*ones(size(out,2),1)'];
    len = [len size(decodes,2)];
    %trial_idx{i} = i*ones(size(out,2),1)';
end
len=[0 len];

% run PCA 
[coeff,score,latent] = pca(decodes');
figure;hold on
cmap=turbo(7);
for i=1:length(score)
    col = cmap(decodes_idx(i),:);
    plot3(score(i,2),score(i,3),score(i,4),'.','MarkerSize',20,'Color',col)
end

% MDS scale
D = pdist(decodes','cosine');
Y = mdscale(D,2);
figure;hold on
cmap=turbo(7);
for i=1:length(Y)
    col = cmap(decodes_idx(i),:);
    %plot3(Y(i,1),Y(i,2),Y(i,3),'.','MarkerSize',20,'Color',col)
    plot(Y(i,1),Y(i,2),'.','MarkerSize',20,'Color',col)
end
Y=Y';

% for single trial look at variation and mean
trial_data={};
for i=1:length(len)-1
    tmp_data = decodes(:,len(i)+1:len(i+1));
    %tmp_data = Y(:,len(i)+1:len(i+1));
    m = mean(tmp_data,2);
    s = std(tmp_data')';
    s1= det(cov(tmp_data'));
    trial_data(i).mean = m;
    trial_data(i).std = s1;
    trial_data(i).targetID = decodes_idx(len(i)+1);
end


figure;hold on
cmap=parula(7);
for i=1:length(trial_data)
    a = [trial_data(i).mean; trial_data(i).std];
    col = cmap(trial_data(i).targetID,:);
    plot3(a(1),a(2),a(3),'.','MarkerSize',30,'Color',col);
end
grid on


