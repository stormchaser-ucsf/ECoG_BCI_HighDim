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
layers = get_layers1(120,759);
%layers = get_layers2(120,120,759);

% train the MLP
net = trainNetwork(XTrain,YTrain,layers,options);

% save the network
net_mlp_v0 = net;
save net_mlp_v0 net_mlp_v0


%% batch update with more recent data collected on covert miming


condn_data={};
load('ECOG_Grid_8596_000067_B3.mat')
for i=12:18%all the original data without covert mime
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
    get_options(condn_data_overall,test_idx,train_idx,1e-4);
options.Plots='training-progress';

% batch update the MLP
clear net_mlp_v0
load net_mlp_v0
layers = net_mlp_v0.Layers;
net_mlp_v0 = trainNetwork(XTrain,YTrain,layers,options);


%% test performance on held out days


condn_data={};
for i=19%all the original data without covert mime
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
aa=mean([diag(acc_batch);diag(acc_online)]);

disp([aa cv_perf])


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







