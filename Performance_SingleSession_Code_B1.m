%% COMPUTES THE PERFORMANCE OF A SINGLE GIVEN SESSION

%%
clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')

day_date= '20240605';

folders=dir(fullfile(root_path,day_date,'Robot3DArrow'));
folders=folders(3:end);


%%%%%% cross_val classification accuracy for imagined data
% get the files
files=[];
for ii=1:9
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders(ii).name,'Imagined');
    if exist(folderpath)
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
end

%load the data
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\ECOG_Grid_8596_000067_B3.mat')
condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0,0);

% get cross val decoding accuracy
iterations=10;
[acc_imagined,train_permutations,acc_bin,bino_pdf,bino_pdf_chance]...
    = accuracy_imagined_data_4DOF(condn_data, iterations);

acc_imagined=squeeze(nanmean(acc_imagined,1));
figure;imagesc(acc_imagined*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_imagined,1)
    for k=1:size(acc_imagined,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:4)
yticks(1:4)
xticklabels({'Rt Thumb','Rt. Wrist Flexion','Lt. Thumb','Rt. Wrist Extension'})
yticklabels({'Rt Thumb','Rt. Wrist Flexion','Lt. Thumb','Rt. Wrist Extension'})
title(['Cross-val trial level OL Acc of ' num2str(100*mean(diag(acc_imagined)))])

%%%% get online decoding accuracy (CL1)
files=[];
for ii=15:17
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders(ii).name,'BCI_Fixed');
    if exist(folderpath)
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
end

%load the data
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\ECOG_Grid_8596_000067_B3.mat')
condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0,0);

% get classifier accuracy and plot
num_targets = length(unique([condn_data.targetID]));
[acc_online,acc_online_bins,bino_pdf] = accuracy_online_data_AnyDOF(files,num_targets);
%acc_online=acc_online_bins;
figure;imagesc(acc_online*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_online,1)
    for k=1:size(acc_online,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:4)
yticks(1:4)
xticklabels({'Rt Thumb','Rt. Wrist Flexion','Lt. Thumb','Rt. Wrist Extension'})
yticklabels({'Rt Thumb','Rt. Wrist Flexion','Lt. Thumb','Rt. Wrist Extension'})
title(['CL1 Acc of ' num2str(100*mean(diag(acc_online)))])



%%%% get batch update decoding accuracy (CL2)
files=[];
for ii=15:17
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders(ii).name,'BCI_Fixed');
    if exist(folderpath)
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
end

%load the data
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\ECOG_Grid_8596_000067_B3.mat')
condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0,0);

% get classifier accuracy and plot
num_targets = length(unique([condn_data.targetID]));
[acc_batch,acc_batch_bins,bino_pdf] = accuracy_online_data_AnyDOF(files,num_targets);
%acc_batch=acc_batch_bins;
figure;imagesc(acc_batch*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_batch,1)
    for k=1:size(acc_batch,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:4)
yticks(1:4)
xticklabels({'Rt Thumb','Rt. Wrist Flexion','Lt. Thumb','Rt. Wrist Extension'})
yticklabels({'Rt Thumb','Rt. Wrist Flexion','Lt. Thumb','Rt. Wrist Extension'})
title(['CL2 Acc of ' num2str(100*mean(diag(acc_batch)))])

%% PLOTTING CENTER OUT TRAJECTORIES

clc;clear;close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')

day_date= '20240605';

folders=dir(fullfile(root_path,day_date,'Robot3D'));
folders=folders(3:end);
%folders=folders(3);

%get the files
files=[];
for ii=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3D',folders(ii).name,'BCI_Fixed');
    if exist(folderpath)
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
end

%plot the center out traj
figure;
hold on
%col={'r','g','b','m','k','c','y','o'};
%col = turbo(8);
col = turbo(4);
col=[col;col];
targets=[];
for i=1:length(files)
    load(files{i})
    kin = TrialData.CursorState;
    task_state = TrialData.TaskState;
    idx= find(task_state==3);
    kin = kin(1:3,idx);
    tid = TrialData.TargetID;
    if tid>4
        tid=tid-2;
    end
    targets(i)=tid;
    %col_id = col{tid};
    %if tid>4 && tid<9
    if tid<=4
        col_id = col(tid,:);
        plot(kin(1,:),kin(2,:),'LineWidth',2,'color',col_id);
        pos=TrialData.TargetPosition;
        plot(pos(1),pos(2),'o','Color',col_id,'MarkerSize',75)
    end
    plot(0,0,'.r','MarkerSize',50)

end
xlim([-300 300])
ylim([-300 300])
