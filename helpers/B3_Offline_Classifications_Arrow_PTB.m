%% B3 MAIN TASK PERFORMANCE WITHIN A BLOCK IN TERMS OF TRIAL LEVEL ACCURACY

clc;clear
addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';

% get performance within a session
day_date = '20230215';
online_folders = {'120852','121443'};
batch_folders = {'122149','122735'};
plot_true=true;

% ONLNE (CL1)
files=[];
for ii=1:length(online_folders)
    folderpath = fullfile(root_path, day_date,...
        'RadialTaskMultiStateDiscreteArrow',online_folders{ii},'BCI_Fixed');    
    files = [files;findfiles('',folderpath)'];
end
% get the classification accuracy
acc_online = accuracy_online_data_B3(files,4);
if plot_true
    figure;imagesc(acc_online)
    colormap bone
    clim([0 1])
    set(gcf,'color','w')
end

% BATCH (CL2)
files=[];
for ii=1:length(batch_folders)
    folderpath = fullfile(root_path, day_date,...
        'RadialTaskMultiStateDiscreteArrow',batch_folders{ii},'BCI_Fixed');    
    files = [files;findfiles('',folderpath)'];
end
% get the classification accuracy
acc_batch = accuracy_online_data_B3(files,4);
if plot_true
    figure;imagesc(acc_batch)
    colormap bone
    clim([0 1])
    set(gcf,'color','w')
end


%% implement pooling and mode filtering

% load the imagined data and train a classifier using pooling



% pass online data through the classifier







