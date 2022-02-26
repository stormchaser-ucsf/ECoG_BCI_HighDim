%% LSTMS for the hand task
% extract 800ms snippets with 400ms overlap, train LSTMS
% use the fine tuning approach: fine tune on subset of trials, test on held
% out trials 


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20211201','20211203','20211206','20211208','20211215','20211217',...
'20220126'};
cd(root_path)

imagined_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'ImaginedMvmtDAQ')
    D=dir(folderpath);
    if i==3
        D = D([1:3 5:7 9:end]);
    elseif i==4
        D = D([1:3 5:end]);
    elseif i==6
        D = D([1:5 7:end]);
    end
    
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
        imagined_files = [imagined_files;findfiles('',filepath)'];
    end
end


% load the data for the imagined files, if they belong to right thumb,
% index, middle, ring, pinky, pinch, tripod, power
D1i={};
D2i={};
D3i={};
D4i={};
D5i={};
D6i={};
D7i={};
D8i={};
for i=1:length(imagined_files)
    disp(i/length(imagined_files)*100)
    try
        load(imagined_files{i})
        file_loaded = true;
    catch
        file_loaded=false;
        disp(['Could not load ' files{j}]);
    end
    
    if file_loaded
        action = TrialData.ImaginedAction;
        idx = find(TrialData.TaskState==3) ;
        raw_data = cell2mat(TrialData.BroadbandData(idx)');
        idx1 = find(TrialData.TaskState==4) ;
        raw_data4 = cell2mat(TrialData.BroadbandData(idx1)');
        s = size(raw_data,1);
        data_seg={};
        bins =1:400:s;
        raw_data = [raw_data;raw_data4];
        for k=1:length(bins)-1
            tmp = raw_data(bins(k)+[0:799],:);
            data_seg = cat(2,data_seg,tmp);
        end
        
        if strcmp('Right Thumb',action)
            D1i = cat(2,D1i,data_seg);
            %D1f = cat(2,D1f,feat_stats1);
        elseif strcmp('Right Index',action)
            D2i = cat(2,D2i,data_seg);
            %D2f = cat(2,D2f,feat_stats1);
        elseif strcmp('Right Middle',action)
            D3i = cat(2,D3i,data_seg);
            %D3f = cat(2,D3f,feat_stats1);
        elseif strcmp('Right Ring',action)
            D4i = cat(2,D4i,data_seg);
            %D4f = cat(2,D4f,feat_stats1);
        elseif strcmp('Right Pinky',action)
            D5i = cat(2,D5i,data_seg);
            %D5f = cat(2,D5f,feat_stats1);
        elseif strcmp('Right Pinch Grasp',action)
            D6i = cat(2,D6i,data_seg);
            %D6f = cat(2,D6f,feat_stats1);
        elseif strcmp('Right Tripod Grasp',action)
            D7i = cat(2,D7i,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        elseif strcmp('Right Power Grasp',action)
            D8i = cat(2,D8i,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        end
    end    
end



% GETTING DATA FROM THE HAND TASK, all but the last day's data 
foldernames = {'20220128','20220204','20220209','20220223'}; 
cd(root_path)

hand_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Hand')
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        tmp=dir(filepath);
        hand_files = [hand_files;findfiles('',filepath)'];
    end
end


% load the data for the imagined files, if they belong to right thumb,
% index, middle, ring, pinky, pinch, tripod, power
D1={};
D2={};
D3={};
D4={};
D5={};
D6={};
D7={};
D8={};
for i=1:length(hand_files)
    disp(i/length(hand_files)*100)
    try
        load(hand_files{i})
        file_loaded = true;
    catch
        file_loaded=false;
        disp(['Could not load ' hand_files{j}]);
    end
    
    if file_loaded
        action = TrialData.TargetID;
        idx = find(TrialData.TaskState==3) ;
        raw_data = cell2mat(TrialData.BroadbandData(idx)');
        idx1 = find(TrialData.TaskState==4) ;
        raw_data4 = cell2mat(TrialData.BroadbandData(idx1)');
        s = size(raw_data,1);
        data_seg={};
        bins =1:400:s;
        raw_data = [raw_data;raw_data4];
        for k=1:length(bins)-1
            tmp = raw_data(bins(k)+[0:799],:);
            data_seg = cat(2,data_seg,tmp);
        end
        
        if action==1
            D1 = cat(2,D1,data_seg);
            %D1f = cat(2,D1f,feat_stats1);
        elseif action==2
            D2 = cat(2,D2,data_seg);
            %D2f = cat(2,D2f,feat_stats1);
        elseif action==3
            D3 = cat(2,D3,data_seg);
            %D3f = cat(2,D3f,feat_stats1);
        elseif action==4
            D4 = cat(2,D4,data_seg);
            %D4f = cat(2,D4f,feat_stats1);
        elseif action==5
            D5 = cat(2,D5,data_seg);
            %D5f = cat(2,D5f,feat_stats1);
        elseif action==6
            D6 = cat(2,D6,data_seg);
            %D6f = cat(2,D6f,feat_stats1);
        elseif action==7
            D7 = cat(2,D7,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        elseif action==8
            D8 = cat(2,D8,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        end
    end    
end


cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save lstm_hand_data D1 D2 D3 D4 D5 D6 D7 D8 D1i D2i D3i D4i D5i D6i D7i D8i -v7.3

