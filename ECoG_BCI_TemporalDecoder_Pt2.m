
%%%%% BUILDING BILSTSM DECODER USING JUST THE ONLINE DATA %%%

%% getting the data 

clc;clear

root_path='/media/reza/WindowsDrive/BRAVO1/CursorPlatform/Data';

% for only 6 DoF original:
%foldernames = {'20210526','20210528','20210602','20210609_pm','20210611'};

foldernames = {'20210615','20210616','20210623','20210625','20210630','20210702',...
    '20210707','20210716','20210728','20210804','20210806','20210813','20210818',...
    '20210825','20210827','20210901','20210903','20210910','20210917','20210924','20210929',...
    '20211001''20211006','20211008','20211013','20211015','20211022'};
cd(root_path)

imag_files={};
online_files={};
k=1;jj=1;
for i=1:length(foldernames)
    disp([i/length(foldernames)]);
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    if i==19 % this is 20210917
        idx = [1 2 5:8 9:10];
        D = D(idx);        
    end
    imag_files_temp=[];
    online_files_temp=[];
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        if exist(filepath)
            imag_files_temp = [imag_files_temp;findfiles('mat',filepath)'];
        end
        filepath1=fullfile(folderpath,D(j).name,'BCI_Fixed');
        if exist(filepath1)
            online_files_temp = [online_files_temp;findfiles('mat',filepath1)'];
        end
    end
    if ~isempty(imag_files_temp)
        imag_files{k} = imag_files_temp;k=k+1;
    end
    if ~isempty(online_files_temp)
        online_files{jj} = online_files_temp;jj=jj+1;
    end
    %     imag_files{i} = imag_files_temp;
    %     online_files{i} = online_files_temp;
end

% GETTING DATA FROM IMAGINED CONTROL IN THE ARROW TASK
D1i={};
D2i={};
D3i={};
D4i={};
D5i={};
D6i={};
D7i={};
D1if={};
D2if={};
D3if={};
D4if={};
D5if={};
D6if={};
D7if={};
for i=1:length(imag_files)
    files = imag_files{i};
    disp(i/length(imag_files))
    for j=1:length(files)
        try
            load(files{j})
            file_loaded = true;
        catch
            file_loaded=false;
            disp(['Could not load ' files{j}]);
        end
        if file_loaded
            idx = find(TrialData.TaskState==3) ;
            %raw_data = cell2mat(TrialData.BroadbandData(idx)');
            raw_data = cell2mat(TrialData.NeuralFeatures(idx));
            idx1 = find(TrialData.TaskState==4) ;
            raw_data4 = cell2mat(TrialData.BroadbandData(idx1)');
            id = TrialData.TargetID;
            s = size(raw_data,1);
            data_seg={};
            if s<800 % for really quick decisions just pad data from state 4
                len = 800-s;
                tmp = raw_data4(1:len,:);
                raw_data = [raw_data;tmp];
                data_seg = raw_data;
            elseif s>800 && s<1000 % if not so quick, prune to data to 600ms
                raw_data = raw_data(1:800,:);
                data_seg = raw_data;
            elseif s>1000% for all other data length, have to parse the data in overlapping chuncks of 600ms, 50% overlap
                bins =1:400:s;
                raw_data = [raw_data;raw_data4];
                for k=1:length(bins)-1
                    tmp = raw_data(bins(k)+[0:799],:);
                    data_seg = cat(2,data_seg,tmp);
                end
            end
            
            feat_stats = TrialData.FeatureStats;
            idx=[129:256 513:640 769:896];
            feat_stats.Mean = feat_stats.Mean(idx);
            feat_stats.Var = feat_stats.Var(idx);
            clear feat_stats1
            feat_stats1(1:length(data_seg)) = feat_stats;
            
            if id==1
                D1i = cat(2,D1i,data_seg);
                D1if = cat(2,D1if,feat_stats1);
            elseif id==2
                D2i = cat(2,D2i,data_seg);
                D2if = cat(2,D2if,feat_stats1);
            elseif id==3
                D3i = cat(2,D3i,data_seg);
                D3if = cat(2,D3if,feat_stats1);
            elseif id==4
                D4i = cat(2,D4i,data_seg);
                D4if = cat(2,D4if,feat_stats1);
            elseif id==5
                D5i = cat(2,D5i,data_seg);
                D5if = cat(2,D5if,feat_stats1);
            elseif id==6
                D6i = cat(2,D6i,data_seg);
                D6if = cat(2,D6if,feat_stats1);
            elseif id==7
                D7i = cat(2,D7i,data_seg);
                D7if = cat(2,D7if,feat_stats1);
            end
        end
    end
    
end


% GETTING DATA FROM ONLINE BCI CONTROL IN THE ARROW TASK
% essentially getting 600ms epochs
D1={};
D2={};
D3={};
D4={};
D5={};
D6={};
D7={};
D1f={};
D2f={};
D3f={};
D4f={};
D5f={};
D6f={};
D7f={};
for i=1:length(online_files)
    files = online_files{i};
    disp(i/length(online_files))
    for j=1:length(files)
        try
            load(files{j})
            file_loaded = true;
        catch
            file_loaded=false;
            disp(['Could not load ' files{j}]);
        end
        if file_loaded
            idx = find(TrialData.TaskState==3) ;
            raw_data = cell2mat(TrialData.BroadbandData(idx)');
            idx1 = find(TrialData.TaskState==4) ;
            raw_data4 = cell2mat(TrialData.BroadbandData(idx1)');
            id = TrialData.TargetID;
            s = size(raw_data,1);
            data_seg={};
           if s<800 % for really quick decisions just pad data from state 4
                len = 800-s;
                tmp = raw_data4(1:len,:);
                raw_data = [raw_data;tmp];
                data_seg = raw_data;
            elseif s>800 && s<1000 % if not so quick, prune to data to 600ms
                raw_data = raw_data(1:800,:);
                data_seg = raw_data;
            elseif s>1000% for all other data length, have to parse the data in overlapping chuncks of 600ms, 50% overlap
                bins =1:400:s;
                raw_data = [raw_data;raw_data4];
                for k=1:length(bins)-1
                    tmp = raw_data(bins(k)+[0:799],:);
                    data_seg = cat(2,data_seg,tmp);
                end
            end
            
            feat_stats = TrialData.FeatureStats;
            idx=[129:256 513:640 769:896];
            feat_stats.Mean = feat_stats.Mean(idx);
            feat_stats.Var = feat_stats.Var(idx);
            clear feat_stats1
            feat_stats1(1:length(data_seg)) = feat_stats;
            
            if id==1
                D1 = cat(2,D1,data_seg);
                D1f = cat(2,D1f,feat_stats1);
            elseif id==2
                D2 = cat(2,D2,data_seg);
                D2f = cat(2,D2f,feat_stats1);
            elseif id==3
                D3 = cat(2,D3,data_seg);
                D3f = cat(2,D3f,feat_stats1);
            elseif id==4
                D4 = cat(2,D4,data_seg);
                D4f = cat(2,D4f,feat_stats1);
            elseif id==5
                D5 = cat(2,D5,data_seg);
                D5f = cat(2,D5f,feat_stats1);
            elseif id==6
                D6 = cat(2,D6,data_seg);
                D6f = cat(2,D6f,feat_stats1);
            elseif id==7
                D7 = cat(2,D7,data_seg);
                D7f = cat(2,D7f,feat_stats1);
            end
        end
    end
    
end

 
cd('/home/reza/Documents/MATLAB/HighDimECoG_Paper')
save lstm_data_with_imag_data D1 D2 D3 D4 D5 D6 D7...
    D1i D2i D3i D4i D5i D6i D7i...
    D1if D2if D3if D4if D5if D6if D7if -v7.3


