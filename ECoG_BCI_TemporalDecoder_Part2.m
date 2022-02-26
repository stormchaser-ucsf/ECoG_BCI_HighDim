 %% TEMPORAL DECODER USING NEURAL FEATURES FROM ONLINE DATA

clc;clear

root_path='E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

% for only 6 DoF original:
%foldernames = {'20210526','20210528','20210602','20210609_pm','20210611'};

foldernames = {'20210613','20210616','20210623','20210625','20210630','20210702',...
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
            
            % get state 3 data
            idx = find(TrialData.TaskState==3) ;            
            raw_data = cell2mat(TrialData.NeuralFeatures(idx));
            
            % get state 4 data
            idx1 = find(TrialData.TaskState==4) ;
            raw_data4 = cell2mat(TrialData.NeuralFeatures(idx1));
            
            % trial id
            id = TrialData.TargetID;
            
            % prune to delta, beta, hg, no pooling
            feat_id = [129:256 513:640 769:896]; 
            raw_data = raw_data(feat_id,:);
            raw_data4 = raw_data4(feat_id,:);
            
            % get things ready
            s = size(raw_data,2);            
            data_seg={};
            
            % extract with 2 bin overlap
             bins =1:3:s;
            raw_data = [raw_data raw_data4];
            for k=1:length(bins)-1
                tmp = raw_data(:,bins(k)+[0:4]);
                data_seg = cat(2,data_seg,tmp');
            end
            
            % extract random 5bins every second
                        
            if id==1
                D1i = cat(2,D1i,data_seg);
                %D1f = cat(2,D1f,feat_stats1);
            elseif id==2
                D2i = cat(2,D2i,data_seg);
                %D2f = cat(2,D2f,feat_stats1);
            elseif id==3
                D3i = cat(2,D3i,data_seg);
                %D3f = cat(2,D3f,feat_stats1);
            elseif id==4
                D4i = cat(2,D4i,data_seg);
                %D4f = cat(2,D4f,feat_stats1);
            elseif id==5
                D5i = cat(2,D5i,data_seg);
                %D5f = cat(2,D5f,feat_stats1);
            elseif id==6
                D6i = cat(2,D6i,data_seg);
                %D6f = cat(2,D6f,feat_stats1);
            elseif id==7
                D7i = cat(2,D7i,data_seg);
                %D7f = cat(2,D7f,feat_stats1);
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
            
            % get state 3 data
            idx = find(TrialData.TaskState==3) ;            
            raw_data = cell2mat(TrialData.NeuralFeatures(idx));
            
            % get state 4 data
            idx1 = find(TrialData.TaskState==4) ;
            raw_data4 = cell2mat(TrialData.NeuralFeatures(idx1));
            
            % trial id
            id = TrialData.TargetID;
            
            % prune to delta, beta, hg, no pooling
            feat_id = [129:256 513:640 769:896]; 
            raw_data = raw_data(feat_id,:);
            raw_data4 = raw_data4(feat_id,:);
            raw_data=raw_data';
            raw_data4=raw_data4';
            
            % get things ready
            s = size(raw_data,1);            
            data_seg={};
            
            % epoch the data 
            if s<5 % for really quick decisions just pad data from state 4
                len = 5-s;
                tmp = raw_data4(1:len,:);
                raw_data = [raw_data;tmp];
                data_seg = raw_data;
            elseif s>5 && s<=6 % if not so quick, prune to data to 600ms
                 bins =1:3:s;
                bins = bins(1:2);
                raw_data = [raw_data ;raw_data4];
                for k=1:length(bins)
                    tmp = raw_data(bins(k)+[0:4],:);
                    data_seg = cat(2,data_seg,tmp);
                end
            elseif s>6% for all other data length, have to parse the data in overlapping chuncks of 600ms, 50% overlap
                 bins =1:3:s;
                raw_data = [raw_data ;raw_data4];
                for k=1:length(bins)-1
                    tmp = raw_data(bins(k)+[0:4],:);
                    data_seg = cat(2,data_seg,tmp);
                end
            end
             
%             feat_stats = TrialData.FeatureStats;
%             feat_stats.Mean = feat_stats.Mean(769:end);
%             feat_stats.Var = feat_stats.Var(769:end);
%             clear feat_stats1
%             feat_stats1(1:length(data_seg)) = feat_stats;
            
            if id==1
                D1 = cat(2,D1,data_seg);
                %D1f = cat(2,D1f,feat_stats1);
            elseif id==2
                D2 = cat(2,D2,data_seg);
                %D2f = cat(2,D2f,feat_stats1);
            elseif id==3
                D3 = cat(2,D3,data_seg);
                %D3f = cat(2,D3f,feat_stats1);
            elseif id==4
                D4 = cat(2,D4,data_seg);
                %D4f = cat(2,D4f,feat_stats1);
            elseif id==5
                D5 = cat(2,D5,data_seg);
                %D5f = cat(2,D5f,feat_stats1);
            elseif id==6
                D6 = cat(2,D6,data_seg);
                %D6f = cat(2,D6f,feat_stats1);
            elseif id==7
                D7 = cat(2,D7,data_seg);
                %D7f = cat(2,D7f,feat_stats1);
            end
        end
    end
    
end

 
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save lstm_data_with_imag_data_NeuralFeatOnlineData ...
    D1 D2 D3 D4 D5 D6 D7 D1i D2i D3i D4i D5i D6i D7i -v7.3

%% %%%%%%%%% BUILDING THE BILSTM WITH DATA EPOCHS

clc;clear
load lstm_data_with_imag_data_NeuralFeatOnlineData
Y=[];

% CONDN 1
condn_data1 = zeros(5,384,length(D1)+length(D1i));
k=1;
for i=1:length(D1)
    disp(k)
    tmp = D1{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
    %Y=cat(2,Y,ones(5,1));
end
for i=1:length(D1i)
    disp(k)
    tmp = D1i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
    %Y=cat(2,Y,ones(5,1));
end
clear D1 D1i


% CONDN 2
condn_data2 = zeros(5,384,length(D2)+length(D2i));
k=1;
for i=1:length(D2)
    disp(k)
    tmp = D2{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
for i=1:length(D2i)
    disp(k)
    tmp = D2i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
clear D2 D2i


% CONDN 3
condn_data3 = zeros(5,384,length(D3)+length(D3i));
k=1;
for i=1:length(D3)
    disp(k)
    tmp = D3{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
for i=1:length(D3i)
    disp(k)
    tmp = D3i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
clear D3 D3i 



% CONDN 4
condn_data4 = zeros(5,384,length(D4)+length(D4i));
k=1;
for i=1:length(D4)
    disp(k)
    tmp = D4{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
for i=1:length(D4i)
    disp(k)
    tmp = D4i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
clear D4 D4i 



% CONDN 5
condn_data5 = zeros(5,384,length(D5)+length(D5i));
k=1;
for i=1:length(D5)
    disp(k)
    tmp = D5{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
for i=1:length(D5i)
    disp(k)
    tmp = D5i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
clear D5 D5i 


% CONDN 6
condn_data6 = zeros(5,384,length(D6)+length(D6i));
k=1;
for i=1:length(D6)
    disp(k)
    tmp = D6{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
for i=1:length(D6i)
    disp(k)
    tmp = D6i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
clear D6 D6i 


% CONDN 7
condn_data7 = zeros(5,384,length(D7)+length(D7i));
k=1;
for i=1:length(D7)
    disp(k)
    tmp = D7{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
for i=1:length(D7i)
    disp(k)
    tmp = D7i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
clear D7 D7i 

condn_data_new=[];
condn_data_new = cat(3,condn_data_new,condn_data1);
condn_data_new = cat(3,condn_data_new,condn_data2);
condn_data_new = cat(3,condn_data_new,condn_data3);
condn_data_new = cat(3,condn_data_new,condn_data4);
condn_data_new = cat(3,condn_data_new,condn_data5);
condn_data_new = cat(3,condn_data_new,condn_data6);
condn_data_new = cat(3,condn_data_new,condn_data7);

% clearvars -except Y condn_data1 condn_data2 condn_data3 condn_data4 condn_data5...
%     condn_data6 condn_data7

clear condn_data1 condn_data2 condn_data3 condn_data4 condn_data5 condn_data6 condn_data7

% 
% idx = randperm(size(condn_data_new,3));
% condn_data_new = condn_data_new(:,:,idx);
% Y=Y(idx);

idx = randperm(size(condn_data_new,3),round(0.9*size(condn_data_new,3)));
I = zeros(size(condn_data_new,3),1);
I(idx)=1;

XTrain={};
XTest={};
%YTrain={};k=1;
%YTest={};j=1;
YTrain=[];
YTest=[];
for i=1:size(condn_data_new,3)    
    tmp = squeeze(condn_data_new(:,:,i));
    if I(i)==1
        %tmp  = tmp+randn(size(tmp))*0.1;
        XTrain = cat(1,XTrain,tmp');
        YTrain = [YTrain Y(i)];
        %YTrain{k} = categorical(Y(i)*ones(1,5));
        %k=k+1;
    else
        XTest = cat(1,XTest,tmp');
        YTest = [YTest Y(i)];
        %YTest{j} = categorical(Y(i)*ones(1,5));
        %j=j+1;
    end    
end

% % shuffle
idx  = randperm(length(YTrain));
XTrain = XTrain(idx);
YTrain = YTrain(idx);
YTrain = categorical(YTrain');
YTest = categorical(YTest');

% for sequencyce classificaion
% YTrain = (YTrain');
% YTest = (YTest');




% specify lstm structure
inputSize = 384;
numHiddenUnits1 = [128 256];
drop1 = [ 0.4 0.4];
numClasses = 7;
for i=2
    numHiddenUnits=numHiddenUnits1(i);
    drop=drop1(i);
    layers = [ ...
        sequenceInputLayer(inputSize)
        bilstmLayer(numHiddenUnits,'OutputMode','sequence')
        dropoutLayer(drop)        
        gruLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(drop)        
        batchNormalizationLayer
        fullyConnectedLayer(25)
        reluLayer
        dropoutLayer(.25)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    
    
    % options
    options = trainingOptions('adam', ...
        'MaxEpochs',10, ...        
        'MiniBatchSize',128, ...
        'GradientThreshold',5, ...
        'Verbose',true, ...
        'ValidationFrequency',20,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XTest,YTest},...
        'ValidationPatience',6,...
        'Plots','training-progress');
    
    % train the model
    net = trainNetwork(XTrain,YTrain,layers,options);
end
% 
net_bilstm=net; 
save net_bilstm net_bilstm

%% TESTING THE BILSTM ONLINE


clc;clear
load net_bilstm
filepath='E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20211029\Robot3DArrow';

% get the name of the files
files = findfiles('mat',filepath,1)';
files1=[];
for i=1:length(files)
   if regexp(files{i},'Data')
       files1=[files1;files(i)];
   end    
end
files=files1;
clear files1


% load the data, and run it through the classifier
decodes_overall=[];
for i=1:length(files)
    disp(i)
    
    % load
    load(files{i})
    
    % create buffer
    data_buffer = randn(5,384)*0.25;
    
    %get data
    raw_data = TrialData.NeuralFeatures;
    feat_id = [129:256 513:640 769:896];
    
    % state of trial
    state_idx = TrialData.TaskState;
    decodes=[];
    for j=1:length(raw_data)
        % get neural features in buffer
        tmp = raw_data{j};
        tmp = tmp(feat_id);
        data_buffer = circshift(data_buffer,-1);
        data_buffer(end,:)=tmp';
        
        % classifier output
        out=predict(net_bilstm,data_buffer');
        %out=mean(out,2);
        [aa bb]=max(out);
        %if aa>0.4
            class_predict = bb;
        %else
        %    class_predict=8;
        %end
        
        % store results
        if state_idx(j)==3
            decodes=[decodes class_predict];
        end
    end
    decodes_overall(i).decodes = decodes;
    decodes_overall(i).tid = TrialData.TargetID;
end

% looking at the accuracy of the bilstm decoder overall
acc=zeros(7,8);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    for j=1:length(tmp)
       acc(tid,tmp(j)) =  acc(tid,tmp(j))+1;
    end
end
for i=1:size(acc,1)
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

% looking at accuracy in terms of max decodes
acc_trial=zeros(7,7);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    acc1=zeros(7,7);
    for j=1:length(tmp)
       acc1(tid,tmp(j)) =  acc1(tid,tmp(j))+1;
    end
    acc1=sum(acc1);
    [aa bb]=max(acc1);
    acc_trial(tid,bb)=acc_trial(tid,bb)+1;
end
for i=1:length(acc_trial)
    acc_trial(i,:) = acc_trial(i,:)/sum(acc_trial(i,:));
end


%comparing to the mlp
acc_mlp=zeros(7,7);
acc_mlp_trial=zeros(7,7);
for i=1:length(files)
    disp(i)
    
    % load
    load(files{i})
    
    decodes = TrialData.ClickerState;
    tid=TrialData.TargetID;
    acc1=zeros(7,7);
    for j=1:length(decodes)
        if decodes(j)>0
            acc_mlp(tid,decodes(j))=acc_mlp(tid,decodes(j))+1;
            acc1(tid,decodes(j))=acc1(tid,decodes(j))+1;
        end
    end
    tmp=sum(acc1);
    [aa bb]=max(tmp);
    acc_mlp_trial(tid,bb)=acc_mlp_trial(tid,bb)+1;    
end
for i=1:length(acc_mlp)
    acc_mlp(i,:) = acc_mlp(i,:)./sum(acc_mlp(i,:));
    acc_mlp_trial(i,:) = acc_mlp_trial(i,:)./sum(acc_mlp_trial(i,:));
end

mean([diag(acc) diag(acc_mlp)])
mean([diag(acc_trial) diag(acc_mlp_trial)])










