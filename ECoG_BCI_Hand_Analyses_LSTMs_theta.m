

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';


% GETTING DATA FROM THE HAND TASK, all but the last day's data
foldernames = {'20220128','20220204','20220209','20220223'};
cd(root_path)

hand_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Hand');
    if ~exist(folderpath)
        folderpath = fullfile(root_path, foldernames{i},'HandOnline');
    end
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
D9={};
D10={};
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
        bins =1:450:s;
        raw_data = [raw_data;raw_data4];
        for k=1:length(bins)-1
            tmp = raw_data(bins(k)+[0:499],:);
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
        elseif action==9
            D9 = cat(2,D9,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        elseif action==10
            D10 = cat(2,D10,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        end
    end
end


cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save lstm_hand_data_ForTheta D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 -v7.3

%% PREPROCESS FOR LSTM
% first only theta band
% then theta-pass of hg

clear;clc

Y=[];

condn_data_new=[];jj=1;

load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20211001\Robot3DArrow\103931\BCI_Fixed\Data0001.mat')
chmap = TrialData.Params.ChMap;


% log spaced hg filters
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [70,77];   % high gamma1
Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
Params.FilterBank(end+1).fpass = [8]; % theta
Params.FilterBank(end+1).fpass = [13,19]; % beta1
Params.FilterBank(end+1).fpass = [19,30]; % beta2
Params.FilterBank(end+1).fpass = [70,150]; % raw_hg

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

%%%% D1%%%%%%%
load('lstm_hand_data_ForTheta','D1')
D1i={};
condn_data1 = zeros(500,128,length(D1)+length(D1i));
k=1;
for i=1:length(D1)
    disp(k)
    tmp = D1{i};
    tmp1(:,1,:)=tmp;   
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end

for ii=1:size(condn_data1,3)
    disp(ii)
    
    tmp = squeeze(condn_data1(:,:,ii));
    tmp_theta = preprocess_bilstm(tmp,Params);
    
    % make new data structure
    %tmp = [tmp_hg tmp_lp];
    tmp = tmp_theta;
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end
%%%% D1 END %%%%



%%%% D2%%%%%%%
load('lstm_hand_data_ForTheta','D2')
D2i={};
condn_data2 = zeros(500,128,length(D2)+length(D2i));
k=1;
for i=1:length(D2)
    disp(k)
    tmp = D2{i};
    tmp1(:,1,:)=tmp;   
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end

for ii=1:size(condn_data2,3)
    disp(ii)
    
    tmp = squeeze(condn_data2(:,:,ii));
    tmp_theta = preprocess_bilstm(tmp,Params);
    
    % make new data structure
    %tmp = [tmp_hg tmp_lp];
    tmp = tmp_theta;
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end
%%%% D2 END %%%%


%%%% D3%%%%%%%
load('lstm_hand_data_ForTheta','D3')
D3i={};
condn_data3 = zeros(500,128,length(D3)+length(D3i));
k=1;
for i=1:length(D3)
    disp(k)
    tmp = D3{i};
    tmp1(:,1,:)=tmp;   
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end

for ii=1:size(condn_data3,3)
    disp(ii)
    
    tmp = squeeze(condn_data3(:,:,ii));
    tmp_theta = preprocess_bilstm(tmp,Params);
    
    % make new data structure
    %tmp = [tmp_hg tmp_lp];
    tmp = tmp_theta;
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end
%%%% D3 END %%%%

%%%% D4%%%%%%%
load('lstm_hand_data_ForTheta','D4')
D4i={};
condn_data4 = zeros(500,128,length(D4)+length(D4i));
k=1;
for i=1:length(D4)
    disp(k)
    tmp = D4{i};
    tmp1(:,1,:)=tmp;   
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end

for ii=1:size(condn_data4,3)
    disp(ii)
    
    tmp = squeeze(condn_data4(:,:,ii));
    tmp_theta = preprocess_bilstm(tmp,Params);
    
    % make new data structure
    %tmp = [tmp_hg tmp_lp];
    tmp = tmp_theta;
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end
%%%% D4 END %%%%
clear condn_data1 condn_data2 condn_data3 condn_data4 D1 D2 D3 D4



%%%% D5%%%%%%%
load('lstm_hand_data_ForTheta','D5')
D5i={};
condn_data5 = zeros(500,128,length(D5)+length(D5i));
k=1;
for i=1:length(D5)
    disp(k)
    tmp = D5{i};
    tmp1(:,1,:)=tmp;   
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end

for ii=1:size(condn_data5,3)
    disp(ii)
    
    tmp = squeeze(condn_data5(:,:,ii));
    tmp_theta = preprocess_bilstm(tmp,Params);
    
    % make new data structure
    %tmp = [tmp_hg tmp_lp];
    tmp = tmp_theta;
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end
%%%% D5 END %%%%


%%%%% STM %%%

% set aside training and testing data in a cell format
idx = randperm(size(condn_data_new,3),round(0.85*size(condn_data_new,3)));
I = zeros(size(condn_data_new,3),1);
I(idx)=1;

XTrain={};
XTest={};
YTrain=[];
YTest=[];
for i=1:size(condn_data_new,3)
    tmp = squeeze(condn_data_new(:,:,i));
    if I(i)==1
        XTrain = cat(1,XTrain,tmp');
        YTrain = [YTrain Y(i)];
    else
        XTest = cat(1,XTest,tmp');
        YTest = [YTest Y(i)];
    end
end

% shuffle
idx  = randperm(length(YTrain));
XTrain = XTrain(idx);
YTrain = YTrain(idx);

YTrain = categorical(YTrain');
YTest = categorical(YTest');


%clear condn_data_new

% specify lstm structure
inputSize = 128;
numHiddenUnits1 = [  96 128 150 200 325];
drop1 = [ 0.3 0.3 0.3  0.4 0.4];
numClasses = 5;
for i=1%1:length(drop1)
    numHiddenUnits=numHiddenUnits1(i);
    drop=drop1(i);
    layers = [
        sequenceInputLayer(inputSize)        
%         bilstmLayer(numHiddenUnits,'OutputMode','sequence')
%         dropoutLayer(drop)
        gruLayer(numHiddenUnits,'OutputMode','last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    
    
    % options
    options = trainingOptions('adam', ...
        'MaxEpochs',30, ...
        'MiniBatchSize',64, ...
        'GradientThreshold',2, ...
        'Verbose',true, ...
        'ValidationFrequency',64,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XTest,YTest},...
        'ValidationPatience',6,...
        'Plots','training-progress');
    
    % train the model
    net = trainNetwork(XTrain,YTrain,layers,options);
end





