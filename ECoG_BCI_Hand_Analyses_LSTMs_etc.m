%% LSTMS for the hand task
% extract 800ms snippets with 400ms overlap, train LSTMS
% use the fine tuning approach: fine tune on subset of trials, test on held
% out trials


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20211201','20211203','20211206','20211208','20211215','20211217',...
    '20220126','20220302'};
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
        if ~exist(filepath)
            filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        end
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
D9i={};
D10i={};
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
    %disp(i/length(hand_files)*100)
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
save lstm_hand_data D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D1i D2i D3i D4i D5i D6i D7i D8i -v7.3

c=0;
files=[];
for i=1:length(hand_files)
    if (regexp(hand_files{i},'Data0008'))
        c=c+1;
        files=[files;hand_files(i)];
    end
end
c

c=[];
for i=1:length(files)
    load(files{i})
    c(i)=TrialData.TargetID;
end
figure;stem(c)

%% LSTMS ON ONLY THE HAND TASK AND NOT IMAGINED 30DOF



clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';


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
D1i={};
D2i={};
D3i={};
D4i={};
D5i={};
D6i={};
D7i={};
D8i={};
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
        data_seg={};
        data_seg{1} = raw_data(1:2700,:);
        data_seg{2} = raw_data(2701:end,:);
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
save lstm_hand_data_ONLYHANDTASK D1 D2 D3 D4 D5 D6 D7 D8 D1i D2i D3i D4i D5i D6i D7i D8i -v7.3

%% TRAIN RNN ON HANDONLY TASK WITH LONGER TIME WINDOWS

clear;clc

Y=[];

condn_data_new=[];jj=1;

load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20211001\Robot3DArrow\103931\BCI_Fixed\Data0001.mat')
chmap = TrialData.Params.ChMap;


% low pass filter of raw
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',20,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

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
Params.FilterBank(end+1).fpass = [0.5,4]; % low pass
Params.FilterBank(end+1).fpass = [13,19]; % beta1
Params.FilterBank(end+1).fpass = [19,30]; % beta2

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

%hg1
hg1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',77, ...
    'SampleRate',1e3);
hg2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',77,'HalfPowerFrequency2',85, ...
    'SampleRate',1e3);
hg3 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',85,'HalfPowerFrequency2',93, ...
    'SampleRate',1e3);
hg4 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',93,'HalfPowerFrequency2',102, ...
    'SampleRate',1e3);
hg5 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',102,'HalfPowerFrequency2',113, ...
    'SampleRate',1e3);
hg6 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',113,'HalfPowerFrequency2',124, ...
    'SampleRate',1e3);
hg7 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',124,'HalfPowerFrequency2',136, ...
    'SampleRate',1e3);
hg8 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',136,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);





cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load('lstm_hand_data','D1','D1i')
condn_data1 = zeros(800,128,length(D1)+length(D1i));
k=1;
for i=1:length(D1)
    disp(k)
    tmp = D1{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
for i=1:length(D1i)
    disp(k)
    tmp = D1i{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
clear D1 D1i


for ii=1:size(condn_data1,3)
    disp(ii)
    
    tmp = squeeze(condn_data1(:,:,ii));
    
    % hg filtering
    %tmp_hg = abs(hilbert(filtfilt(hgFilt,tmp)));
    %tmp_hg = filtfilt(lpFilt1,tmp_hg);
    
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    % make new data structure
    %tmp = [tmp_hg tmp_beta tmp_delta];
    tmp = [tmp_hg tmp_lp];
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

load('lstm_hand_data','D2','D2i')
condn_data2 = zeros(800,128,length(D2)+length(D2i));
k=1;
for i=1:length(D2)
    disp(k)
    tmp = D2{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
for i=1:length(D2i)
    disp(k)
    tmp = D2i{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
clear D2 D2i condn_data1


for ii=1:size(condn_data2,3)
    disp(ii)
    
    tmp = squeeze(condn_data2(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure
    %tmp = [tmp_hg tmp_beta tmp_delta];
    tmp = [tmp_hg tmp_lp];
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end





load('lstm_hand_data','D3','D3i')
condn_data3 = zeros(800,128,length(D3)+length(D3i));
k=1;
for i=1:length(D3)
    disp(k)
    tmp = D3{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
for i=1:length(D3i)
    disp(k)
    tmp = D3i{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
clear D3 D3i condn_data2


for ii=1:size(condn_data3,3)
    disp(ii)
    
    tmp = squeeze(condn_data3(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure
    %tmp = [tmp_hg tmp_beta tmp_delta];
    tmp = [tmp_hg tmp_lp];
    
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end


load('lstm_hand_data','D4','D4i')
condn_data4 = zeros(800,128,length(D4)+length(D4i));
k=1;
for i=1:length(D4)
    disp(k)
    tmp = D4{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
for i=1:length(D4i)
    disp(k)
    tmp = D4i{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
clear D4 D4i condn_data3


for ii=1:size(condn_data4,3)
    disp(ii)
    
    tmp = squeeze(condn_data4(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    
    
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure
    %tmp = [tmp_hg tmp_beta tmp_delta];
    tmp = [tmp_hg tmp_lp];
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end


load('lstm_hand_data','D5','D5i')
condn_data5 = zeros(800,128,length(D5)+length(D5i));
k=1;
for i=1:length(D5)
    disp(k)
    tmp = D5{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
for i=1:length(D5i)
    disp(k)
    tmp = D5i{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
clear D5 D5i condn_data4


for ii=1:size(condn_data5,3)
    disp(ii)
    
    tmp = squeeze(condn_data5(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    
    
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure
    %tmp = [tmp_hg tmp_beta tmp_delta];
    tmp = [tmp_hg tmp_lp];
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('lstm_hand_data','D6','D6i')
condn_data6 = zeros(800,128,length(D6)+length(D6i));
k=1;
for i=1:length(D6)
    disp(k)
    tmp = D6{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
for i=1:length(D6i)
    disp(k)
    tmp = D6i{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
clear D6 D6i condn_data5


for ii=1:size(condn_data6,3)
    disp(ii)
    
    tmp = squeeze(condn_data6(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    
    
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure
    %tmp = [tmp_hg tmp_beta tmp_delta];
    tmp = [tmp_hg tmp_lp];
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('lstm_hand_data','D7','D7i')
condn_data7 = zeros(800,128,length(D7)+length(D7i));
k=1;
for i=1:length(D7)
    disp(k)
    tmp = D7{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
for i=1:length(D7i)
    disp(k)
    tmp = D7i{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
clear D7 D7i condn_data6


for ii=1:size(condn_data7,3)
    disp(ii)
    
    tmp = squeeze(condn_data7(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    
    
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure
    %tmp = [tmp_hg tmp_beta tmp_delta];
    tmp = [tmp_hg tmp_lp];
    
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('lstm_hand_data','D8','D8i')
condn_data8 = zeros(800,128,length(D8)+length(D8i));
k=1;
for i=1:length(D8)
    disp(k)
    tmp = D8{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data8(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;8];
end
for i=1:length(D8i)
    disp(k)
    tmp = D8i{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data8(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;8];
end
clear D8 D8i condn_data7


for ii=1:size(condn_data8,3)
    disp(ii)
    
    tmp = squeeze(condn_data8(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure
    %tmp = [tmp_hg tmp_beta tmp_delta];
    tmp = [tmp_hg tmp_lp];
    
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end




load('lstm_hand_data','D9')
D9i={};
condn_data9 = zeros(800,128,length(D9)+length(D9i));
k=1;
for i=1:length(D9)
    disp(k)
    tmp = D9{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data9(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;9];
end
for i=1:length(D9i)
    disp(k)
    tmp = D9i{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data9(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;9];
end
clear D9 D9i condn_data8


for ii=1:size(condn_data9,3)
    disp(ii)
    
    tmp = squeeze(condn_data9(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure
    %tmp = [tmp_hg tmp_beta tmp_delta];
    tmp = [tmp_hg tmp_lp];
    
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('lstm_hand_data','D10')
D10i={};
condn_data10 = zeros(800,128,length(D10)+length(D10i));
k=1;
for i=1:length(D10)
    disp(k)
    tmp = D10{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data10(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;10];
end
for i=1:length(D10i)
    disp(k)
    tmp = D10i{i};
    tmp1(:,1,:)=tmp;
    %     if size(tmp,1)>2700
    %         tmp1=tmp(1:2700,:);
    %     elseif size(tmp,1)<2700
    %         len = 2700-size(tmp,1);
    %         tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    %     elseif size(tmp,1)==2700
    %         tmp1=tmp;
    %     end
    condn_data10(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;10];
end
clear D10 D10i condn_data9


for ii=1:size(condn_data10,3)
    disp(ii)
    
    tmp = squeeze(condn_data10(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure
    %tmp = [tmp_hg tmp_beta tmp_delta];
    tmp = [tmp_hg tmp_lp];
    
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

save downsampled_lstm_hand_data_all_conditions condn_data_new Y -v7.3
clear condn_data  condn_data10

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


% implement label smoothing to see how that does
%save training_data_bilstm_pooled3Feat condn_data_new Y -v7.3

%
% tmp=str2num(cell2mat(tmp));
% a=0.01;
% tmp1 = (1-a).*tmp + (a)*(1/7);
% clear YTrain
% YTrain = tmp1;
% YTrain =categorical(YTrain);
clear condn_data_new

% specify lstm structure
inputSize = 256;
numHiddenUnits1 = [  90 128 150 200 325];
drop1 = [ 0.3 0.3 0.3  0.4 0.4];
numClasses = 10;
for i=1%1:length(drop1)
    numHiddenUnits=numHiddenUnits1(i);
    drop=drop1(i);
    layers = [
        sequenceInputLayer(inputSize)
        %convolution1dLayer(12,128,'Stride',6)
        bilstmLayer(256,'OutputMode','sequence')
        dropoutLayer(drop)
        gruLayer(128,'OutputMode','last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    
    
    % options
    options = trainingOptions('adam', ...
        'MaxEpochs',20, ...
        'MiniBatchSize',64, ...
        'GradientThreshold',2, ...
        'Verbose',true, ...
        'ValidationFrequency',50,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XTest,YTest},...
        'ValidationPatience',6,...
        'Plots','training-progress',...
        'ExecutionEnvironment','parallel');
    
    % train the model
    net = trainNetwork(XTrain,YTrain,layers,options);
end
%
% net_800 =net;
% save net_800 net_800

net_bilstm_hand = net;
save net_bilstm_hand net_bilstm_hand

%% looking at the performance of the hand task

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames={'20220302'};


hand_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'HandOnline')
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        tmp=dir(filepath);
        hand_files = [hand_files;findfiles('',filepath)'];
    end
end


acc=zeros(8);
for i=1:length(hand_files)
    disp(i/length(hand_files)*100)
    load(hand_files{i})
    action = TrialData.TargetID;
    decodes = TrialData.FilteredClickerState;
    decodes = decodes(decodes~=0);
    if action >6
        action=action-2;
    end
    out=mode(decodes);
    acc(action,out)=acc(action,out)+1;
end


acc


%% Looking at real-time signals and getting higher res ERPs
% ERPs higher res

clc;clear

% filter design
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [0.5,4]; % low pass
Params.FilterBank(end+1).fpass = [4,8]; % theta
Params.FilterBank(end+1).fpass = [8,13]; % alpha
Params.FilterBank(end+1).fpass = [13,19]; % beta1
Params.FilterBank(end+1).fpass = [19,30]; % beta2
Params.FilterBank(end+1).fpass = [70,77];   % high gamma1
Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
Params.FilterBank(end+1).fpass = [20]; % raw

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

%load a block of data and filter it, with markers.
folderpath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220610\20220610\HandImagined'
D=dir(folderpath);
foldernames={};
for j=3:length(D)
    foldernames = cat(2,foldernames,D(j).name);
end

files=[];
for i=1:length(foldernames)
    filepath = fullfile(folderpath,foldernames{i},'BCI_Fixed');
    files= [files; findfiles('',filepath)'];
end

delta=[];
theta=[];
alpha=[];
beta=[];
hg=[];
raw=[];
trial_len=[];
state1_len=[];
for i=1:length(files)
    load(files{i})
    if TrialData.TargetID==7
        raw_data = cell2mat(TrialData.BroadbandData');
        trial_len=[trial_len;size(raw_data,1)];
        raw=[raw;raw_data];
        idx=find(TrialData.TaskState==1);
        tmp_state1_len = idx(end) *  (1/TrialData.Params.ScreenRefreshRate);
        tmp_state1_len = tmp_state1_len*1e3; %in ms         
        state1_len=[state1_len tmp_state1_len];
    end
end
trial_len_total=cumsum(trial_len);

% extracting band specific information
delta = filter(Params.FilterBank(1).b,...
    Params.FilterBank(1).a,...
    raw);
%delta=abs(hilbert(delta));

theta = filter(Params.FilterBank(2).b,...
    Params.FilterBank(2).a,...
    raw);

alpha = filter(Params.FilterBank(3).b,...
    Params.FilterBank(3).a,...
    raw);

beta1 = filter(Params.FilterBank(4).b,...
    Params.FilterBank(4).a,...
    raw);
beta2 = filter(Params.FilterBank(5).b,...
    Params.FilterBank(5).a,...
    raw);
%beta1=(beta1.^2);
%beta2=(beta2.^2);
%beta = log10((beta1+beta2)/2);
beta = (abs(hilbert(beta1)) + abs(hilbert(beta2)))/2;

% hg filter bank approach -> square samples, log 10 and then average across
% bands
hg_bank=[];
for i=6:length(Params.FilterBank)-1
    tmp = filter(Params.FilterBank(i).b,...
        Params.FilterBank(i).a,...
        raw);
    %tmp=tmp.^2;
    tmp=abs(hilbert(tmp));
    hg_bank = cat(3,hg_bank,tmp);
end
%hg = log10(squeeze(mean(hg_bank,3)));
hg = (squeeze(mean(hg_bank,3)));

% lpf the raw
raw = filter(Params.FilterBank(end).b,...
    Params.FilterBank(end).a,...
    raw);
for j=1:size(raw,2)
    raw(:,j) = smooth(raw(:,j),100);
end

% now going and referencing each trial to state 1 data
raw_ep={};
delta_ep={};
beta_ep={};
hg_ep={};
trial_len_total=[0 ;trial_len_total];
for i=1:length(trial_len_total)-1
    % raw
    tmp_data = raw(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    raw_ep = cat(2,raw_ep,tmp_data);

    %delta
    tmp_data = delta(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    delta_ep = cat(2,delta_ep,tmp_data);

    %beta
    tmp_data = beta(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    beta_ep = cat(2,beta_ep,tmp_data);

    %hg
    tmp_data = hg(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    hg_ep = cat(2,hg_ep,tmp_data);
end


figure;
temp=cell2mat(hg_ep');
plot(temp(:,3));
axis tight
vline(trial_len_total,'r')


%
% figure;
% subplot(4,1,1)
% plot(raw(:,3))
% vline(trial_len,'r')
% title('raw')
% axis tight
%
% subplot(4,1,2)
% plot(abs(hilbert(delta(:,3))))
% vline(trial_len,'r')
% title('delta')
% axis tight
%
% subplot(4,1,3)
% plot(beta(:,3))
% vline(trial_len,'r')
% title('beta')
% axis tight
%
% subplot(4,1,4)
% plot(hg(:,4))
% vline(trial_len,'r')
% title('hg')
% axis tight
%
% sgtitle('Target 1, Ch3')
% set(gcf,'Color','w')

%hg erps - take the frst 13k time points
hg_data=[];
for i=1:length(hg_ep)
    tmp=hg_ep{i};
    hg_data = cat(3,hg_data,tmp(1:13000,:));
end

figure;
subplot(2,1,1)
ch=31;
plot(squeeze(hg_data(:,ch,:)),'Color',[.5 .5 .5 .5])
hold on
plot(squeeze(mean(hg_data(:,ch,:),3)),'Color','b','LineWidth',2)
vline(TrialData.Params.InstructedDelayTime*1e3,'r')
vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
vline(TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3 + ...
    [TrialData.Params.CycleDuration*[1:TrialData.Params.NumCycles-1]]*1e3,'g')
vline(TrialData.Params.CycleDuration*3*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3, 'k')
hline(0)
hline(0)
set(gcf,'Color','w')
xlabel('Time in ms')
ylabel('uV')
title('Hg Ch 106 ')
set(gca,'LineWidth',1)
set(gca,'FontSize',14)
axis tight
xlim([0 13000])
ylim([-5 10])

%delta erps - take the frst 7800 time points
delta_data=[];
for i=1:length(raw_ep)
    tmp=raw_ep{i};
%     for j=1:size(tmp,2)
%         tmp(:,j)=smooth(tmp(:,j),100);
%     end
    delta_data = cat(3,delta_data,tmp(1:13000,:));
end

%figure;
subplot(2,1,2)
ch=31;
plot(squeeze(delta_data(:,ch,:)),'Color',[.5 .5 .5 .5])
hold on
plot(squeeze(mean(delta_data(:,ch,:),3)),'Color','b','LineWidth',2)
vline(TrialData.Params.InstructedDelayTime*1e3,'r')
vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
vline(TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3 + ...
    [TrialData.Params.CycleDuration*[1:TrialData.Params.NumCycles-1]]*1e3,'g')
vline(TrialData.Params.CycleDuration*3*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3, 'k')
hline(0)
hline(0)
set(gcf,'Color','w')
xlabel('Time in ms')
ylabel('uV')
title('Raw Ch 106 ')
set(gca,'LineWidth',1)
set(gca,'FontSize',14)
axis tight
xlim([0 13000])
ylim([-5 5])
sgtitle('Thumb')

% 
% ch=106;
% tmp=squeeze(delta_data(:,ch,:));
% for i=1:size(tmp,2)
%     figure;plot(tmp(:,i))
%     vline(TrialData.Params.InstructedDelayTime*1e3,'r')
%     vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
%     vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
%         +  TrialData.Params.CueTime*1e3, 'k')
%     hline(0)
% end
%  ylim([-3 3])


% % plotting covariance of raw
% tmp=zscore(raw);
% figure;imagesc(cov(raw))
% [c,s,l]=pca((raw));
% chmap=TrialData.Params.ChMap;
% tmp1=c(:,1);
% figure;imagesc(tmp1(chmap))
% figure;
% stem(cumsum(l)./sum(l))


%% Looking at real-time signals and getting higher res ERPs
% ERPs higher res
% with traveling waves 

clc;clear

% filter design
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [0.5,4]; % low pass
Params.FilterBank(end+1).fpass = [4,8]; % theta
Params.FilterBank(end+1).fpass = [8,13]; % alpha
Params.FilterBank(end+1).fpass = [13,19]; % beta1
Params.FilterBank(end+1).fpass = [19,30]; % beta2
Params.FilterBank(end+1).fpass = [70,77];   % high gamma1
Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
Params.FilterBank(end+1).fpass = [20]; % raw

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

%load a block of data and filter it, with markers.
folderpath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220601\Robot3DArrow'
foldernames = {'105544','110134','110453','110817'}
files=[];
for i=1:length(foldernames)
    filepath = fullfile(folderpath,foldernames{i},'Imagined');
    files= [files; findfiles('',filepath)'];
end

delta=[];
theta=[];
alpha=[];
beta=[];
hg=[];
raw=[];
trial_len=[];
state1_len=[];
for i=1:length(files)
    load(files{i})
    if TrialData.TargetID==1
        raw_data = cell2mat(TrialData.BroadbandData');
        trial_len=[trial_len;size(raw_data,1)];
        raw=[raw;raw_data];
        idx=find(TrialData.TaskState==1);
        tmp=cell2mat(TrialData.BroadbandData(idx)');
        state1_len=[state1_len size(tmp,1)];
    end
end
trial_len_total=cumsum(trial_len);

% extracting band specific information
delta = filter(Params.FilterBank(1).b,...
    Params.FilterBank(1).a,...
    raw);
delta=abs(hilbert(delta));

theta = filter(Params.FilterBank(2).b,...
    Params.FilterBank(2).a,...
    raw);

alpha = filter(Params.FilterBank(3).b,...
    Params.FilterBank(3).a,...
    raw);

beta1 = filter(Params.FilterBank(4).b,...
    Params.FilterBank(4).a,...
    raw);
beta2 = filter(Params.FilterBank(5).b,...
    Params.FilterBank(5).a,...
    raw);
%beta1=(beta1.^2);
%beta2=(beta2.^2);
%beta = log10((beta1+beta2)/2);
beta = (abs(hilbert(beta1)) + abs(hilbert(beta2)))/2;

% hg filter bank approach -> square samples, log 10 and then average across
% bands
hg_bank=[];
for i=6:length(Params.FilterBank)-1
    tmp = filter(Params.FilterBank(i).b,...
        Params.FilterBank(i).a,...
        raw);
    %tmp=tmp.^2;
    tmp=abs(hilbert(tmp));
    hg_bank = cat(3,hg_bank,tmp);
end
%hg = log10(squeeze(mean(hg_bank,3)));
hg = (squeeze(mean(hg_bank,3)));

% lpf the raw
raw = filter(Params.FilterBank(end).b,...
    Params.FilterBank(end).a,...
    raw);
for j=1:size(raw,2)
    raw(:,j) = smooth(raw(:,j),100);
end

% now going and referencing each trial to state 1 data
raw_ep={};
delta_ep={};
beta_ep={};
hg_ep={};
trial_len_total=[0 ;trial_len_total];
for i=1:length(trial_len_total)-1
    % raw
    tmp_data = raw(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    raw_ep = cat(2,raw_ep,tmp_data);

    %delta
    tmp_data = delta(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    delta_ep = cat(2,delta_ep,tmp_data);

    %beta
    tmp_data = beta(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    beta_ep = cat(2,beta_ep,tmp_data);

    %hg
    tmp_data = hg(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    hg_ep = cat(2,hg_ep,tmp_data);
end


figure;
temp=cell2mat(hg_ep');
plot(temp(:,3));
axis tight
vline(trial_len_total,'r')


%
% figure;
% subplot(4,1,1)
% plot(raw(:,3))
% vline(trial_len,'r')
% title('raw')
% axis tight
%
% subplot(4,1,2)
% plot(abs(hilbert(delta(:,3))))
% vline(trial_len,'r')
% title('delta')
% axis tight
%
% subplot(4,1,3)
% plot(beta(:,3))
% vline(trial_len,'r')
% title('beta')
% axis tight
%
% subplot(4,1,4)
% plot(hg(:,4))
% vline(trial_len,'r')
% title('hg')
% axis tight
%
% sgtitle('Target 1, Ch3')
% set(gcf,'Color','w')

%hg erps - take the frst 7800 time points
hg_data=[];
for i=1:length(hg_ep)
    tmp=hg_ep{i};
    hg_data = cat(3,hg_data,tmp(1:7800,:));
end

figure;
subplot(2,1,1)
ch=106;
plot(squeeze(hg_data(:,ch,:)),'Color',[.5 .5 .5 .5])
hold on
plot(squeeze(mean(hg_data(:,ch,:),3)),'Color','b','LineWidth',2)
vline(TrialData.Params.InstructedDelayTime*1e3,'r')
vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3, 'k')
hline(0)
hline(0)
set(gcf,'Color','w')
xlabel('Time in ms')
ylabel('uV')
title('Hg Ch 106 Left Leg')
set(gca,'LineWidth',1)
set(gca,'FontSize',14)
axis tight
ylim([-5 10])

%delta erps - take the frst 7800 time points
delta_data=[];
for i=1:length(delta_ep)
    tmp=delta_ep{i};
%     for j=1:size(tmp,2)
%         tmp(:,j)=smooth(tmp(:,j),100);
%     end
    delta_data = cat(3,delta_data,tmp(1:7800,:));
end

%figure;
subplot(2,1,2)
ch=106;
plot(squeeze(delta_data(:,ch,:)),'Color',[.5 .5 .5 .5])
hold on
plot(squeeze(mean(delta_data(:,ch,:),3)),'Color','b','LineWidth',2)
vline(TrialData.Params.InstructedDelayTime*1e3,'r')
vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3, 'k')
hline(0)
set(gcf,'Color','w')
xlabel('Time in ms')
ylabel('uV')
title('Raw Ch 106 Left Leg')
set(gca,'LineWidth',1)
set(gca,'FontSize',14)
axis tight
ylim([-5 5])

% 
% ch=106;
% tmp=squeeze(delta_data(:,ch,:));
% for i=1:size(tmp,2)
%     figure;plot(tmp(:,i))
%     vline(TrialData.Params.InstructedDelayTime*1e3,'r')
%     vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
%     vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
%         +  TrialData.Params.CueTime*1e3, 'k')
%     hline(0)
% end
%  ylim([-3 3])


% plotting covariance of raw
tmp=zscore(raw);
figure;imagesc(cov(raw))
[c,s,l]=pca((raw));
chmap=TrialData.Params.ChMap;
tmp1=c(:,1);
figure;imagesc(tmp1(chmap))
figure;
stem(cumsum(l)./sum(l))


%% NEW HAND DATA MULTI CYCLIC
% extract single trials 
% get the time-freq features in raw and in hG
% train a bi-GRU


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20220608','20220610','20220622','20220624'};
cd(root_path)

imagined_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'HandImagined')
    D=dir(folderpath);
    
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        if ~exist(filepath)
            filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        end
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
D9i={};
D10i={};
D11i={};
D12i={};
D13i={};
D14i={};
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
        action = TrialData.TargetID;
        %disp(action)

        % find the bins when state 3 happened and then extract each
        % individual cycle (2.6s length) as a trial
        
        % get times for state 3 from the sample rate of screen refresh
        time  = TrialData.Time;
        time = time - time(1);        
        idx = find(TrialData.TaskState==3) ;
        task_time = time(idx);     

        % get the kinematics and extract times in state 3 when trials
        % started and ended
        kin = TrialData.CursorState;
        kin=kin(1,idx);
        kind = [0 diff(kin)];
        aa=find(kind==0);
        kin_st=[];
        kin_stp=[];
        for j=1:length(aa)-1
            if (aa(j+1)-aa(j))>1
                kin_st = [kin_st aa(j)];
                kin_stp = [kin_stp aa(j+1)-1];
            end
        end

        %getting start and stop times
        start_time = task_time(kin_st);
        stp_time = task_time(kin_stp);
        

        % get corresponding neural times indices
%         neural_time  = TrialData.NeuralTime;
%         neural_time = neural_time-neural_time(1);
%         neural_st=[];
%         neural_stp=[];
%         st_time_neural=[];
%         stp_time_neural=[];
%         for j=1:length(start_time)
%             [aa bb]=min(abs(neural_time-start_time(j)));
%             neural_st = [neural_st; bb];
%             st_time_neural = [st_time_neural;neural_time(bb)];
%             [aa bb]=min(abs(neural_time-stp_time(j)));
%             neural_stp = [neural_stp; bb-1];
%             stp_time_neural = [stp_time_neural;neural_time(bb)];
%         end

        % get the broadband data for each trial
        raw_data=cell2mat(TrialData.BroadbandData');

        % extract the broadband data (Fs-1KhZ) based on rough estimate of
        % the start and stop times from the kinematic data
        start_time_neural = round(start_time*1e3);
        stop_time_neural = round(stp_time*1e3);
        data_seg={};
        for j=1:length(start_time_neural)
            tmp = (raw_data(start_time_neural(j):stop_time_neural(j),:));   
            tmp=tmp(1:round(size(tmp,1)/2),:);
            % pca step
            %m=mean(tmp);
            %[c,s,l]=pca(tmp,'centered','off');
            %tmp = (s(:,1)*c(:,1)')+m;
            data_seg = cat(2,data_seg,tmp);            
        end
        
        if action==1
            D1i = cat(2,D1i,data_seg);        
        elseif action==2
            D2i = cat(2,D2i,data_seg);            
        elseif action==3
            D3i = cat(2,D3i,data_seg);            
        elseif action==4
            D4i = cat(2,D4i,data_seg);            
        elseif action==5
            D5i = cat(2,D5i,data_seg);            
        elseif action==6
            D6i = cat(2,D6i,data_seg);
        elseif action==7
            D7i = cat(2,D7i,data_seg);
        elseif action==8
            D8i = cat(2,D8i,data_seg);
        elseif action==9
            D9i = cat(2,D9i,data_seg);
        elseif action==10
            D10i = cat(2,D10i,data_seg);
        elseif action==11
            D11i = cat(2,D11i,data_seg);
        elseif action==12
            D12i = cat(2,D12i,data_seg);
        elseif action==13
            D13i = cat(2,D13i,data_seg);
        elseif action==14
            D14i = cat(2,D14i,data_seg);
        end
    end
end


%%%%% looking at PCA 
%take any action
figure;
subplot(2,7,1)
temp=cell2mat(D1i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Thumb')
axis off
set(gcf,'Color','w')

subplot(2,7,2)
temp=cell2mat(D2i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Index')
axis off
set(gcf,'Color','w')

subplot(2,7,3)
temp=cell2mat(D3i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Middle')
axis off
set(gcf,'Color','w')

subplot(2,7,4)
temp=cell2mat(D4i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Ring')
axis off
set(gcf,'Color','w')

subplot(2,7,5)
temp=cell2mat(D5i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Pinky')
axis off
set(gcf,'Color','w')

subplot(2,7,6)
temp=cell2mat(D6i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Power')
axis off
set(gcf,'Color','w')

subplot(2,7,7)
temp=cell2mat(D7i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Pinch')
axis off
set(gcf,'Color','w')

subplot(2,7,8)
temp=cell2mat(D8i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Tripod')
axis off
set(gcf,'Color','w')

subplot(2,7,9)
temp=cell2mat(D9i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Wrist')
axis off
set(gcf,'Color','w')

subplot(2,7,10)
temp=cell2mat(D10i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Middle')
axis off
set(gcf,'Color','w')

subplot(2,7,3)
temp=cell2mat(D3i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Middle')
axis off
set(gcf,'Color','w')

subplot(2,7,3)
temp=cell2mat(D3i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Middle')
axis off
set(gcf,'Color','w')

subplot(2,7,3)
temp=cell2mat(D3i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Middle')
axis off
set(gcf,'Color','w')

subplot(2,7,3)
temp=cell2mat(D3i');
[c,s,l]=pca(temp);
chmap=TrialData.Params.ChMap;
tmp=c(:,1);
imagesc(tmp(chmap))
title('Middle')
axis off
set(gcf,'Color','w')



%%%% downsampling extracting hG and LMP and then running it through a bi-GRU

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
Params.FilterBank(end+1).fpass = [30]; % LFP
Params.FilterBank(end+1).fpass = [4,8]; % theta
%Params.FilterBank(end+1).fpass = [13,19]; % beta1
%Params.FilterBank(end+1).fpass = [19,30]; % beta2
%Params.FilterBank(end+1).fpass = [70,150]; % raw_hg

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

% process and store the data in a giant structure, with single trials 
Data={};
Y=[];

%%%% D1%%%%%%%
Y = [Y; 1*ones(length(D1i),1)];
for ii=1:length(D1i)
    disp(ii)
    
    tmp = D1i{ii};
    tmp_processed = preprocess_bilstm(tmp,Params);    
    
    % store
    Data = cat(2,Data,tmp_processed');
end
%%%% D1 END %%%%


%%%% D2%%%%%%%
Y = [Y; 2*ones(length(D2i),1)];
for ii=1:length(D2i)
    disp(ii)
    
    tmp = D2i{ii};
    tmp_processed = preprocess_bilstm(tmp,Params);    
    
    % store
    Data = cat(2,Data,tmp_processed');
end
%%%% D2 END %%%%

%%%% D3%%%%%%%
Y = [Y; 3*ones(length(D3i),1)];
for ii=1:length(D3i)
    disp(ii)
    
    tmp = D3i{ii};
    tmp_processed = preprocess_bilstm(tmp,Params);    
    
    % store
    Data = cat(2,Data,tmp_processed');
end
%%%% D3 END %%%%

%%%% D4%%%%%%%
Y = [Y; 4*ones(length(D4i),1)];
for ii=1:length(D4i)
    disp(ii)
    
    tmp = D4i{ii};
    tmp_processed = preprocess_bilstm(tmp,Params);    
    
    % store
    Data = cat(2,Data,tmp_processed');
end
%%%% D4 END %%%%


%%%% D5%%%%%%%
Y = [Y; 5*ones(length(D5i),1)];
for ii=1:length(D5i)
    disp(ii)
    
    tmp = D5i{ii};
    tmp_processed = preprocess_bilstm(tmp,Params);    
    
    % store
    Data = cat(2,Data,tmp_processed');
end
%%%% D5 END %%%%



% set aside training and testing data in a cell format
idx = randperm(length(Data),round(0.8*length(Data)));
I = zeros(length(Data),1);
I(idx)=1;
test_idx = find(I==0);

XTrain={};
XTest={};
YTrain=[];
YTest=[];

XTrain  = Data(logical(I))';
YTrain = categorical(Y(logical(I)));
XTest = Data(test_idx)';
YTest = categorical(Y(test_idx));

% specify lstm structure
inputSize = 256;
numHiddenUnits1 = [  96 128 150 200 325];
drop1 = [ 0.3 0.3 0.3  0.4 0.4];
numClasses = 5;
for i=1%1:length(drop1)
    numHiddenUnits=numHiddenUnits1(i);
    drop=drop1(i);
    layers = [        
        sequenceInputLayer(inputSize)        
        %convolution1dLayer(5,128,'Stride',3)
        %reluLayer
        bilstmLayer(numHiddenUnits,'OutputMode','sequence')
        %dropoutLayer(drop)
        gruLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(drop)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    
    
    % options
    options = trainingOptions('adam', ...
        'MaxEpochs',200, ...
        'MiniBatchSize',32, ...
        'GradientThreshold',5, ...
        'Verbose',true, ...
        'ValidationFrequency',32,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XTest,YTest},...
        'ValidationPatience',50,...
        'Plots','training-progress');
    
    % train the model
    net = trainNetwork(XTrain,YTrain,layers,options);
end




% 
% 
% % movie of theta activity after filtering in theta band
% chmap = TrialData.Params.ChMap;
% figure;
% tmp_theta=Data{158}';
% for i=1:size(tmp_theta,1)
%     t  = tmp_theta(i,:);
%     imagesc(t(chmap));
%     colormap bone
%     title(num2str(i))
%     pause(0.05)    
% end
% 








