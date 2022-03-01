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
    'PassbandFrequency',30,'PassbandRipple',0.2, ...
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
load('lstm_hand_data_ONLYHANDTASK','D1','D1i')
condn_data1 = zeros(2700,128,length(D1)+length(D1i));
k=1;
for i=1:length(D1)
    disp(k)
    tmp = D1{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
for i=1:length(D1i)
    disp(k)
    tmp = D1i{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
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
    tmp_lp = resample(tmp_lp,1500,2700);
    tmp_hg = resample(tmp_hg,1500,2700)*5e2;
    
    
    
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

load('lstm_hand_data_ONLYHANDTASK','D2','D2i')
condn_data2 = zeros(2700,128,length(D2)+length(D2i));
k=1;
for i=1:length(D2)
    disp(k)
    tmp = D2{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
for i=1:length(D2i)
    disp(k)
    tmp = D2i{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
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
    tmp_lp = resample(tmp_lp,1500,2700);
    tmp_hg = resample(tmp_hg,1500,2700)*5e2;
    
    
    
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





load('lstm_hand_data_ONLYHANDTASK','D3','D3i')
condn_data3 = zeros(2700,128,length(D3)+length(D3i));
k=1;
for i=1:length(D3)
    disp(k)
    tmp = D3{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
for i=1:length(D3i)
    disp(k)
    tmp = D3i{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
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
    tmp_lp = resample(tmp_lp,1500,2700);
    tmp_hg = resample(tmp_hg,1500,2700)*5e2;
    
 
    
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


load('lstm_hand_data_ONLYHANDTASK','D4','D4i')
condn_data4 = zeros(2700,128,length(D4)+length(D4i));
k=1;
for i=1:length(D4)
    disp(k)
    tmp = D4{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
for i=1:length(D4i)
    disp(k)
    tmp = D4i{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
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
    tmp_lp = resample(tmp_lp,1500,2700);
    tmp_hg = resample(tmp_hg,1500,2700)*5e2;
    
    
   
    
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


load('lstm_hand_data_ONLYHANDTASK','D5','D5i')
condn_data5 = zeros(2700,128,length(D5)+length(D5i));
k=1;
for i=1:length(D5)
    disp(k)
    tmp = D5{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
for i=1:length(D5i)
    disp(k)
    tmp = D5i{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
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
    tmp_lp = resample(tmp_lp,1500,2700);
    tmp_hg = resample(tmp_hg,1500,2700)*5e2;
    
    
  
    
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



load('lstm_hand_data_ONLYHANDTASK','D6','D6i')
condn_data6 = zeros(2700,128,length(D6)+length(D6i));
k=1;
for i=1:length(D6)
    disp(k)
    tmp = D6{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
for i=1:length(D6i)
    disp(k)
    tmp = D6i{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
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
    tmp_lp = resample(tmp_lp,1500,2700);
    tmp_hg = resample(tmp_hg,1500,2700)*5e2;
    
    
   
    
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



load('lstm_hand_data_ONLYHANDTASK','D7','D7i')
condn_data7 = zeros(2700,128,length(D7)+length(D7i));
k=1;
for i=1:length(D7)
    disp(k)
    tmp = D7{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
for i=1:length(D7i)
    disp(k)
    tmp = D7i{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
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
    tmp_lp = resample(tmp_lp,1500,2700);
    tmp_hg = resample(tmp_hg,1500,2700)*5e2;
    
    

    
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



load('lstm_hand_data_ONLYHANDTASK','D8','D8i')
condn_data8 = zeros(2700,128,length(D8)+length(D8i));
k=1;
for i=1:length(D8)
    disp(k)
    tmp = D8{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
    condn_data8(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;8];
end
for i=1:length(D8i)
    disp(k)
    tmp = D8i{i};
    %tmp1(:,1,:)=tmp;
    if size(tmp,1)>2700
        tmp1=tmp(1:2700,:);
    elseif size(tmp,1)<2700
        len = 2700-size(tmp,1);
        tmp(end+1:end+len,:)=  repmat(tmp(end,:),len,1);
    elseif size(tmp,1)==2700
        tmp1=tmp;
    end
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
    tmp_lp = resample(tmp_lp,1500,2700);
    tmp_hg = resample(tmp_hg,1500,2700)*5e2;
    
  
    
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


save downsampled_lstm_hand_data_ONLYHANDTASK condn_data_new Y -v7.3
clear condn_data  condn_data8

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
numClasses = 8;
for i=1%1:length(drop1)
    numHiddenUnits=numHiddenUnits1(i);
    drop=drop1(i);
    layers = [       
        sequenceInputLayer(inputSize)        
        %convolution1dLayer(12,128,'Stride',6)
        bilstmLayer(numHiddenUnits,'OutputMode','sequence')
        dropoutLayer(drop)
        gruLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(drop)
        batchNormalizationLayer
        fullyConnectedLayer(40)
        reluLayer
        dropoutLayer(.3)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    
    
    % options
    options = trainingOptions('adam', ...
        'MaxEpochs',20, ...
        'MiniBatchSize',64, ...
        'GradientThreshold',2, ...
        'Verbose',true, ...
        'ValidationFrequency',10,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XTest,YTest},...
        'ValidationPatience',6,...
        'Plots','training-progress');
    
    % train the model
    net = trainNetwork(XTrain,YTrain,layers,options);
end
%
% net_800 =net;
% save net_800 net_800

net_bilstm_hand = net;
save net_bilstm_hand net_bilstm_hand
