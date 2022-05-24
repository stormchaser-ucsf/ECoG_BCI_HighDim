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






%% BUILDIN THE DECODER USIGN IMAGINED PLUS ONLINE DATA

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
load('lstm_hand_data','D1','D1i')
condn_data1 = zeros(800,128,length(D1)+length(D1i));
k=1;
for i=1:length(D1)
    disp(k)
    tmp = D1{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
for i=1:length(D1i)
    disp(k)
    tmp = D1i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
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
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data8(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;8];
end
for i=1:length(D8i)
    disp(k)
    tmp = D8i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
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


save downsampled_lstm_hand_data condn_data_new Y -v7.3
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
drop1 = [ 0.2 0.3 0.3  0.4 0.4];
numClasses = 8;
for i=3%1:length(drop1)
    numHiddenUnits=numHiddenUnits1(i);
    drop=drop1(i);
    layers = [ ...
        sequenceInputLayer(inputSize)
        bilstmLayer(numHiddenUnits,'OutputMode','sequence')
        dropoutLayer(drop)
        gruLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(drop)
        batchNormalizationLayer
        fullyConnectedLayer(40)
        reluLayer
        dropoutLayer(.2)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    
    
    % options
    options = trainingOptions('adam', ...
        'MaxEpochs',15, ...
        'MiniBatchSize',128, ...
        'GradientThreshold',2, ...
        'Verbose',true, ...
        'ValidationFrequency',30,...
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


%% TESTING THE DATA ON ONLINE DATA
clc;clear
load net_bilstm_hand
filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220218\Hand';

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


% filter bank hg
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
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',30,'PassbandRipple',0.2, ...
    'SampleRate',1e3);


% load the data, and run it through the classifier
decodes_overall=[];k=1;
for i=1:length(files)
    disp(i)
    
    % load
    load(files{i})
    
    if TrialData.TargetID<=8
        
        % create buffer
        data_buffer = randn(800,128)*0.25;
        
        %get data
        raw_data = TrialData.BroadbandData;
        raw_data1=cell2mat(raw_data');
        
        % state of trial
        state_idx = TrialData.TaskState;
        decodes=[];
        for j=1:length(raw_data)
            tmp = raw_data{j};
            s=size(tmp,1);
            if s<800
                data_buffer = circshift(data_buffer,-s);
                data_buffer(end-s+1:end,:)=tmp;
            else
                data_buffer(1:end,:)=tmp(s-800+1:end,:);
            end
            
            %hg features
            filtered_data=zeros(size(data_buffer,1),size(data_buffer,2),8);
            for ii=1:length(Params.FilterBank)
                filtered_data(:,:,ii) =  ((filter(...
                    Params.FilterBank(ii).b, ...
                    Params.FilterBank(ii).a, ...
                    data_buffer)));
            end
            tmp_hg = squeeze(mean(filtered_data.^2,3));
            tmp_hg = resample(tmp_hg,200,800)*5e2;
            
            
            
            % low-pass features
            tmp_lp = filter(lpFilt,data_buffer);
            tmp_lp = resample(tmp_lp,200,800);
            
            % concatenate
            neural_features = [tmp_hg tmp_lp];
            
            % classifier output
            out=predict(net_bilstm_hand,neural_features');
            [aa bb]=max(out);
            class_predict = bb;
            
            % store results
            if state_idx(j)==3
                decodes=[decodes class_predict];
            end
        end
        decodes_overall(k).decodes = decodes;
        decodes_overall(k).tid = TrialData.TargetID;
        k=k+1;
    end
end

% looking at the accuracy of the bilstm decoder overall
acc=zeros(8,8);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    for j=1:length(tmp)
        acc(tid,tmp(j)) =  acc(tid,tmp(j))+1;
    end
end
for i=1:length(acc)
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

% looking at accuracy in terms of max decodes
acc_trial=zeros(8,8);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    acc1=zeros(8,8);
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


%% LSTM FINE TUNING APPROACH
% update the model weights on 90% of trials and test on held out 10% of
% trials using CV split


clc;clear
load net_bilstm_hand


% filter bank hg
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
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',30,'PassbandRipple',0.2, ...
    'SampleRate',1e3);


% get the trials
filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220218\Hand';

% get the name of the files
files = findfiles('mat',filepath,1)';
files1=[];
for i=1:length(files)
    if ~isempty(regexp(files{i},'Data0001')) || ~isempty(regexp(files{i},'Data0002')) ||...
            ~isempty(regexp(files{i},'Data0003')) || ~isempty(regexp(files{i},'Data0004'))||...
            ~isempty(regexp(files{i},'Data0005')) || ~isempty(regexp(files{i},'Data0006'))||...
            ~isempty(regexp(files{i},'Data0007')) || ~isempty(regexp(files{i},'Data0008'))
        files1=[files1;files(i)];
    end
end
files=files1;
clear files1

% set aside 90% of trials for training and 10% for testing
idx = randperm(length(files),round(0.8*length(files)));
I=zeros(length(files),1);
I(idx)=1;
train_idx = find(I==1);
test_idx = find(I==0);
train_files=files(train_idx);
test_files=files(test_idx);

% get neural features for training trials
D1={};
D2={};
D3={};
D4={};
D5={};
D6={};
D7={};
D8={};
for i=1:length(train_files)
    disp(i/length(train_files)*100)
    try
        load(train_files{i})
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
D1i={};
D2i={};
D3i={};
D4i={};
D5i={};
D6i={};
D7i={};
D8i={};
Y=[];
condn_data_new=[];jj=1;

condn_data1 = zeros(800,128,length(D1)+length(D1i));
k=1;
for i=1:length(D1)
    disp(k)
    tmp = D1{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
for i=1:length(D1i)
    disp(k)
    tmp = D1i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end


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

condn_data2 = zeros(800,128,length(D2)+length(D2i));
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

condn_data3 = zeros(800,128,length(D3)+length(D3i));
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

condn_data4 = zeros(800,128,length(D4)+length(D4i));
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

condn_data5 = zeros(800,128,length(D5)+length(D5i));
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


condn_data6 = zeros(800,128,length(D6)+length(D6i));
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

condn_data7 = zeros(800,128,length(D7)+length(D7i));
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

condn_data8 = zeros(800,128,length(D8)+length(D8i));
k=1;
for i=1:length(D8)
    disp(k)
    tmp = D8{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data8(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;8];
end
for i=1:length(D8i)
    disp(k)
    tmp = D8i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data8(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;8];
end

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

%%%%%% update LSTM weights

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

% options
options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize',64, ...
    'GradientThreshold',2, ...
    'Verbose',true, ...
    'ValidationFrequency',10,...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest,YTest},...
    'ValidationPatience',5,...
    'Plots','training-progress');

% train the model
layers = net_bilstm_hand.Layers;
net_bilstm_hand_FT = trainNetwork(XTrain,YTrain,layers,options);



%%%% get neural features for testing and test LSTM

% load the data, and run it through the classifier
decodes_overall=[];k=1;
for i=1:length(test_files)
    disp(i)
    
    % load
    load(test_files{i})
    
    if TrialData.TargetID<=8
        
        % create buffer
        data_buffer = randn(800,128)*0.25;
        
        %get data
        raw_data = TrialData.BroadbandData;
        raw_data1=cell2mat(raw_data');
        
        % state of trial
        state_idx = TrialData.TaskState;
        decodes=[];
        for j=1:length(raw_data)
            tmp = raw_data{j};
            s=size(tmp,1);
            if s<800
                data_buffer = circshift(data_buffer,-s);
                data_buffer(end-s+1:end,:)=tmp;
            else
                data_buffer(1:end,:)=tmp(s-800+1:end,:);
            end
            
            %hg features
            filtered_data=zeros(size(data_buffer,1),size(data_buffer,2),8);
            for ii=1:length(Params.FilterBank)
                filtered_data(:,:,ii) =  ((filter(...
                    Params.FilterBank(ii).b, ...
                    Params.FilterBank(ii).a, ...
                    data_buffer)));
            end
            tmp_hg = squeeze(mean(filtered_data.^2,3));
            tmp_hg = resample(tmp_hg,200,800)*5e2;
            
            
            
            % low-pass features
            tmp_lp = filter(lpFilt,data_buffer);
            tmp_lp = resample(tmp_lp,200,800);
            
            % concatenate
            neural_features = [tmp_hg tmp_lp];
            
            % classifier output
            out=predict(net_bilstm_hand_FT,neural_features');
            [aa bb]=max(out);
            class_predict = bb;
            
            % store results
            if state_idx(j)==3
                decodes=[decodes class_predict];
            end
        end
        decodes_overall(k).decodes = decodes;
        decodes_overall(k).tid = TrialData.TargetID;
        k=k+1;
    end
end

% looking at the accuracy of the bilstm decoder overall
acc=zeros(8,8);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    for j=1:length(tmp)
        acc(tid,tmp(j)) =  acc(tid,tmp(j))+1;
    end
end
for i=1:length(acc)
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

% looking at accuracy in terms of max decodes
acc_trial=zeros(8,8);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    acc1=zeros(8,8);
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




%% BUILDING MLP FOR THE HAND MODEL

clc;clear
close all

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

% GETTING DATA FROM THE HAND TASK, all but the last day's data
foldernames = {'20220128','20220204','20220209','20220218','20220223','20220302'};
cd(root_path)

hand_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Hand')
    %     if ~exist(folderpath)
    %         folderpath = fullfile(root_path, foldernames{i},'HandOnline')
    %     end
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        tmp=dir(filepath);
        hand_files = [hand_files;findfiles('',filepath)'];
    end
end


% load the data for the imagined files, if they belong to right thumb,
% index, middle, ring, pinky, pinch, tripod, power
D1=[];%thumb
D2=[];%index
D3=[];%middle
D4=[];%ring
D5=[];%pinky
D6=[];%power
D7=[];%pinch
D8=[];%tripod
D9=[];%wrist out
D10=[];%wrist in



for i=1:length(hand_files)
    disp(i/length(hand_files)*100)
    load(hand_files{i})
    
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    kinax = find(kinax==3);
    temp = cell2mat(features(kinax));
    
    if regexp(hand_files{i},'20220302')
        len = size(temp,2);
        if len>20
            temp=temp(:,1:20);
        end
    end
    
    % get smoothed delta hg and beta features
    new_temp=[];
    [xx yy] = size(TrialData.Params.ChMap);
    for k=1:size(temp,2)
        tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
        tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
        tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
        pooled_data=[];
        for i=1:2:xx
            for j=1:2:yy
                delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                pooled_data = [pooled_data; delta; beta ;hg];
            end
        end
        new_temp= [new_temp pooled_data];
    end
    temp=new_temp;
    
    
    if TrialData.TargetID == 1
        D1 = [D1 temp];
    elseif TrialData.TargetID == 2
        D2 = [D2 temp];
    elseif TrialData.TargetID == 3
        D3 = [D3 temp];
    elseif TrialData.TargetID == 4
        D4 = [D4 temp];
    elseif TrialData.TargetID == 5
        D5 = [D5 temp];
    elseif TrialData.TargetID == 6
        D6 = [D6 temp];
    elseif TrialData.TargetID == 7
        D7 = [D7 temp];
    elseif TrialData.TargetID == 8
        D8 = [D8 temp];
    elseif TrialData.TargetID == 9
        D9 = [D9 temp];
    elseif TrialData.TargetID == 10
        D10 = [D10 temp];
    end
end


idx = [1:96];
condn_data{1}=[D1(idx,:) ]';
condn_data{2}= [D2(idx,:)]';
condn_data{3}=[D3(idx,:)]';
condn_data{4}=[D4(idx,:)]';
condn_data{5}=[D5(idx,:)]';
condn_data{6}=[D6(idx,:)]';
condn_data{7}=[D7(idx,:)]';
condn_data{8}=[D8(idx,:)]';
condn_data{9}=[D9(idx,:)]';
condn_data{10}=[D10(idx,:)]';


% 2norm
for i=1:length(condn_data)
    tmp = condn_data{i};
    for j=1:size(tmp,1)
        tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
    end
    condn_data{i}=tmp;
end



A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};
H = condn_data{8};
I = condn_data{9};
J = condn_data{10};


clear N
N = [A' B' C' D' E' F' G' H' I' J'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1);8*ones(size(H,1),1);...
    9*ones(size(I,1),1);10*ones(size(J,1),1)];

T = zeros(size(T1,1),10);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;
[aa bb]=find(T1==8);[aa(1) aa(end)]
T(aa(1):aa(end),8)=1;
[aa bb]=find(T1==9);[aa(1) aa(end)]
T(aa(1):aa(end),9)=1;
[aa bb]=find(T1==10);[aa(1) aa(end)]
T(aa(1):aa(end),10)=1;


% code to train a neural network
clear net_hand_mlp_0303
net_hand_mlp_0303 = patternnet([64 64 64]) ;
net_hand_mlp_0303.performParam.regularization=0.2;
net_hand_mlp_0303 = train(net_hand_mlp_0303,N,T','UseParallel','yes');
genFunction(net_hand_mlp_0303,'MLP_Hand_03032022')
save net_hand_mlp_0303 net_hand_mlp_0303

% using custom layers
layers = [ ...
    featureInputLayer(96)
    fullyConnectedLayer(96)
    batchNormalizationLayer
    leakyReluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(96)
    batchNormalizationLayer
    leakyReluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(96)
    batchNormalizationLayer
    leakyReluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(8)
    softmaxLayer
    classificationLayer
    ];



X = N;
Y=categorical(T1);
idx = randperm(length(Y),round(0.8*length(Y)));
Xtrain = X(:,idx);
Ytrain = Y(idx);
I = ones(length(Y),1);
I(idx)=0;
idx1 = find(I~=0);
Xtest = X(:,idx1);
Ytest = Y(idx1);



%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',50, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',256,...
    'ValidationFrequency',100,...
    'ValidationPatience',5,...
    'LearnRateSchedule','piecewise',...
    'ExecutionEnvironment','GPU',...
    'ValidationData',{Xtest',Ytest});

% build the classifier
net_mlp_hand = trainNetwork(Xtrain',Ytrain,layers,options);
net_mlp_hand_adam_64=net_mlp_hand;
save net_mlp_hand_adam_64 net_mlp_hand_adam_64



%% LDA on hand imagined data



clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20220302'};
cd(root_path)

imagined_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'HandOnline')
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
idx = randperm(length(imagined_files),round(0.8*length(imagined_files)));
train_files = imagined_files(idx);
I = ones(length(imagined_files),1);
I(idx)=0;
test_files = imagined_files(find(I==1));

for i=1:length(train_files)
    disp(i/length(train_files)*100)
    try
        load(train_files{i})
        file_loaded = true;
    catch
        file_loaded=false;
        disp(['Could not load ' files{j}]);
    end
    
    
    if file_loaded
        action = TrialData.TargetID;
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = find(kinax==3);
        temp = cell2mat(features(kinax));
        temp = temp(:,5:end);
        
        % get the smoothed and pooled data
        % get smoothed delta hg and beta features
        new_temp=[];
        [xx yy] = size(TrialData.Params.ChMap);
        for k=1:size(temp,2)
            tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
            tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
            tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
            pooled_data=[];
            for i=1:2:xx
                for j=1:2:yy
                    delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                    beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                    hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                    pooled_data = [pooled_data; delta; beta ;hg];
                end
            end
            new_temp= [new_temp pooled_data];
        end
        temp=new_temp;
        data_seg = temp(1:96,:); % only high gamma
        %data_seg = mean(data_seg,2);
        
        if action ==1
            D1i = cat(2,D1i,data_seg);
            %D1f = cat(2,D1f,feat_stats1);
        elseif action ==2
            D2i = cat(2,D2i,data_seg);
            %D2f = cat(2,D2f,feat_stats1);
        elseif action ==3
            D3i = cat(2,D3i,data_seg);
            %D3f = cat(2,D3f,feat_stats1);
        elseif action ==4
            D4i = cat(2,D4i,data_seg);
            %D4f = cat(2,D4f,feat_stats1);
        elseif action ==5
            D5i = cat(2,D5i,data_seg);
            %D5f = cat(2,D5f,feat_stats1);
        elseif action ==6
            D6i = cat(2,D6i,data_seg);
            %D6f = cat(2,D6f,feat_stats1);
        elseif action ==7
            D7i = cat(2,D7i,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        elseif action ==8
            D8i = cat(2,D8i,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        elseif action ==9
            D9i = cat(2,D9i,data_seg);
        elseif action ==10
            D10i = cat(2,D10i,data_seg);
        end
    end
end

data=[];
Y=[];
data=[data cell2mat(D1i)]; Y=[Y;0*ones(size(cell2mat(D1i),2),1)];
data=[data cell2mat(D2i)];  Y=[Y;1*ones(size(cell2mat(D2i),2),1)];
data=[data cell2mat(D3i)];  Y=[Y;2*ones(size(cell2mat(D3i),2),1)];
data=[data cell2mat(D4i)];  Y=[Y;3*ones(size(cell2mat(D4i),2),1)];
data=[data cell2mat(D5i)];  Y=[Y;4*ones(size(cell2mat(D5i),2),1)];
data=[data cell2mat(D6i)];  Y=[Y;5*ones(size(cell2mat(D6i),2),1)];
%data=[data cell2mat(D7i)];  Y=[Y;6*ones(size(cell2mat(D7i),2),1)];
%data=[data cell2mat(D8i)];  Y=[Y;7*ones(size(cell2mat(D8i),2),1)];
data=[data cell2mat(D9i)];  Y=[Y;6*ones(size(cell2mat(D9i),2),1)];
data=[data cell2mat(D10i)];  Y=[Y;7*ones(size(cell2mat(D10i),2),1)];
data=data';

% run LDA
W = LDA(data,Y);

% run it on the held out files and get classification accuracies
acc=zeros(size(W,1));
for i=1:length(test_files)
    disp(i/length(test_files)*100)
    try
        load(test_files{i})
        file_loaded = true;
    catch
        file_loaded=false;
        disp(['Could not load ' files{j}]);
    end
    
    
    if file_loaded
        action = TrialData.TargetID;
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = find(kinax==3);
        temp = cell2mat(features(kinax));
        temp = temp(:,5:end);
        
        % get the smoothed and pooled data
        % get smoothed delta hg and beta features
        new_temp=[];
        [xx yy] = size(TrialData.Params.ChMap);
        for k=1:size(temp,2)
            tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
            tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
            tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
            pooled_data=[];
            for i=1:2:xx
                for j=1:2:yy
                    delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                    beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                    hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                    pooled_data = [pooled_data; delta; beta ;hg];
                end
            end
            new_temp= [new_temp pooled_data];
        end
        temp=new_temp;
        data_seg = temp(1:96,:); % only high gamma
        %data_seg = mean(data_seg,2);
    end
    data_seg = data_seg';
    
    % run it thru the LDA
    L = [ones(size(data_seg,1),1) data_seg] * W';
    
    % get classification prob
    P = exp(L) ./ repmat(sum(exp(L),2),[1 size(L,2)]);
    
    %average prob
    decision = mean(P(1:end,:));
    %decision = P;
    [aa bb]=max(decision);
    
    % correction for online trials
    if TrialData.TargetID==9
        TrialData.TargetID = 7;
    elseif TrialData.TargetID==10
        TrialData.TargetID = 8;
    end
    
    
    % store results
    if TrialData.TargetID <=10
        acc(TrialData.TargetID,bb) = acc(TrialData.TargetID,bb)+1;
    end
end

for i=1:length(acc)
    acc(i,:)= acc(i,:)/sum(acc(i,:));
end
figure;imagesc(acc)
diag(acc)
mean(ans)




%% USING GRU ON THE NEURAL FEATURES ITSELF 







