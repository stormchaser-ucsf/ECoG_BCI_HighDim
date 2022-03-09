% getting the temporal data for a RNN/LSTM based decoder


clc;clear

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

% for only 6 DoF original:
%foldernames = {'20210526','20210528','20210602','20210609_pm','20210611'};

foldernames = {'20210613','20210616','20210623','20210625','20210630','20210702',...
    '20210707','20210716','20210728','20210804','20210806','20210813','20210818',...
    '20210825','20210827','20210901','20210903','20210910','20210917','20210924','20210929',...
    '20211001''20211006','20211008','20211013','20211015','20211022','20211027','20211029','20211103',...
    '20211105','20211117','20211119','20220126','20220128','20220202','20220204','20220209','20220211',...
    '20220218','20220223','20220225','20220302'};
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
    elseif i==25
        idx=[1:2 4:length(D)];
        D=D(idx);  
    elseif i==29
        idx= [1:2 8:9];
        D=D(idx);
    elseif i==33
        idx= [1:2 3:5 7:16];
        D=D(idx);        
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
            feat_stats.Mean = feat_stats.Mean(769:end);
            feat_stats.Var = feat_stats.Var(769:end);
            clear feat_stats1
            feat_stats1(1:length(data_seg)) = feat_stats;
            
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


cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save lstm_7DoF_imag_data D1i D2i D3i D4i D5i D6i D7i -v7.3

clearvars -except online_files foldernames

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
                    try                        
                        tmp = raw_data(bins(k)+[0:799],:);
                    catch
                        tmp=[];
                    end
                    data_seg = cat(2,data_seg,tmp);
                end
            end
            
            feat_stats = TrialData.FeatureStats;
            feat_stats.Mean = feat_stats.Mean(769:end);
            feat_stats.Var = feat_stats.Var(769:end);
            clear feat_stats1
            feat_stats1(1:length(data_seg)) = feat_stats;
            
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

 
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save lstm_7DoF_online_data D1 D2 D3 D4 D5 D6 D7 -v7.3


%% BUILDIN THE DECODER USIGN IMAGINED PLUS ONLINE DATA

clear;clc

Y=[];

condn_data_new=[];jj=1;

load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20211001\Robot3DArrow\103931\BCI_Fixed\Data0001.mat')
chmap = TrialData.Params.ChMap;

% filter and downsample the data, with spatial smoothing
% keep hG envelope as well as LFO activity

% hg filter
hgFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);
% low pass filter of raw
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
   'PassbandFrequency',40,'PassbandRipple',0.2, ...
   'SampleRate',1e3);
lpFilt1 = designfilt('lowpassiir','FilterOrder',4, ...
   'PassbandFrequency',10,'PassbandRipple',0.2, ...
   'SampleRate',1e3);
% lpFilt = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',1,'HalfPowerFrequency2',30, ...
%     'SampleRate',1e3);
% delta signal
deltaFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
    'SampleRate',1e3);
% beta signal
betaFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',13,'HalfPowerFrequency2',30, ...
    'SampleRate',1e3);
%lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
%    'PassbandFrequency',30,'PassbandRipple',0.2, ...
%    'SampleRate',1e3);
%fvtool(lpFilt)



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
load('lstm_data_with_imag_data','D1','D1i')
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
    
    
    
%     % delta through hilbert approach
%     i=9;
%     filtered_data = ((filter(...
%         Params.FilterBank(i).b, ...
%         Params.FilterBank(i).a, ...
%         tmp)));
%     tmp_delta=abs(hilbert(filtered_data));
%     tmp_delta = tmp_delta./norm(tmp_delta(:));
    
%     % beta through filter bank approach
%     filtered_data=zeros(size(tmp,1),size(tmp,2),2);
%     k=10:11;
%     for i=1:length(k)
%         filtered_data(:,:,i) =  ((filter(...
%             Params.FilterBank(k(i)).b, ...
%             Params.FilterBank(k(i)).a, ...
%             tmp)));
%     end
%     tmp_beta = squeeze(mean(filtered_data.^2,3));  
%     tmp_beta=filter(lpFilt,tmp_beta);
%     tmp_beta = tmp_beta./norm(tmp_beta(:));
    
    
    
    
%     
%     filtered_data(:,:,1) = abs(hilbert(filter(hg1,tmp)));
%     filtered_data(:,:,2) = abs(hilbert(filter(hg2,tmp)));
%     filtered_data(:,:,3) = abs(hilbert(filter(hg3,tmp)));
%     filtered_data(:,:,4) = abs(hilbert(filter(hg4,tmp)));
%     filtered_data(:,:,5) = abs(hilbert(filter(hg5,tmp)));
%     filtered_data(:,:,6) = abs(hilbert(filter(hg6,tmp)));
%     filtered_data(:,:,7) = abs(hilbert(filter(hg7,tmp)));
%     filtered_data(:,:,8) = abs(hilbert(filter(hg8,tmp)));
    %tmp_hg = squeeze(mean(filtered_data.^2,3));       
    %tmp_hg = filter(lpFilt,tmp_hg);
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % delta filtering
    
    
    % beta filtering
    
    
    
    
    
    
%     for j=1:size(tmp_hg,2)
%         tmp_hg(:,j) = smooth(tmp_hg(:,j),20);
%     end
%     
%     % low pass filtered signal
%     m=mean(tmp);
%     tmp_lp = filter(lpFilt,tmp)+m;
    
    % get beta
%     tmp_beta = abs(hilbert(filter(betaFilt,tmp)));
%     for j=1:size(tmp_beta,2)
%         tmp_beta(:,j) = smooth(tmp_beta(:,j),20);
%     end
%     
%     % get delta
%     m=mean(tmp);
%     tmp_delta = filter(deltaFilt,tmp)+m;
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    %tmp_beta = resample(tmp_beta,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    
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

load('lstm_data_with_imag_data','D2','D2i')
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
    
    
    
%     % delta through hilbert approach
%     i=9;
%     filtered_data = ((filter(...
%         Params.FilterBank(i).b, ...
%         Params.FilterBank(i).a, ...
%         tmp)));
%     tmp_delta=abs(hilbert(filtered_data));
%     tmp_delta = tmp_delta./norm(tmp_delta(:));
    
%     % beta through filter bank approach
%     filtered_data=zeros(size(tmp,1),size(tmp,2),2);
%     k=10:11;
%     for i=1:length(k)
%         filtered_data(:,:,i) =  ((filter(...
%             Params.FilterBank(k(i)).b, ...
%             Params.FilterBank(k(i)).a, ...
%             tmp)));
%     end
%     tmp_beta = squeeze(mean(filtered_data.^2,3));  
%     tmp_beta=filter(lpFilt,tmp_beta);
%     tmp_beta = tmp_beta./norm(tmp_beta(:));
    
    
    
    
%     
%     filtered_data(:,:,1) = abs(hilbert(filter(hg1,tmp)));
%     filtered_data(:,:,2) = abs(hilbert(filter(hg2,tmp)));
%     filtered_data(:,:,3) = abs(hilbert(filter(hg3,tmp)));
%     filtered_data(:,:,4) = abs(hilbert(filter(hg4,tmp)));
%     filtered_data(:,:,5) = abs(hilbert(filter(hg5,tmp)));
%     filtered_data(:,:,6) = abs(hilbert(filter(hg6,tmp)));
%     filtered_data(:,:,7) = abs(hilbert(filter(hg7,tmp)));
%     filtered_data(:,:,8) = abs(hilbert(filter(hg8,tmp)));
    %tmp_hg = squeeze(mean(filtered_data.^2,3));       
    %tmp_hg = filter(lpFilt,tmp_hg);
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % delta filtering
    
    
    % beta filtering
    
    
    
    
    
    
%     for j=1:size(tmp_hg,2)
%         tmp_hg(:,j) = smooth(tmp_hg(:,j),20);
%     end
%     
%     % low pass filtered signal
%     m=mean(tmp);
%     tmp_lp = filter(lpFilt,tmp)+m;
    
    % get beta
%     tmp_beta = abs(hilbert(filter(betaFilt,tmp)));
%     for j=1:size(tmp_beta,2)
%         tmp_beta(:,j) = smooth(tmp_beta(:,j),20);
%     end
%     
%     % get delta
%     m=mean(tmp);
%     tmp_delta = filter(deltaFilt,tmp)+m;
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    %tmp_beta = resample(tmp_beta,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    
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





load('lstm_data_with_imag_data','D3','D3i')
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
    
    
    
%     % delta through hilbert approach
%     i=9;
%     filtered_data = ((filter(...
%         Params.FilterBank(i).b, ...
%         Params.FilterBank(i).a, ...
%         tmp)));
%     tmp_delta=abs(hilbert(filtered_data));
%     tmp_delta = tmp_delta./norm(tmp_delta(:));
    
%     % beta through filter bank approach
%     filtered_data=zeros(size(tmp,1),size(tmp,2),2);
%     k=10:11;
%     for i=1:length(k)
%         filtered_data(:,:,i) =  ((filter(...
%             Params.FilterBank(k(i)).b, ...
%             Params.FilterBank(k(i)).a, ...
%             tmp)));
%     end
%     tmp_beta = squeeze(mean(filtered_data.^2,3));  
%     tmp_beta=filter(lpFilt,tmp_beta);
%     tmp_beta = tmp_beta./norm(tmp_beta(:));
    
    
    
    
%     
%     filtered_data(:,:,1) = abs(hilbert(filter(hg1,tmp)));
%     filtered_data(:,:,2) = abs(hilbert(filter(hg2,tmp)));
%     filtered_data(:,:,3) = abs(hilbert(filter(hg3,tmp)));
%     filtered_data(:,:,4) = abs(hilbert(filter(hg4,tmp)));
%     filtered_data(:,:,5) = abs(hilbert(filter(hg5,tmp)));
%     filtered_data(:,:,6) = abs(hilbert(filter(hg6,tmp)));
%     filtered_data(:,:,7) = abs(hilbert(filter(hg7,tmp)));
%     filtered_data(:,:,8) = abs(hilbert(filter(hg8,tmp)));
    %tmp_hg = squeeze(mean(filtered_data.^2,3));       
    %tmp_hg = filter(lpFilt,tmp_hg);
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % delta filtering
    
    
    % beta filtering
    
    
    
    
    
    
%     for j=1:size(tmp_hg,2)
%         tmp_hg(:,j) = smooth(tmp_hg(:,j),20);
%     end
%     
%     % low pass filtered signal
%     m=mean(tmp);
%     tmp_lp = filter(lpFilt,tmp)+m;
    
    % get beta
%     tmp_beta = abs(hilbert(filter(betaFilt,tmp)));
%     for j=1:size(tmp_beta,2)
%         tmp_beta(:,j) = smooth(tmp_beta(:,j),20);
%     end
%     
%     % get delta
%     m=mean(tmp);
%     tmp_delta = filter(deltaFilt,tmp)+m;
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    %tmp_beta = resample(tmp_beta,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    
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


load('lstm_data_with_imag_data','D4','D4i')
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
    
    
    
%     % delta through hilbert approach
%     i=9;
%     filtered_data = ((filter(...
%         Params.FilterBank(i).b, ...
%         Params.FilterBank(i).a, ...
%         tmp)));
%     tmp_delta=abs(hilbert(filtered_data));
%     tmp_delta = tmp_delta./norm(tmp_delta(:));
    
%     % beta through filter bank approach
%     filtered_data=zeros(size(tmp,1),size(tmp,2),2);
%     k=10:11;
%     for i=1:length(k)
%         filtered_data(:,:,i) =  ((filter(...
%             Params.FilterBank(k(i)).b, ...
%             Params.FilterBank(k(i)).a, ...
%             tmp)));
%     end
%     tmp_beta = squeeze(mean(filtered_data.^2,3));  
%     tmp_beta=filter(lpFilt,tmp_beta);
%     tmp_beta = tmp_beta./norm(tmp_beta(:));
    
    
    
    
%     
%     filtered_data(:,:,1) = abs(hilbert(filter(hg1,tmp)));
%     filtered_data(:,:,2) = abs(hilbert(filter(hg2,tmp)));
%     filtered_data(:,:,3) = abs(hilbert(filter(hg3,tmp)));
%     filtered_data(:,:,4) = abs(hilbert(filter(hg4,tmp)));
%     filtered_data(:,:,5) = abs(hilbert(filter(hg5,tmp)));
%     filtered_data(:,:,6) = abs(hilbert(filter(hg6,tmp)));
%     filtered_data(:,:,7) = abs(hilbert(filter(hg7,tmp)));
%     filtered_data(:,:,8) = abs(hilbert(filter(hg8,tmp)));
    %tmp_hg = squeeze(mean(filtered_data.^2,3));       
    %tmp_hg = filter(lpFilt,tmp_hg);
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % delta filtering
    
    
    % beta filtering
    
    
    
    
    
    
%     for j=1:size(tmp_hg,2)
%         tmp_hg(:,j) = smooth(tmp_hg(:,j),20);
%     end
%     
%     % low pass filtered signal
%     m=mean(tmp);
%     tmp_lp = filter(lpFilt,tmp)+m;
    
    % get beta
%     tmp_beta = abs(hilbert(filter(betaFilt,tmp)));
%     for j=1:size(tmp_beta,2)
%         tmp_beta(:,j) = smooth(tmp_beta(:,j),20);
%     end
%     
%     % get delta
%     m=mean(tmp);
%     tmp_delta = filter(deltaFilt,tmp)+m;
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    %tmp_beta = resample(tmp_beta,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    
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


load('lstm_data_with_imag_data','D5','D5i')
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
    
    
    
%     % delta through hilbert approach
%     i=9;
%     filtered_data = ((filter(...
%         Params.FilterBank(i).b, ...
%         Params.FilterBank(i).a, ...
%         tmp)));
%     tmp_delta=abs(hilbert(filtered_data));
%     tmp_delta = tmp_delta./norm(tmp_delta(:));
    
%     % beta through filter bank approach
%     filtered_data=zeros(size(tmp,1),size(tmp,2),2);
%     k=10:11;
%     for i=1:length(k)
%         filtered_data(:,:,i) =  ((filter(...
%             Params.FilterBank(k(i)).b, ...
%             Params.FilterBank(k(i)).a, ...
%             tmp)));
%     end
%     tmp_beta = squeeze(mean(filtered_data.^2,3));  
%     tmp_beta=filter(lpFilt,tmp_beta);
%     tmp_beta = tmp_beta./norm(tmp_beta(:));
    
    
    
    
%     
%     filtered_data(:,:,1) = abs(hilbert(filter(hg1,tmp)));
%     filtered_data(:,:,2) = abs(hilbert(filter(hg2,tmp)));
%     filtered_data(:,:,3) = abs(hilbert(filter(hg3,tmp)));
%     filtered_data(:,:,4) = abs(hilbert(filter(hg4,tmp)));
%     filtered_data(:,:,5) = abs(hilbert(filter(hg5,tmp)));
%     filtered_data(:,:,6) = abs(hilbert(filter(hg6,tmp)));
%     filtered_data(:,:,7) = abs(hilbert(filter(hg7,tmp)));
%     filtered_data(:,:,8) = abs(hilbert(filter(hg8,tmp)));
    %tmp_hg = squeeze(mean(filtered_data.^2,3));       
    %tmp_hg = filter(lpFilt,tmp_hg);
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % delta filtering
    
    
    % beta filtering
    
    
    
    
    
    
%     for j=1:size(tmp_hg,2)
%         tmp_hg(:,j) = smooth(tmp_hg(:,j),20);
%     end
%     
%     % low pass filtered signal
%     m=mean(tmp);
%     tmp_lp = filter(lpFilt,tmp)+m;
    
    % get beta
%     tmp_beta = abs(hilbert(filter(betaFilt,tmp)));
%     for j=1:size(tmp_beta,2)
%         tmp_beta(:,j) = smooth(tmp_beta(:,j),20);
%     end
%     
%     % get delta
%     m=mean(tmp);
%     tmp_delta = filter(deltaFilt,tmp)+m;
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    %tmp_beta = resample(tmp_beta,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    
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



load('lstm_data_with_imag_data','D6','D6i')
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
    
    
    
%     % delta through hilbert approach
%     i=9;
%     filtered_data = ((filter(...
%         Params.FilterBank(i).b, ...
%         Params.FilterBank(i).a, ...
%         tmp)));
%     tmp_delta=abs(hilbert(filtered_data));
%     tmp_delta = tmp_delta./norm(tmp_delta(:));
    
%     % beta through filter bank approach
%     filtered_data=zeros(size(tmp,1),size(tmp,2),2);
%     k=10:11;
%     for i=1:length(k)
%         filtered_data(:,:,i) =  ((filter(...
%             Params.FilterBank(k(i)).b, ...
%             Params.FilterBank(k(i)).a, ...
%             tmp)));
%     end
%     tmp_beta = squeeze(mean(filtered_data.^2,3));  
%     tmp_beta=filter(lpFilt,tmp_beta);
%     tmp_beta = tmp_beta./norm(tmp_beta(:));
    
    
    
    
%     
%     filtered_data(:,:,1) = abs(hilbert(filter(hg1,tmp)));
%     filtered_data(:,:,2) = abs(hilbert(filter(hg2,tmp)));
%     filtered_data(:,:,3) = abs(hilbert(filter(hg3,tmp)));
%     filtered_data(:,:,4) = abs(hilbert(filter(hg4,tmp)));
%     filtered_data(:,:,5) = abs(hilbert(filter(hg5,tmp)));
%     filtered_data(:,:,6) = abs(hilbert(filter(hg6,tmp)));
%     filtered_data(:,:,7) = abs(hilbert(filter(hg7,tmp)));
%     filtered_data(:,:,8) = abs(hilbert(filter(hg8,tmp)));
    %tmp_hg = squeeze(mean(filtered_data.^2,3));       
    %tmp_hg = filter(lpFilt,tmp_hg);
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % delta filtering
    
    
    % beta filtering
    
    
    
    
    
    
%     for j=1:size(tmp_hg,2)
%         tmp_hg(:,j) = smooth(tmp_hg(:,j),20);
%     end
%     
%     % low pass filtered signal
%     m=mean(tmp);
%     tmp_lp = filter(lpFilt,tmp)+m;
    
    % get beta
%     tmp_beta = abs(hilbert(filter(betaFilt,tmp)));
%     for j=1:size(tmp_beta,2)
%         tmp_beta(:,j) = smooth(tmp_beta(:,j),20);
%     end
%     
%     % get delta
%     m=mean(tmp);
%     tmp_delta = filter(deltaFilt,tmp)+m;
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    %tmp_beta = resample(tmp_beta,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    
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



load('lstm_data_with_imag_data','D7','D7i')
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
    
    
    
%     % delta through hilbert approach
%     i=9;
%     filtered_data = ((filter(...
%         Params.FilterBank(i).b, ...
%         Params.FilterBank(i).a, ...
%         tmp)));
%     tmp_delta=abs(hilbert(filtered_data));
%     tmp_delta = tmp_delta./norm(tmp_delta(:));
    
%     % beta through filter bank approach
%     filtered_data=zeros(size(tmp,1),size(tmp,2),2);
%     k=10:11;
%     for i=1:length(k)
%         filtered_data(:,:,i) =  ((filter(...
%             Params.FilterBank(k(i)).b, ...
%             Params.FilterBank(k(i)).a, ...
%             tmp)));
%     end
%     tmp_beta = squeeze(mean(filtered_data.^2,3));  
%     tmp_beta=filter(lpFilt,tmp_beta);
%     tmp_beta = tmp_beta./norm(tmp_beta(:));
    
    
    
    
%     
%     filtered_data(:,:,1) = abs(hilbert(filter(hg1,tmp)));
%     filtered_data(:,:,2) = abs(hilbert(filter(hg2,tmp)));
%     filtered_data(:,:,3) = abs(hilbert(filter(hg3,tmp)));
%     filtered_data(:,:,4) = abs(hilbert(filter(hg4,tmp)));
%     filtered_data(:,:,5) = abs(hilbert(filter(hg5,tmp)));
%     filtered_data(:,:,6) = abs(hilbert(filter(hg6,tmp)));
%     filtered_data(:,:,7) = abs(hilbert(filter(hg7,tmp)));
%     filtered_data(:,:,8) = abs(hilbert(filter(hg8,tmp)));
    %tmp_hg = squeeze(mean(filtered_data.^2,3));       
    %tmp_hg = filter(lpFilt,tmp_hg);
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
    % delta filtering
    
    
    % beta filtering
    
    
    
    
    
    
%     for j=1:size(tmp_hg,2)
%         tmp_hg(:,j) = smooth(tmp_hg(:,j),20);
%     end
%     
%     % low pass filtered signal
%     m=mean(tmp);
%     tmp_lp = filter(lpFilt,tmp)+m;
    
    % get beta
%     tmp_beta = abs(hilbert(filter(betaFilt,tmp)));
%     for j=1:size(tmp_beta,2)
%         tmp_beta(:,j) = smooth(tmp_beta(:,j),20);
%     end
%     
%     % get delta
%     m=mean(tmp);
%     tmp_delta = filter(deltaFilt,tmp)+m;
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    %tmp_beta = resample(tmp_beta,200,800);
    %tmp_delta = resample(tmp_delta,200,800);
    
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


%save downsampled_lstm_data_below30Hz_hg_imag_online condn_data_new Y -v7.3


% set aside training and testing data in a cell format
clear condn_data  condn_data7
idx = randperm(size(condn_data_new,3),round(0.9*size(condn_data_new,3)));
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
numHiddenUnits1 = [  90 120 150 200 325];
drop1 = [ 0.2 0.2 0.4  0.4 0.4];
numClasses = 7;
for i=4%1:length(drop1)
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
        'MaxEpochs',13, ...        
        'MiniBatchSize',256, ...
        'GradientThreshold',2, ...
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
% net_800 =net;
% save net_800 net_800

%net_bilstm = net;
%save net_bilstm net_bilstm


%% TESTING THE DATA ON ONLINE DATA
clc;clear
load net_bilstm
filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220304\RealRobotBatch';

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
   'PassbandFrequency',20,'PassbandRipple',0.2, ...
   'SampleRate',1e3);

% hg filter
hgFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);


% another low pass filter
lpFilt1 = designfilt('lowpassiir','FilterOrder',4, ...
   'PassbandFrequency',10,'PassbandRipple',0.2, ...
   'SampleRate',1e3);

% load the data, and run it through the classifier
decodes_overall=[];
data=[];
for i=1:length(files)
    disp(i)
    
    % load
    load(files{i})
    
    % create buffer
    data_buffer = randn(800,128)*0.25;
    
    %get data
    raw_data = TrialData.BroadbandData;
    raw_data1=cell2mat(raw_data');
    
    % state of trial
    state_idx = TrialData.TaskState;
    decodes=[];
    trial_data={};
    for j=1:length(raw_data)
        tmp = raw_data{j};
        s=size(tmp,1);
        if s<800
            data_buffer = circshift(data_buffer,-s);
            data_buffer(end-s+1:end,:)=tmp;
        else
            data_buffer(1:end,:)=tmp(s-800+1:end,:);
        end
        
        % storing the data
        trial_data{j} = data_buffer;
        
        
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
        neural_features = [tmp_hg tmp_lp tmp_lp];
        
        % classifier output
        out=predict(net_bilstm,neural_features');
        [aa bb]=max(out);
        class_predict = bb;
        
        % store results
        if state_idx(j)==3
            decodes=[decodes class_predict];
        end
    end
    data(i).task_state = TrialData.TaskState ;
    data(i).raw_data = trial_data;
    data(i).TargetID = TrialData.TargetID;
    data(i).Task = 'Robot_Online_Data';
    decodes_overall(i).decodes = decodes;
    decodes_overall(i).tid = TrialData.TargetID;
end
val_robot_data=data;
save val_robot_data val_robot_data -v7.3

% looking at the accuracy of the bilstm decoder overall
acc=zeros(7,7);
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

%% BUILDING THE DEOCDER usng only online data


% for now, reshape the data to be 8X16X600 and then use a 3D convolutional
% layer to perform spatiotemporal convolutions

clc;clear

% data shoudl be in format 600 X 1 X 128 X samples
condn_data=[];
Y=[];

load('lstm_data','D1')
condn_data1 = zeros(600,128,length(D1));
for i=1:length(D1)
    disp(i)
    tmp = D1{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,i) = tmp1;
    Y = [Y ;1];
end
clear D1

load('lstm_data','D2')
condn_data2 = zeros(600,128,length(D2));
for i=1:length(D2)
    disp(i)
    tmp = D2{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data2(:,:,i) = tmp1;
    Y = [Y ;2];
end
clear D2

load('lstm_data','D3')
condn_data3 = zeros(600,128,length(D3));
for i=1:length(D3)
    disp(i)
    tmp = D3{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data3(:,:,i) = tmp1;
    Y = [Y ;3];
end
clear D3


load('lstm_data','D4')
condn_data4 = zeros(600,128,length(D4));
for i=1:length(D4)
    disp(i)
    tmp = D4{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data4(:,:,i) = tmp1;
    Y = [Y ;4];
end
clear D4


load('lstm_data','D5')
condn_data5 = zeros(600,128,length(D5));
for i=1:length(D5)
    disp(i)
    tmp = D5{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data5(:,:,i) = tmp1;
    Y = [Y ;5];
end
clear D5


load('lstm_data','D6')
condn_data6 = zeros(600,128,length(D6));
for i=1:length(D6)
    disp(i)
    tmp = D6{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data6(:,:,i) = tmp1;
    Y = [Y ;6];
end
clear D6


load('lstm_data','D7')
condn_data7 = zeros(600,128,length(D7));
for i=1:length(D7)
    disp(i)
    tmp = D7{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data7(:,:,i) = tmp1;
    Y = [Y ;7];
end
clear D7

condn_data=cat(3,condn_data,condn_data1);clear condn_data1
condn_data=cat(3,condn_data,condn_data2);clear condn_data2
condn_data=cat(3,condn_data,condn_data3);clear condn_data3
condn_data=cat(3,condn_data,condn_data4);clear condn_data4
condn_data=cat(3,condn_data,condn_data5);clear condn_data5
condn_data=cat(3,condn_data,condn_data6);clear condn_data6
condn_data=cat(3,condn_data,condn_data7);clear condn_data7
condn_data_new(:,1,:,:) = condn_data;
condn_data = condn_data_new; clear condn_data_new

save temporal_condn_data_7DoF condn_data Y -v7.3

% randomizing 
idx = randperm(length(Y));
condn_data = condn_data(:,:,:,idx);
Y = Y(idx);

len =  round(0.9*length(Y));
XTrain = condn_data(:,:,:,1:len);
XTest = condn_data(:,:,:,len+1:end);
YTrain = categorical(Y(1:len));
YTest = categorical(Y(len+1:end));

clear condn_data

%%%%%% CNN construction %%%%%
layers = [
    imageInputLayer([600 1 128],'Normalization','none')
    
    convolution2dLayer([50 1],10,'Padding','same')
    batchNormalizationLayer
    %maxPooling2dLayer([3 1],'Stride',3)
    
    convolution2dLayer([10 1],10,'Padding','same')
    batchNormalizationLayer
    maxPooling2dLayer([3 1],'Stride',3)    
        
%     convolution2dLayer([10 1],10,'Padding','same')
%     batchNormalizationLayer
    %maxPooling2dLayer([3 1],'Stride',3)    
    
    fullyConnectedLayer(50)
    layerNormalizationLayer
    reluLayer
    dropoutLayer(.5)
    
    
    fullyConnectedLayer(7)    
    softmaxLayer
    classificationLayer];

%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',256,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ValidationData',{XTest,YTest},...
    'ExecutionEnvironment','auto');
    %'ValidationData',{XTest,YTest},...
 
%Y1=categorical(Y);    
net = trainNetwork(XTrain,YTrain,layers,options);    
    

%%%%%% CNN construction %%%%%


% build the classifier
%net = trainNetwork(XTrain,YTrain,layers,options);

%% build LSTM layers



clc;clear

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd(root_path)
load temporal_condn_data_7DoF

load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20211001\Robot3DArrow\103931\BCI_Fixed\Data0001.mat')
chmap = TrialData.Params.ChMap;

% filter and downsample the data, with spatial smoothing
% keep hG envelope as well as LFO activity

bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);
bpFilt2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',30, ...
    'SampleRate',1e3);
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',30,'PassbandRipple',0.2, ...
    'SampleRate',1e3);
fvtool(lpFilt)
    
condn_data =  squeeze(condn_data);
condn_data_new=[];
for ii=1:size(condn_data,3)
    disp(ii)
    
    tmp = squeeze(condn_data(:,:,ii));
    % filter the data
    tmp_hg = abs(hilbert(filter(bpFilt,tmp)));
    tmp_lp = filter(lpFilt,tmp);
    
    % get lfo of hg
    m=mean(tmp_hg);
    tmp_hg = filter(bpFilt2,tmp_hg)+m;
    
    % downsample the data
    tmp_lp = resample(tmp_lp,150,600);
    tmp_hg = resample(tmp_hg,150,600);
    
    % spatial pool
    tmp_lp = spatial_pool(tmp_lp,TrialData);
    tmp_hg = spatial_pool(tmp_hg,TrialData);
    
    % make new data structure 
    tmp = [tmp_hg tmp_lp]; 
   
    % store
    condn_data_new(:,:,ii) = tmp;
end

save downsampled_lstm_data_below30Hz_hg condn_data_new -v7.3


% set aside training and testing data in a cell format
clear condn_data
idx = randperm(size(condn_data_new,3),round(0.9*size(condn_data_new,3)));
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


% specify lstm structure
inputSize = 64;
numHiddenUnits = 50;
numClasses = 7;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits)       
    bilstmLayer(numHiddenUnits)        
    bilstmLayer(numHiddenUnits,'OutputMode','last')    
    batchNormalizationLayer
    fullyConnectedLayer(25)
    sigmoidLayer
    dropoutLayer(.5)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];



% options
options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize',64, ...
    'GradientThreshold',10, ...
    'Verbose',true, ...
    'ValidationFrequency',70,...
    'ValidationData',{XTest,YTest},...
    'Plots','training-progress');

% train the model
net = trainNetwork(XTrain,YTrain,layers,options);


% test the model performance on a held out day




%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',96,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ValidationData',{XTest,YTest},...
    'ExecutionEnvironment','auto');

%% BUILD A BILSTM MODEL USING ONLINE DATA I.E. NEURAL FEATURES GOING BACK 1S
% this will be a model with only 5 time-steps at best
% 5 time-steps with 2 time steps overlap 
% validate on held out day
% do it for smoothed data and non-smoothed data 














