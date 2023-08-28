% getting the temporal data for a RNN/LSTM based decoder


clc;clear

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath('C:\Users\nikic\Documents\MATLAB')

% for only 6 DoF original:
%foldernames = {'20210526','20210528','20210602','20210609_pm','20210611'};

% foldernames = {'20230301','20230302','20230308','20230309','20230315','20230316',...
%     '20230322','20230323','20230329','20230330','20230405','20230406','20230412',...
%     '20230419','20230420','20230426'};

foldernames = {'20220929','20220930'};
cd(root_path)

imag_files={};
online_files={};
k=1;jj=1;
for i=1:length(foldernames)
    disp([i/length(foldernames)]);
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
%     if i==19 % this is 20210917
%         idx = [1 2 5:8 9:10];
%         D = D(idx);
%     elseif i==25
%         idx=[1:2 4:length(D)];
%         D=D(idx);
%     elseif i==29
%         idx= [1:2 8:9];
%         D=D(idx);
%     elseif i==33
%         idx= [1:2 3:5 7:16];
%         D=D(idx);
%     elseif i==55
%         D=D([1 2 4:end]);
%     end
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
% get state 2 data, randomly selected between 300 and 400ms after target on
% for state 3 data, start daq randomly between onset and 200ms, with next
% bin randomly between 400 and 500ms.
D1i={};
D2i={};
D3i={};
D4i={};
D5i={};
D6i={};
D7i={};
files_not_loaded=[];
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
            files_not_loaded=[files_not_loaded;files(j)];
        end
        if file_loaded
            idx0 = find(TrialData.TaskState==2) ;
            idx = find(TrialData.TaskState==3) ;
            idx=[idx0 idx];
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
                %bins =1:400:s; % originally only for state 3
                bins = 250:500:s;
                jitter = round(100*rand(size(bins)));
                bins=bins+jitter;
                raw_data = [raw_data;raw_data4];
                for k=1:length(bins)-1
                    tmp = raw_data(bins(k)+[0:999],:);
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
save lstm_7DoF_imag_data_with_state2 D1i D2i D3i D4i D5i D6i D7i -v7.3

clearvars -except online_files foldernames files_not_loaded

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
            files_not_loaded=[files_not_loaded;files(j)];
        end
        if file_loaded
            idx0 = find(TrialData.TaskState==2) ;
            idx = find(TrialData.TaskState==3) ;
            idx=[idx0 idx];
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
                raw_data = raw_data(1:1000,:);
                data_seg = raw_data;
            elseif s>1000% for all other data length, have to parse the data in overlapping chuncks of 600ms, 50% overlap
                % old for state 3 alone
                %                 bins =1:400:s;
                %                 raw_data = [raw_data;raw_data4];
                %                 for k=1:length(bins)-1
                %                     try
                %                         tmp = raw_data(bins(k)+[0:799],:);
                %                     catch
                %                         tmp=[];
                %                     end
                %                     data_seg = cat(2,data_seg,tmp);
                %                 end

                % new for state 2 and 3
                bins =250:500:s;
                jitter = round(100*rand(size(bins)));
                bins=bins+jitter;
                raw_data = [raw_data;raw_data4];
                for k=1:length(bins)-1
                    try
                        tmp = raw_data(bins(k)+[0:999],:);
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
save lstm_7DoF_online_data_with_state2 D1 D2 D3 D4 D5 D6 D7 -v7.3


%% BUILDIN THE DECODER USIGN IMAGINED PLUS ONLINE DATA

clear;clc

Y=[];
addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers')
condn_data_new=[];jj=1;

load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20211001\Robot3DArrow\103931\BCI_Fixed\Data0001.mat')
chmap = TrialData.Params.ChMap;

% low pass filter of raw
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

% band pass filter of raw
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',1,'HalfPowerFrequency2',25, ...
    'SampleRate',1e3);
%lpFilt=bpFilt;

% loading chmap file
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210526\Robot3DArrow\112357\Imagined\Data0005.mat')
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
Params.FilterBank(end+1).fpass = [30,36]; % lg1
Params.FilterBank(end+1).fpass = [36,42]; % lg2
Params.FilterBank(end+1).fpass = [42,50]; % lg3

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

% preallocate
condn_data_new = zeros(100,256,5e4);

len=1000;
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load('lstm_7DoF_online_data_with_state2','D1');
load('lstm_7DoF_imag_data_with_state2','D1i');
condn_data1 = zeros(len,128,length(D1)+length(D1i));
k=1;
for i=1:length(D1)
    %disp(k)
    tmp = D1{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
for i=1:length(D1i)
    %disp(k)
    tmp = D1i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
clear D1 D1i

disp('Processing action 1')
for ii=1:size(condn_data1,3)
     %disp(ii)

    tmp = squeeze(condn_data1(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt,chmap);

    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

load('lstm_7DoF_online_data_with_state2','D2');
load('lstm_7DoF_imag_data_with_state2','D2i');
condn_data2 = zeros(len,128,length(D2)+length(D2i));
k=1;
for i=1:length(D2)
    %disp(k)
    tmp = D2{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
for i=1:length(D2i)
    %disp(k)
    tmp = D2i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
clear D2 D2i condn_data1

disp('Processing action 2')
for ii=1:size(condn_data2,3)
    %disp(ii)

    tmp = squeeze(condn_data2(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt,chmap);

    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

load('lstm_7DoF_online_data_with_state2','D3');
load('lstm_7DoF_imag_data_with_state2','D3i');
condn_data3 = zeros(len,128,length(D3)+length(D3i));
k=1;
for i=1:length(D3)
    %disp(k)
    tmp = D3{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
for i=1:length(D3i)
    %disp(k)
    tmp = D3i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
clear D3 D3i condn_data2

disp('Processing action 3')
for ii=1:size(condn_data3,3)
    %disp(ii)

    tmp = squeeze(condn_data3(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt,chmap);

    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('lstm_7DoF_online_data_with_state2','D4');
load('lstm_7DoF_imag_data_with_state2','D4i');
condn_data4 = zeros(len,128,length(D4)+length(D4i));
k=1;
for i=1:length(D4)
    %disp(k)
    tmp = D4{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
for i=1:length(D4i)
    %disp(k)
    tmp = D4i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
clear D4 D4i condn_data3

disp('Processing action 4')
for ii=1:size(condn_data4,3)
    %disp(ii)

    tmp = squeeze(condn_data4(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt,chmap);


    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end


load('lstm_7DoF_online_data_with_state2','D5');
load('lstm_7DoF_imag_data_with_state2','D5i');
condn_data5 = zeros(len,128,length(D5)+length(D5i));
k=1;
for i=1:length(D5)
    %disp(k)
    tmp = D5{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
for i=1:length(D5i)
    %disp(k)
    tmp = D5i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
clear D5 D5i condn_data4

disp('Processing action 5')
for ii=1:size(condn_data5,3)
    %disp(ii)

    tmp = squeeze(condn_data5(:,:,ii));


    tmp = extract_lstm_features(tmp,Params,lpFilt,chmap);

    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('lstm_7DoF_online_data_with_state2','D6');
load('lstm_7DoF_imag_data_with_state2','D6i');
condn_data6 = zeros(len,128,length(D6)+length(D6i));
k=1;
for i=1:length(D6)
    %disp(k)
    tmp = D6{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
for i=1:length(D6i)
    %disp(k)
    tmp = D6i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
clear D6 D6i condn_data5

disp('Processing action 6')
for ii=1:size(condn_data6,3)
    %disp(ii)

    tmp = squeeze(condn_data6(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt,chmap);

    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('lstm_7DoF_online_data_with_state2','D7');
load('lstm_7DoF_imag_data_with_state2','D7i');
condn_data7 = zeros(len,128,length(D7)+length(D7i));
k=1;
for i=1:length(D7)
    %disp(k)
    tmp = D7{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
for i=1:length(D7i)
    %disp(k)
    tmp = D7i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
clear D7 D7i condn_data6

disp('Processing action 7')
for ii=1:size(condn_data7,3)
    %disp(ii)

    tmp = squeeze(condn_data7(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt,chmap);


    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save decimated_lstm_data_below25Hz condn_data_new Y -v7.3
%save downsampled_lstm_data_below25Hz condn_data_new Y -v7.3
%save decimated_lstm_data_below25Hz_WithState2_with_lg condn_data_new Y -v7.3
%load decimated_lstm_data_below25Hz_WithState2

% set aside training and testing data in a cell format
%clear condn_data  condn_data7

% get rid of artifacts, any channel with activity >15SD, set it to near zero
for i=1:size(condn_data_new,3)
    xx=squeeze(condn_data_new(:,1:128,i));
    I = abs(xx)>15;
    I = sum(I);
    [aa bb]=find(I>0);
    xx(:,bb) = 1e-5*randn(size(xx(:,bb)));
    condn_data_new(:,1:128,i)=xx;

    xx=squeeze(condn_data_new(:,129:256,i));
    I = abs(xx)>15;
    I = sum(I);
    [aa bb]=find(I>0);
    xx(:,bb) = 1e-5*randn(size(xx(:,bb)));
    condn_data_new(:,129:256,i)=xx;

    %     xx=squeeze(condn_data_new(:,257:384,i));
    %     I = abs(xx)>15;
    %     I = sum(I);
    %     [aa bb]=find(I>0);
    %     xx(:,bb) = 1e-5*randn(size(xx(:,bb)));
    %     condn_data_new(:,257:384,i)=xx;
end

% normalize the data to be between 0 and 1
for i=1:size(condn_data_new,3)
    tmp=squeeze(condn_data_new(:,:,i));
    tmp1=tmp(:,1:128);
    tmp1 = (tmp1 - min(tmp1(:)))/(max(tmp1(:))-min(tmp1(:)));

    tmp2=tmp(:,129:256);
    tmp2 = (tmp2 - min(tmp2(:)))/(max(tmp2(:))-min(tmp2(:)));

    %     tmp3=tmp(:,257:384);
    %     tmp3 = (tmp3 - min(tmp3(:)))/(max(tmp3(:))-min(tmp3(:)));

    %tmp = [tmp1 tmp2 tmp3];
    tmp = [tmp1 tmp2 ];
    condn_data_new(:,:,i)=tmp;
end


% plot the mean of random samples in lfo range
mean_val=[];
for i = 1:size(condn_data_new,3)
    disp(i)
    tmp=squeeze(condn_data_new(:,:,i));
    tmp = mean(tmp(:,129:end));
    mean_val = [mean_val;mean(tmp(:))];
end
figure;hist(mean_val)

%
% % plotting for presentation
% tmp=squeeze(condn_data_new(:,:,1245));
% figure;
% imagesc(tmp(:,1:128)')
% caxis([0 .5])
% figure;%hg
% offset = 0:.1:127*.1;
% tmp1=tmp(:,1:128)+offset;
% tt=(1:80)*(1/100);
% plot(tt,tmp1(:,1:15),'k','LineWidth',1,'Color',[.2 .3 .9])
% axis tight
% set(gcf,'Color','w')
% set(gca,'FontSize',14)
% xlabel('Time in sec')
% ylabel('hG norm')
% box off
% yticks ''
%
% figure;%lmp
% offset = 0:.2:127*.2;
% tmp1=tmp(:,129:256)+offset;
% tt=(1:80)*(1/100);
% plot(tt,tmp1(:,1:15),'k','LineWidth',1,'Color',[.2 .3 .9])
% axis tight
% set(gcf,'Color','w')
% set(gca,'FontSize',14)
% xlabel('Time in sec')
% ylabel('LPF norm')
% box off
% yticks ''



% normalize the data to 2-norm
for i=1:size(condn_data_new,3)
    tmp=squeeze(condn_data_new(:,:,i));
    tmp1=tmp(:,1:128);
    tmp1=tmp1./norm(tmp1(:));
    tmp2=tmp(:,129:256);
    tmp2=tmp2./norm(tmp2(:));
    tmp = [tmp1 tmp2];    
    condn_data_new(:,:,i)=tmp;
end

% a random split into trainin and testing
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

% 
% % splitting training into balanced classes and throwing the rest into
% % testing
% indices={};
% for i=1:length(unique(Y))
%     idx = find(Y==i);
%     indices(i).idx=idx;
%     indices(i).len = length(idx);
% end
% %min_length = prop*min([indices.len]);
% min_length = 5600;
% idx_train=[];
% idx_test=[];
% for i=1:length(unique(Y))
%     idx = indices(i).idx;
%     a = randperm(length(idx),min_length);
%     b = ones(size(idx));
%     b(a)=0;
%     b=(find(b==1))';
% 
%     idx_train = [idx_train; idx((a))];
%     idx_test = [idx_test; idx((b))];
% end
% length(idx_train)/(length(idx_train)+length(idx_test))
% 
% XTrain={};
% YTrain=[];
% for i=1:length(idx_train)
%     tmp = squeeze(condn_data_new(:,:,idx_train(i)));
%     XTrain = cat(1,XTrain,tmp');
%     YTrain = [YTrain Y(idx_train(i))];
% end
% 
% XTest={};
% YTest=[];
% for i=1:length(idx_test)
%     tmp = squeeze(condn_data_new(:,:,idx_test(i)));
%     XTest = cat(1,XTest,tmp');
%     YTest = [YTest Y(idx_test(i))];
% end


% shuffle
idx  = randperm(length(YTrain));
XTrain = XTrain(idx);
YTrain = YTrain(idx);

YTrain = categorical(YTrain');
YTest = categorical(YTest');

% data augmentation: introduce random noise plus some mean shift to each
% channel for about 50k samples
aug_idx = randperm(length(XTrain));
for i=1:length(aug_idx)
    disp(i)
    tmp = XTrain{aug_idx(i)}';
    t_id=categorical(YTrain(aug_idx(i)));

    % hG
    tmp1 = tmp(:,1:128);
    % add variable noise
    %var_noise=randsample(400:1200,size(tmp1,2))/1e3;
    var_noise=0.7;
    add_noise=randn(size(tmp1)).*std(tmp1).*var_noise;
    tmp1n = tmp1 + add_noise;
    % add variable mean offset between 5 and 25%
    m=mean(tmp1);
    add_mean =  m*.25;
    %add_mean=randsample(0:500,size(tmp1,2))/1e3;
    flip_sign = rand(size(add_mean));
    flip_sign(flip_sign>0.5)=1;
    flip_sign(flip_sign<=0.5)=-1;
    add_mean=add_mean.*flip_sign+m;
    tmp1m = tmp1n + add_mean;
    %tmp1m = (tmp1m-min(tmp1m(:)))/(max(tmp1m(:))-min(tmp1m(:)));
    %  figure;plot(tmp1(:,3));hold on;plot(tmp1m(:,3))

    % lmp
    tmp2 = tmp(:,129:256);
    % add variable noise
    var_noise=0.7;
    %var_noise=randsample(400:1200,size(tmp2,2))/1e3;
    add_noise=randn(size(tmp2)).*std(tmp2).*var_noise;
    tmp2n = tmp2 + add_noise;
    % add variable mean offset between 5 and 25%
    m=mean(tmp2);
    add_mean =  m*.35;
    %add_mean=randsample(0:500,size(tmp2,2))/1e3;
    flip_sign = rand(size(add_mean));
    flip_sign(flip_sign>0.5)=1;
    flip_sign(flip_sign<=0.5)=-1;
    add_mean=add_mean.*flip_sign+m;
    tmp2m = tmp2n + add_mean;
    % tmp2m = (tmp2m-min(tmp2m(:)))/(max(tmp2m(:))-min(tmp2m(:)));

    %     %lg
    %     tmp3 = tmp(:,257:384);
    %     % add noise var
    %     add_noise=randn(size(tmp3)).*std(tmp3).*.795;
    %     tmp3n = tmp3 + add_noise;
    %     % add mean offset by 20%
    %     m=mean(tmp3);
    %     add_mean =  m*.2;
    %     flip_sign = rand(size(add_mean));
    %     flip_sign(flip_sign>0.5)=1;
    %     flip_sign(flip_sign<=0.5)=-1;
    %     add_mean=add_mean.*flip_sign+m;
    %     tmp3m = tmp3n + add_mean;
    %     tmp3m = (tmp3m-min(tmp3m(:)))/(max(tmp3m(:))-min(tmp3m(:)));

    %tmp=[tmp1m tmp2m tmp3m]';
    tmp=[tmp1m tmp2m]';

    XTrain=cat(1,XTrain,tmp);
    YTrain = cat(1,YTrain,t_id);
end

%
% % plotting examples
% tmp=tmp1;
% figure;%hg
% offset = 0:.2:127*.2;
% tmp11=tmp(:,1:128)+offset;
% tt=(1:80)*(1/100);
% plot(tt,tmp11(:,15:17),'k','LineWidth',1,'Color',[.2 .3 .9])
% axis tight
% axis off
% set(gcf,'Color','w')
% tmp=add_noise;
% figure;%hg
% offset = 0:.2:127*.2;
% tmp11=tmp(:,1:128)+offset;
% tt=(1:80)*(1/100);
% plot(tt,tmp11(:,15:17),'k','LineWidth',1,'Color',[.2 .3 .9])
% axis tight
% axis off
% set(gcf,'Color','w')
% tmp=repmat(add_mean,80,1);
% figure;%hg
% offset = 0:.2:127*.2;
% tmp11=tmp(:,1:128)+offset;
% tt=(1:80)*(1/100);
% plot(tt,tmp11(:,15:17),'k','LineWidth',1,'Color',[.2 .3 .9])
% axis tight
% axis off
% set(gcf,'Color','w')
% tmp=tmp1m;
% figure;%hg
% offset = 0:.2:127*.2;
% tmp11=tmp(:,1:128)+offset;
% tt=(1:80)*(1/100);
% plot(tt,tmp11(:,15:17),'k','LineWidth',1,'Color',[.2 .3 .9])
% axis tight
% axis off
% set(gcf,'Color','w')

% implement label smoothing to see how that does
%save training_data_bilstm_pooled3Feat condn_data_new Y -v7.3

%
% tmp=str2num(cell2mat(tmp));
% a=0.01;
% tmp1 = (1-a).*tmp + (a)*(1/7);
% clear YTrain
% YTrain = tmp1;
% YTrain =categorical(YTrain);
%clear condn_data_new

% specify lstm structure
inputSize = 256;
numHiddenUnits1 = [  90 120 250 128 325];
drop1 = [ 0.2 0.2 0.3  0.3 0.4];
numClasses = 7;
for i=3%1:length(drop1)
    numHiddenUnits=numHiddenUnits1(i);
    drop=drop1(i);
    layers = [ ...
        sequenceInputLayer(inputSize)
        bilstmLayer(numHiddenUnits,'OutputMode','sequence','Name','lstm_1')
        dropoutLayer(drop)
        layerNormalizationLayer
        gruLayer(numHiddenUnits/2,'OutputMode','last','Name','lstm_2')
        dropoutLayer(drop)
        %layerNormalizationLayer
        fullyConnectedLayer(25)
        leakyReluLayer
        batchNormalizationLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];



    % options
    batch_size=64;
    val_freq = floor(length(XTrain)/batch_size);
    options = trainingOptions('adam', ...
        'MaxEpochs',150, ...
        'MiniBatchSize',batch_size, ...
        'GradientThreshold',10, ...
        'Verbose',true, ...
        'ValidationFrequency',val_freq,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XTest,YTest},...
        'ValidationPatience',6,...
        'Plots','training-progress',...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',0.1,...
        'OutputNetwork','best-validation-loss',...
        'LearnRateDropPeriod',75,...
        'InitialLearnRate',0.001);

    % train the model
    net = trainNetwork(XTrain,YTrain,layers,options);
end
%
% net_800 =net;
% save net_800 net_800

%net_bilstm_lg = net;
%save net_bilstm_lg net_bilstm_lg

%net_bilstmhg=net; % this has more noise variance in the data augmentation
%save net_bilstmhg net_bilstmhg

net_bilstm_20220929 = net;
save net_bilstm_20220929 net_bilstm_20220929

net1=net;

% looking at accuracy against the validation data
ytest=[];
for i=1:length(YTest)
    ytest(i)  = str2num(string(YTest(i)));
end

num_elem=[];
for i=1:length(unique(ytest))
    num_elem(i) = sum(ytest==i);
end
figure;bar(num_elem)
min_elem = min(num_elem);

acc=zeros(7);
for iter = 1:10
    for i=1:7
        idx = find(ytest==i);
        I=randperm(length(idx),min_elem);
        idx = idx(I);
        ytest1=ytest(idx);
        xtest1=XTest(idx);
        out = predict(net,xtest1);
        for j=1:size(out,1)
            [aa bb] = (max(out(j,:)));
            acc(ytest1(j),bb)=acc(ytest1(j),bb)+1;
        end
    end
end

for i=1:length(acc)
    acc(i,:)=acc(i,:)./sum(acc(i,:));
end
mean(diag(acc))
figure;imagesc(acc)
figure;stem(diag(acc))


%% TESTING THE DATA ON ONLINE DATA

clear;clc

%filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220304\RealRobotBatch';
acc_mlp_days=[];
acc_days=[];
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd(root_path)
%foldernames = {'20220803','20220810','20220812'};
foldernames = {'20230419'};


% load the lstm 
load net_bilstm_20220824
net_bilstm = net_bilstm_20220824;



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
Params.FilterBank(end+1).fpass = [30,36]; % lg1
Params.FilterBank(end+1).fpass = [36,42]; % lg2
Params.FilterBank(end+1).fpass = [42,50]; % lg3
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

for i=1:length(foldernames)
    disp(i)
    filepath = fullfile(root_path,foldernames{i},'Robot3DArrow');
    [acc_lstm_sample,acc_mlp_sample,acc_lstm_trial,acc_mlp_trial]...
        =get_lstm_performance(filepath,net_bilstm,Params,lpFilt);

    acc_days = [acc_days diag(acc_lstm_sample)];
    acc_mlp_days = [acc_mlp_days diag(acc_mlp_sample)];
end


%
figure;
tmp = [acc_days(:) acc_mlp_days(:)];
boxplot(tmp)
hold on
cmap=turbo(size(acc_days,2));
for i=1:size(acc_days,2)
    s=scatter(ones(1,1)+0.05*randn(1,1) , mean(acc_days(:,i))','MarkerEdgeColor',cmap(i,:),...
        'LineWidth',2);
    s.SizeData=100;
    s=scatter(2*ones(1,1)+0.05*randn(1,1) , mean(acc_mlp_days(:,i))',...
        'MarkerEdgeColor',cmap(i,:),'LineWidth',2);
    s.SizeData=100;
end
ylim([0.3 1])
xticks(1:2)
xticklabels({'stack biLSTM','MLP'})
legend({'0803','','0810','','0812','0817'})
title('Arrow Task')
ylabel('Accuracy of inidiv. samples at 5Hz')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
box off

% plotting the success of individual actions
figure;
hold on
for i=1:7
    idx = i:7:size(tmp,1);
    decodes = tmp(idx,:);
    disp(decodes)
    h=bar(2*i-0.25,mean(decodes(:,1)));
    h1=bar(2*i+0.25,mean(decodes(:,2)));
    h.BarWidth=0.4;
    h.FaceColor=[0.2 0.2 0.7];
    h1.BarWidth=0.4;
    h1.FaceColor=[0.7 0.2 0.2];
    h.FaceAlpha=0.85;
    h1.FaceAlpha=0.85;

    %     s=scatter(ones(3,1)*2*i-0.25+0.05*randn(3,1),decodes(:,1),'LineWidth',2);
    %     s.CData = [0.2 0.2 0.7];
    %     s.SizeData=50;
    %
    %     s=scatter(ones(3,1)*2*i+0.25+0.05*randn(3,1),decodes(:,2),'LineWidth',2);
    %     s.CData = [0.7 0.2 0.2];
    %     s.SizeData=50;
end
xticks([2:2:14])
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
ylabel('Decoding Accuracy')
legend('LSTM','MLP')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)


%% FINE TUNING LSTM MODEL FOR A BATCH UPDATE ON ARROW DATA

clc;clear
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd(root_path)
%foldernames = {'20220803','20220810','20220812'};
foldernames = {'20221006'};

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
Params.FilterBank(end+1).fpass = [30,36]; % lg1
Params.FilterBank(end+1).fpass = [36,42]; % lg2
Params.FilterBank(end+1).fpass = [42,50]; % lg3
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

% get all the folders
filepath = fullfile(root_path,foldernames{1},'Robot3DArrow');
folders = dir(filepath);
folders=folders(3:end);
%folders=folders(3:8);

% load the decoder
load net_bilstm_20220929
net_bilstm = net_bilstm_20220929;


% within a loop, leave out one folder and then update lstm on all other
% folders
acc_batch_samples_overall = [];
acc_orig_samples_overall = [];
acc_batch_trial_overall = [];
acc_orig_trial_overall = [];
for i=1:length(folders)
    % i is is the testing folder everything else is the training folder
    idx = ones(length(folders),1);
    idx(i)=0;   
    idx_train = find(idx==1);
    idx_test = find(idx==0);
    folders_train = folders(idx_train);
    folders_test = folders(idx_test);

    % get the files
    files_train=[];
    for j=1:length(folders_train)
        subfolder = fullfile(folders_train(j).folder,folders_train(j).name,'BCI_Fixed');
        tmp = findfiles('mat',subfolder,1)';
        files_train =[files_train;tmp];
    end
    files_test=[];
    for j=1:length(folders_test)
        subfolder = fullfile(folders_test(j).folder,folders_test(j).name,'BCI_Fixed');
        tmp = findfiles('mat',subfolder,1)';
        files_test =[files_test;tmp];
    end

    % get the neural features from the files    
    [XTrain,XTest,YTrain,YTest] = get_lstm_features(files_train,Params,lpFilt);
    %[XTrain,XTest,YTrain,YTest] = get_lstm_features_robotBatch(files_train,Params,lpFilt);

    % update the decoder
    batch_size=128;
    val_freq = floor(length(XTrain)/batch_size);
    options = trainingOptions('adam', ...
        'MaxEpochs',50, ...
        'MiniBatchSize',batch_size, ...
        'GradientThreshold',10, ...
        'Verbose',true, ...
        'ValidationFrequency',val_freq,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XTest,YTest},...
        'ValidationPatience',6,...
        'Plots','training-progress',...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',0.1,...
        'OutputNetwork','best-validation-loss',...
        'LearnRateDropPeriod',30,...
        'InitialLearnRate',2e-4);

    % train the model
    clear net
    layers = net_bilstm.Layers;
    net = trainNetwork(XTrain,YTrain,layers,options);

    % test it out on the held out folder and store accuracy, compare it to
    % the original accuracy     
    [acc_batchUdpate_sample,acc_orig_sample,acc_batchUpdate_trial,acc_orig_trial]...
        = get_lstm_performance_afterBatch(files_test,net,Params,lpFilt);

    acc_batch_samples_overall = [acc_batch_samples_overall diag(acc_batchUdpate_sample)];
    acc_orig_samples_overall = [acc_orig_samples_overall diag(acc_orig_sample)];
    acc_batch_trial_overall = [acc_batch_trial_overall diag(acc_batchUpdate_trial)];
    acc_orig_trial_overall = [acc_orig_trial_overall diag(acc_orig_trial)];

end

%
figure;
tmp = [acc_batch_samples_overall(:) acc_orig_samples_overall(:)];
boxplot(tmp)
hold on
%cmap=turbo(size(acc_batch_samples_overall,2));
cmap=[.2 .2 .2];
cmap=repmat(cmap,size(acc_batch_samples_overall,2),1);
for i=1:size(acc_batch_samples_overall,2)
    s=scatter(ones(1,1)+0.05*randn(1,1) , mean(acc_batch_samples_overall(:,i))','MarkerEdgeColor',cmap(i,:),...
        'LineWidth',1);
    s.SizeData=100;
    s=scatter(2*ones(1,1)+0.05*randn(1,1) , mean(acc_orig_samples_overall(:,i))',...
        'MarkerEdgeColor',cmap(i,:),'LineWidth',1);
    s.SizeData=100;
end
ylim([0.0 1])
xticks(1:2)
xticklabels({'Batch Update','Original'})
title('Arrow Task')
ylabel('Accuracy of inidiv. samples at 5Hz')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
box off

% plotting the success of individual actions
figure;
hold on
for i=1:7
    idx = i:7:size(tmp,1);
    decodes = tmp(idx,:);
    %disp(decodes);
    m = mean(decodes);
    s = std(decodes)/sqrt(length(decodes));
    h=bar(2*i-0.25,mean(decodes(:,1)));
    er = errorbar(2*i-0.25,m(1),s(1),s(1));
    er.Color = [0 0 0];
    er.LineStyle = 'none';
    er.LineWidth=1;
    h1=bar(2*i+0.25,mean(decodes(:,2)));
    er1 = errorbar(2*i+0.25,m(2),s(2),s(2));
    er1.Color = [0 0 0];
    er1.LineStyle = 'none';
    er1.LineWidth=1;
    h.BarWidth=0.4;
    h.FaceColor=[0.2 0.2 0.7];
    h1.BarWidth=0.4;
    h1.FaceColor=[0.7 0.2 0.2];
    h.FaceAlpha=0.85;
    h1.FaceAlpha=0.85;

    %     s=scatter(ones(3,1)*2*i-0.25+0.05*randn(3,1),decodes(:,1),'LineWidth',2);
    %     s.CData = [0.2 0.2 0.7];
    %     s.SizeData=50;
    %
    %     s=scatter(ones(3,1)*2*i+0.25+0.05*randn(3,1),decodes(:,2),'LineWidth',2);
    %     s.CData = [0.7 0.2 0.2];
    %     s.SizeData=50;
end
xticks([2:2:14])
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
ylabel('Decoding Accuracy')
legend('Batch Update','','Orig')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)





%% GET REALROBOT BATCH DATA ON A TRIAL BY TRIAL LEVEL

clc;clear
close all
addpath('C:\Users\nikic\Documents\MATLAB')

root_folder='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
folder_days = {'20210716','20210728','20210804','20210806', '20220202','20220211',...
    '20220225','20220304','20220309','20220311','20220316','20220323','20220325',...
    '20220330','20220420','20220422','20220429','20220504','20220506','20220513',...
    '20220518','20220520','20220715','20220722','20220727','20220729',...
    '20220216','20220803','20220819','20220831','20220902',...
    '20220907','20220909','20220916'};

% 20220216 has 9D robotbatch seems important

files=[];
python_files=[];
for i=1:length(folder_days)
    folder_path = fullfile(root_folder,folder_days{i},'RealRobotBatch');
    files = [files;findfiles('',folder_path,1)'];

    %     python_folder_path = dir(folder_path);
    %     python_folders={};
    %     for j=3:length(python_folder_path)
    %         python_folders=cat(2,python_folders,python_folder_path(j).name);
    %     end
    %
    %     for j=1:length(python_folders)
    %         folder_path = fullfile(root_folder,folder_days{i},'Python',folder_days{i});
    %     end
end

files1=[];
for i=1:length(files)
    if length(regexp(files{i},'Data'))>0
        files1=[files1;files(i)];
    end
end
files=files1;


% load the robot data for the LSTM format
D1={};
D2={};
D3={};
D4={};
D5={};
D6={};
D7={};
robot_batch_trials_lstm=[];
files_not_loaded=[];
for i=1:length(files)
    disp(i/length(files)*100)

    try
        load(files{i})
        file_loaded = true;
        warning('off')
    catch
        file_loaded=false;
        disp(['Could not load ' files{i}]);
        files_not_loaded=[files_not_loaded;files(i)];
    end
    if file_loaded

        idx00 = find(TrialData.TaskState==1) ;
        idx0 = find(TrialData.TaskState==2) ;
        idx = find(TrialData.TaskState==3) ;
        idx=[idx00 idx0 idx];
        raw_data = cell2mat(TrialData.BroadbandData(idx)');
        idx1 = find(TrialData.TaskState==4) ;
        raw_data4 = cell2mat(TrialData.BroadbandData(idx1)');
        id = TrialData.TargetID;
        s = size(raw_data,1);
        %         if s>7800
        %             s=7800;
        %         end
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
            %bins =1:400:s; % originally only for state 3
            bins = 1:500:s;
            jitter = round(100*rand(size(bins)));
            bins=bins+jitter;
            raw_data = [raw_data;raw_data4];
            for k=1:length(bins)-1
                tmp = raw_data(bins(k)+[0:999],:);
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
        robot_batch_trials_lstm(i).TargetID = id;
        robot_batch_trials_lstm(i).RawData = data_seg;

        folder_name=files{i}(61:68);
        for j=1:length(folder_days)
            if strcmp(folder_days{j},folder_name)
                robot_batch_trials_lstm(i).Day = j;
                break
            end
        end
    end
end

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save robot_batch_trials_lstm D1 D2 D3 D4 D5 D6 D7 -v7.3


%%%%% get the data for LSTMs on a trial by trial level

robot_batch_trials_lstm_features=[];

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

% low pass filter
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

% get the processed samples
for ii=1:length(robot_batch_trials_lstm)
    disp(ii/length(robot_batch_trials_lstm)*100)

    tmp_data = robot_batch_trials_lstm(ii).RawData;
    temp_lstm=[];

    for j=1:length(tmp_data)

        tmp=tmp_data{j};

        % extract features
        tmp = extract_lstm_features(tmp,Params,lpFilt);

        % store it
        temp_lstm{j}=tmp;
    end

    robot_batch_trials_lstm_features(ii).TargetID = ...
        robot_batch_trials_lstm(ii).TargetID;
    robot_batch_trials_lstm_features(ii).NeuralFeatures = ...
        temp_lstm;
    robot_batch_trials_lstm_features(ii).Day = ...
        robot_batch_trials_lstm(ii).Day;
end
save robot_batch_trials_lstm_features robot_batch_trials_lstm_features -v7.3


%% FINE TUNE LSTM ON THE ROBOT BATCH DATA

clear;clc

Y=[];
addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers')
condn_data_new=[];jj=1;

load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20211001\Robot3DArrow\103931\BCI_Fixed\Data0001.mat')
chmap = TrialData.Params.ChMap;

% low pass filter of raw
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

% band pass filter of raw
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',1,'HalfPowerFrequency2',25, ...
    'SampleRate',1e3);
%lpFilt=bpFilt;

% loading chmap file
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210526\Robot3DArrow\112357\Imagined\Data0005.mat')
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
Params.FilterBank(end+1).fpass = [30,36]; % lg1
Params.FilterBank(end+1).fpass = [36,42]; % lg2
Params.FilterBank(end+1).fpass = [42,50]; % lg3

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

len=1000;
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load('robot_batch_trials_lstm','D1');
condn_data1 = zeros(len,128,length(D1));
k=1;
for i=1:length(D1)
    %disp(k)
    tmp = D1{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
clear D1

disp('Processing action 1')
for ii=1:size(condn_data1,3)
    % disp(ii)

    tmp = squeeze(condn_data1(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt);

    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

load('robot_batch_trials_lstm','D2');
condn_data2 = zeros(len,128,length(D2));
k=1;
for i=1:length(D2)
    %disp(k)
    tmp = D2{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
clear D2 D2i condn_data1

disp('Processing action 2')
for ii=1:size(condn_data2,3)
    %disp(ii)

    tmp = squeeze(condn_data2(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt);

    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

load('robot_batch_trials_lstm','D3');
condn_data3 = zeros(len,128,length(D3));
k=1;
for i=1:length(D3)
    %disp(k)
    tmp = D3{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
clear D3 D3i condn_data2

disp('Processing action 3')
for ii=1:size(condn_data3,3)
    %disp(ii)

    tmp = squeeze(condn_data3(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt);

    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

load('robot_batch_trials_lstm','D4');
condn_data4 = zeros(len,128,length(D4));
k=1;
for i=1:length(D4)
    %disp(k)
    tmp = D4{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
clear D4 D4i condn_data3

disp('Processing action 4')
for ii=1:size(condn_data4,3)
    %disp(ii)

    tmp = squeeze(condn_data4(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt);


    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

load('robot_batch_trials_lstm','D5');
condn_data5 = zeros(len,128,length(D5));
k=1;
for i=1:length(D5)
    %disp(k)
    tmp = D5{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
clear D5 D5i condn_data4

disp('Processing action 5')
for ii=1:size(condn_data5,3)
    %disp(ii)

    tmp = squeeze(condn_data5(:,:,ii));


    tmp = extract_lstm_features(tmp,Params,lpFilt);

    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('robot_batch_trials_lstm','D6');
condn_data6 = zeros(len,128,length(D6));
k=1;
for i=1:length(D6)
    %disp(k)
    tmp = D6{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
clear D6 D6i condn_data5

disp('Processing action 6')
for ii=1:size(condn_data6,3)
    %disp(ii)

    tmp = squeeze(condn_data6(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt);

    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

load('robot_batch_trials_lstm','D7');
condn_data7 = zeros(len,128,length(D7));
k=1;
for i=1:length(D7)
    %disp(k)
    tmp = D7{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
clear D7 D7i condn_data6

disp('Processing action 7')
for ii=1:size(condn_data7,3)
    %disp(ii)

    tmp = squeeze(condn_data7(:,:,ii));

    tmp = extract_lstm_features(tmp,Params,lpFilt);


    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save decimated_lstm_robot_batchData condn_data_new Y -v7.3

% get rid of artifacts, any channel with activity >15SD, set it to near zero
for i=1:size(condn_data_new,3)
    xx=squeeze(condn_data_new(:,1:128,i));
    I = abs(xx)>15;
    I = sum(I);
    [aa bb]=find(I>0);
    xx(:,bb) = 1e-5*randn(size(xx(:,bb)));
    condn_data_new(:,1:128,i)=xx;

    xx=squeeze(condn_data_new(:,129:256,i));
    I = abs(xx)>15;
    I = sum(I);
    [aa bb]=find(I>0);
    xx(:,bb) = 1e-5*randn(size(xx(:,bb)));
    condn_data_new(:,129:256,i)=xx;

    %     xx=squeeze(condn_data_new(:,257:384,i));
    %     I = abs(xx)>15;
    %     I = sum(I);
    %     [aa bb]=find(I>0);
    %     xx(:,bb) = 1e-5*randn(size(xx(:,bb)));
    %     condn_data_new(:,257:384,i)=xx;
end

% normalize the data to be between 0 and 1
for i=1:size(condn_data_new,3)
    tmp=squeeze(condn_data_new(:,:,i));
    tmp1=tmp(:,1:128);
    tmp1 = (tmp1 - min(tmp1(:)))/(max(tmp1(:))-min(tmp1(:)));

    tmp2=tmp(:,129:256);
    tmp2 = (tmp2 - min(tmp2(:)))/(max(tmp2(:))-min(tmp2(:)));

    %     tmp3=tmp(:,257:384);
    %     tmp3 = (tmp3 - min(tmp3(:)))/(max(tmp3(:))-min(tmp3(:)));

    %tmp = [tmp1 tmp2 tmp3];
    tmp = [tmp1 tmp2 ];
    condn_data_new(:,:,i)=tmp;
end



% % normalize the data to 2-norm
% for i=1:size(condn_data_new,3)
%     tmp=squeeze(condn_data_new(:,:,i));
%     tmp1=tmp(:,1:256);
%     tmp1=tmp1./norm(tmp1(:));
%     %tmp2=tmp(:,129:256);
%     %tmp2=tmp2./norm(tmp2(:));
%     %tmp = [tmp1 tmp2];
%     tmp = [tmp1 ];
%     condn_data_new(:,:,i)=tmp;
% end

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

%%%% optional PCA step to see if dim red helps
% on training data 
tmp=cell2mat(XTrain');
tmp_hg = tmp(1:128,:);
[c,s,l]=pca(tmp_hg','centered','on');
ll=cumsum(l)./sum(l);
figure;stem(ll)
tmp_hg = s(:,1:40);

tmp_lfo = tmp(129:end,:);
[c1,s1,l1]=pca(tmp_lfo','centered','on');
l11=cumsum(l1)./sum(l1);
figure;stem(l11)
tmp_lfo = s1(:,1:20);

tmp_data={};k=1;
for i=1:100:size(tmp_lfo,1)
    tmp_data{k} = [tmp_hg(i:i+99,:) tmp_lfo(i:i+99,:)]';
    k=k+1;
end
tmp_data=tmp_data';
XTrain = tmp_data;

% normalize each sample to be between 0 and 1
for i=1:length(XTrain)
    tmp1=XTrain{i};
    tmp1 = (tmp1 - min(tmp1(:)))/(max(tmp1(:))-min(tmp1(:)));
    XTrain{i}=tmp1;
end

% on testing data 
tmp=cell2mat(XTest');
tmp_hg = tmp(1:128,:);
tmp_hg = tmp_hg-mean(tmp_hg);
tmp_hg = tmp_hg'*c(:,1:40);

tmp_lfo = tmp(129:end,:);
tmp_lfo = tmp_lfo-mean(tmp_lfo);
tmp_lfo = tmp_lfo'*c1(:,1:20);

tmp_data={};k=1;
for i=1:100:size(tmp_lfo,1)
    tmp_data{k} = [tmp_hg(i:i+99,:) tmp_lfo(i:i+99,:)]';
    k=k+1;
end
tmp_data=tmp_data';
XTest = tmp_data;

% normalize each sample to be between 0 and 1
for i=1:length(XTest)
    tmp1=XTest{i};
    tmp1 = (tmp1 - min(tmp1(:)))/(max(tmp1(:))-min(tmp1(:)));
    XTest{i}=tmp1;
end
%%% end PCA step

% shuffle
idx  = randperm(length(YTrain));
XTrain = XTrain(idx);
YTrain = YTrain(idx);

YTrain = categorical(YTrain');
YTest = categorical(YTest');

% data augmentation: introduce random noise plus some mean shift to each
% channel for about 50k samples
aug_idx = randperm(length(XTrain));
for i=1:length(aug_idx)
    tmp = XTrain{aug_idx(i)}';
    t_id=categorical(YTrain(aug_idx(i)));

    % hG
    tmp1 = tmp(:,1:128);
    % add variable noise
    %var_noise=randsample(400:1200,size(tmp1,2))/1e3;
    var_noise=0.7;
    add_noise=randn(size(tmp1)).*std(tmp1).*var_noise;
    tmp1n = tmp1 + add_noise;
    % add variable mean offset between 5 and 25%
    m=mean(tmp1);
    add_mean =  m*.25;
    %add_mean=randsample(0:500,size(tmp1,2))/1e3;
    flip_sign = rand(size(add_mean));
    flip_sign(flip_sign>0.5)=1;
    flip_sign(flip_sign<=0.5)=-1;
    add_mean=add_mean.*flip_sign+m;
    tmp1m = tmp1n + add_mean;
    %tmp1m = (tmp1m-min(tmp1m(:)))/(max(tmp1m(:))-min(tmp1m(:)));

    % lmp
    tmp2 = tmp(:,129:256);
    % add variable noise
    var_noise=0.7;
    %var_noise=randsample(400:1200,size(tmp2,2))/1e3;
    add_noise=randn(size(tmp2)).*std(tmp2).*var_noise;
    tmp2n = tmp2 + add_noise;
    % add variable mean offset between 5 and 25%
    m=mean(tmp2);
    add_mean =  m*.35;
    %add_mean=randsample(0:500,size(tmp2,2))/1e3;
    flip_sign = rand(size(add_mean));
    flip_sign(flip_sign>0.5)=1;
    flip_sign(flip_sign<=0.5)=-1;
    add_mean=add_mean.*flip_sign+m;
    tmp2m = tmp2n + add_mean;
    %tmp2m = (tmp2m-min(tmp2m(:)))/(max(tmp2m(:))-min(tmp2m(:)));

    %     %lg
    %     tmp3 = tmp(:,257:384);
    %     % add noise var
    %     add_noise=randn(size(tmp3)).*std(tmp3).*.795;
    %     tmp3n = tmp3 + add_noise;
    %     % add mean offset by 20%
    %     m=mean(tmp3);
    %     add_mean =  m*.2;
    %     flip_sign = rand(size(add_mean));
    %     flip_sign(flip_sign>0.5)=1;
    %     flip_sign(flip_sign<=0.5)=-1;
    %     add_mean=add_mean.*flip_sign+m;
    %     tmp3m = tmp3n + add_mean;
    %     tmp3m = (tmp3m-min(tmp3m(:)))/(max(tmp3m(:))-min(tmp3m(:)));

    %tmp=[tmp1m tmp2m tmp3m]';
    tmp=[tmp1m tmp2m]';

    XTrain=cat(1,XTrain,tmp);
    YTrain = cat(1,YTrain,t_id);
end


% data augmentation for PCA
aug_idx = randperm(length(XTrain));
for i=1:length(aug_idx)
    tmp = XTrain{aug_idx(i)}';
    t_id=categorical(YTrain(aug_idx(i)));

    % hG and lmp
    tmp1 = tmp(:,1:end);
    % add variable noise
    %var_noise=randsample(400:1200,size(tmp1,2))/1e3;
    var_noise=0.7;
    add_noise=randn(size(tmp1)).*std(tmp1).*var_noise;
    tmp1n = tmp1 + add_noise;
    % add variable mean offset between 5 and 25%
    m=mean(tmp1);
    add_mean =  m*.25;
    %add_mean=randsample(0:500,size(tmp1,2))/1e3;
    flip_sign = rand(size(add_mean));
    flip_sign(flip_sign>0.5)=1;
    flip_sign(flip_sign<=0.5)=-1;
    add_mean=add_mean.*flip_sign+m;
    tmp1m = tmp1n + add_mean;
    %tmp1m = (tmp1m-min(tmp1m(:)))/(max(tmp1m(:))-min(tmp1m(:)));
    
    tmp=[tmp1m]';

    XTrain=cat(1,XTrain,tmp);
    YTrain = cat(1,YTrain,t_id);
end

% implement label smoothing to see how that does
%
% tmp=str2num(cell2mat(tmp));
% a=0.01;
% tmp1 = (1-a).*tmp + (a)*(1/7);
% clear YTrain
% YTrain = tmp1;
% YTrain =categorical(YTrain);
%clear condn_data_new


% load pretrained LSTM structure
%load net_bilstm
load net_bilstm_20220929
net_bilstm=net_bilstm_20220929;
layers = net_bilstm.Layers;

% define training options
batch_size=128;
val_freq = floor(length(XTrain)/batch_size);
options = trainingOptions('adam', ...
    'MaxEpochs',120, ...
    'MiniBatchSize',batch_size, ...
    'GradientThreshold',10, ...
    'Verbose',true, ...
    'ValidationFrequency',val_freq,...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest,YTest},...
    'ValidationPatience',6,...
    'Plots','training-progress',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'OutputNetwork','best-validation-loss',...
    'LearnRateDropPeriod',25,...
    'InitialLearnRate',5e-4);

% train the model
clear net
net = trainNetwork(XTrain,YTrain,layers,options);

% save the network
net_bilstm_robot_20220929 = net;
save net_bilstm_robot_20220929 net_bilstm_robot_20220929


net_bilstm_robot_20220824_early_stop = net_bilstm_robot_20220824C
save net_bilstm_robot_20220824_early_stop net_bilstm_robot_20220824_early_stop


%%% TRAINING LSTM FROM SCRATCH
% specify lstm structure
inputSize = 256;
numHiddenUnits1 = [  90 120 128 128 325 64];
drop1 = [ 0.2 0.2 0.3  0.3 0.4 0.3];
numClasses = 7;
for i=3%1:length(drop1)
    numHiddenUnits=numHiddenUnits1(i);
    drop=drop1(i);
    layers = [ ...
        sequenceInputLayer(inputSize)  
        fullyConnectedLayer(150)
        leakyReluLayer
        dropoutLayer(drop)
        bilstmLayer(numHiddenUnits,'OutputMode','sequence')
        dropoutLayer(drop)
        layerNormalizationLayer
        gruLayer(numHiddenUnits/2,'OutputMode','last')
        dropoutLayer(drop)        
        fullyConnectedLayer(25)
        leakyReluLayer
        batchNormalizationLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];



    % options
    batch_size=256;
    val_freq = floor(length(XTrain)/batch_size);
    options = trainingOptions('adam', ...
        'MaxEpochs',120, ...
        'MiniBatchSize',batch_size, ...
        'GradientThreshold',10, ...
        'Verbose',true, ...
        'ValidationFrequency',val_freq,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XTest,YTest},...
        'ValidationPatience',6,...
        'Plots','training-progress',...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',0.1,...
        'OutputNetwork','best-validation-loss',...
        'LearnRateDropPeriod',50,...
        'InitialLearnRate',0.001);

    % train the model
    net = trainNetwork(XTrain,YTrain,layers,options);
end


% save the network
net_bilstm_robotOnly_20220929 = net;
save net_bilstm_robotOnly_20220929 net_bilstm_robotOnly_20220929


%
% net_800 =net;
% save net_800 net_800

%net_bilstm_lg = net;
%save net_bilstm_lg net_bilstm_lg


%% TEST OUT FINE TUNED LSTM ON HELD OUT ROBOT BATCH/ARROW DATA

clear
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%load net_bilstm_stacked
%net_bilstm = net_bilstm_stacked;
%load net_bilstm_lg
%net_bilstm = net_bilstm_lg;
%load net_bilstm
load net_bilstm_robot_20220824
net_bilstm = net_bilstm_robot_20220824;
%load net_bilstm_20220824B
%net_bilstm = net_bilstm_20220824B;

addpath('C:\Users\nikic\Documents\MATLAB')
%filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220304\RealRobotBatch';
acc_mlp_days=[];
acc_days=[];
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20220930'};

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
Params.FilterBank(end+1).fpass = [30,36]; % lg1
Params.FilterBank(end+1).fpass = [36,42]; % lg2
Params.FilterBank(end+1).fpass = [42,50]; % lg3
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

for i=1:length(foldernames)
    disp(i)
    filepath = fullfile(root_path,foldernames{i},'RealRobot3D');
    [acc_lstm_sample,acc_mlp_sample,acc_lstm_trial,acc_mlp_trial]...
        = get_lstm_performance(filepath,net_bilstm,Params,lpFilt);

    acc_days = [acc_days diag(acc_lstm_sample)];
    acc_mlp_days = [acc_mlp_days diag(acc_mlp_sample)];
end


% plotting confusion matrix
figure;
subplot(1,2,1)
imagesc(acc_mlp_sample)
caxis([0 1])
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:7)
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
yticks(1:7)
yticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
set(gca,'FontSize',14)
subplot(1,2,2)
stem(diag(acc_mlp_sample),'LineWidth',2)
xlim([0.5 7.5])
ylabel('Accuracy')
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:7)
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
sgtitle('20220930 Arrow Experiment Sample Accuracy')

figure;
subplot(1,2,1)
imagesc(acc_mlp_trial)
caxis([0 1])
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:7)
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
yticks(1:7)
yticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
set(gca,'FontSize',14)
subplot(1,2,2)
stem(diag(acc_mlp_trial),'LineWidth',2)
xlim([0.5 7.5])
ylabel('Accuracy')
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:7)
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
sgtitle('20220930 Arrow Experiment Trial Accuracy')

%
figure;
tmp = [acc_days(:) acc_mlp_days(:)];
boxplot(tmp)
hold on
cmap=turbo(size(acc_days,2));
for i=1:size(acc_days,2)
    s=scatter(ones(1,1)+0.05*randn(1,1) , mean(acc_days(:,i))','MarkerEdgeColor',cmap(i,:),...
        'LineWidth',2);
    s.SizeData=100;
    s=scatter(2*ones(1,1)+0.05*randn(1,1) , mean(acc_mlp_days(:,i))',...
        'MarkerEdgeColor',cmap(i,:),'LineWidth',2);
    s.SizeData=100;
end
ylim([0.3 1])
xticks(1:2)
xticklabels({'stack biLSTM','MLP'})
legend({'0803','','0810','','0812'})
title('Center Out Robot')
ylabel('Accuracy of inidiv. samples at 5Hz')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
box off

% plotting the success of individual actions
figure;
hold on
for i=1:7
    %idx = i:7:21;
    idx=i;
    decodes = tmp(idx,:);
    h=bar(2*i-0.25,mean(decodes(:,1)));
    h1=bar(2*i+0.25,mean(decodes(:,2)));
    h.BarWidth=0.4;
    h.FaceColor=[0.2 0.2 0.7];
    h1.BarWidth=0.4;
    h1.FaceColor=[0.7 0.2 0.2];
    h.FaceAlpha=0.85;
    h1.FaceAlpha=0.85;

    %     s=scatter(ones(3,1)*2*i-0.25+0.05*randn(3,1),decodes(:,1),'LineWidth',2);
    %     s.CData = [0.2 0.2 0.7];
    %     s.SizeData=50;
    %
    %     s=scatter(ones(3,1)*2*i+0.25+0.05*randn(3,1),decodes(:,2),'LineWidth',2);
    %     s.CData = [0.7 0.2 0.2];
    %     s.SizeData=50;
end
xticks([2:2:14])
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
ylabel('Decoding Accuracy')
legend('LSTM','MLP')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)



%% TEST OUT FINE TUNED LSTM ON HELD OUT REAL ROBOT 3D DATA

clear;clc
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%load net_bilstm_stacked
%net_bilstm = net_bilstm_stacked;
%load net_bilstm_lg
%net_bilstm = net_bilstm_lg;
%load net_bilstm
%load net_bilstm_robot_20220824B
%net_bilstm = net_bilstm_robot_20220824B;
%load net_bilstm_20220824B
%net_bilstm = net_bilstm_20220824B;
load net_bilstm_robotOnly_20220929
net_bilstm = net_bilstm_robotOnly_20220929;

addpath('C:\Users\nikic\Documents\MATLAB')
%filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220304\RealRobotBatch';
acc_mlp_days=[];
acc_days=[];
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20220930'};

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
Params.FilterBank(end+1).fpass = [30,36]; % lg1
Params.FilterBank(end+1).fpass = [36,42]; % lg2
Params.FilterBank(end+1).fpass = [42,50]; % lg3
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

for i=1:length(foldernames)
    disp(i)
    filepath = fullfile(root_path,foldernames{i},'RealRobot3D');
    [acc_lstm_sample,acc_mlp_sample,acc_lstm_trial,acc_mlp_trial]...
        = get_lstm_performance_real_robot(filepath,net_bilstm,Params,lpFilt);

    acc_days = [acc_days diag(acc_lstm_sample)];
    acc_mlp_days = [acc_mlp_days diag(acc_mlp_sample)];
end


% plotting confusion matrix
figure;
subplot(1,2,1)
imagesc(acc_mlp_sample)
caxis([0 1])
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:7)
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
yticks(1:7)
yticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
set(gca,'FontSize',14)
subplot(1,2,2)
stem(diag(acc_mlp_sample),'LineWidth',2)
xlim([0.5 7.5])
ylabel('Accuracy')
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:7)
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
sgtitle('20220930 Arrow Experiment Sample Accuracy')

figure;
subplot(1,2,1)
imagesc(acc_mlp_trial)
caxis([0 1])
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:7)
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
yticks(1:7)
yticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
set(gca,'FontSize',14)
subplot(1,2,2)
stem(diag(acc_mlp_trial),'LineWidth',2)
xlim([0.5 7.5])
ylabel('Accuracy')
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:7)
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
sgtitle('20220930 Arrow Experiment Trial Accuracy')

%
figure;
tmp = [acc_days(:) acc_mlp_days(:)];
boxplot(tmp)
hold on
cmap=turbo(size(acc_days,2));
for i=1:size(acc_days,2)
    s=scatter(ones(1,1)+0.05*randn(1,1) , mean(acc_days(:,i))','MarkerEdgeColor',cmap(i,:),...
        'LineWidth',2);
    s.SizeData=100;
    s=scatter(2*ones(1,1)+0.05*randn(1,1) , mean(acc_mlp_days(:,i))',...
        'MarkerEdgeColor',cmap(i,:),'LineWidth',2);
    s.SizeData=100;
end
ylim([0.3 1])
xticks(1:2)
xticklabels({'stack biLSTM','MLP'})
legend({'0803','','0810','','0812'})
title('Center Out Robot')
ylabel('Accuracy of inidiv. samples at 5Hz')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
box off

% plotting the success of individual actions
figure;
hold on
for i=1:7
    %idx = i:7:21;
    idx=i;
    decodes = tmp(idx,:);
    h=bar(2*i-0.25,mean(decodes(:,1)));
    h1=bar(2*i+0.25,mean(decodes(:,2)));
    h.BarWidth=0.4;
    h.FaceColor=[0.2 0.2 0.7];
    h1.BarWidth=0.4;
    h1.FaceColor=[0.7 0.2 0.2];
    h.FaceAlpha=0.85;
    h1.FaceAlpha=0.85;

    %     s=scatter(ones(3,1)*2*i-0.25+0.05*randn(3,1),decodes(:,1),'LineWidth',2);
    %     s.CData = [0.2 0.2 0.7];
    %     s.SizeData=50;
    %
    %     s=scatter(ones(3,1)*2*i+0.25+0.05*randn(3,1),decodes(:,2),'LineWidth',2);
    %     s.CData = [0.7 0.2 0.2];
    %     s.SizeData=50;
end
xticks([2:2:14])
xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
ylabel('Decoding Accuracy')
legend('LSTM','MLP')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)


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



%% laplacian referencing code

clc;clear
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210526\Robot3DArrow\112357\Imagined\Data0005.mat')
chmap=TrialData.Params.ChMap;
temp=randn(128,10);
[xx yy] = size(TrialData.Params.ChMap);
new_temp=zeros(size(temp));
for k=1:size(temp,2)
    tmp1 = temp(1:128,k);tmp1 = tmp1(TrialData.Params.ChMap);    
    out_data=zeros(xx,yy);
    for i=1:xx
        for j=1:yy
            % get the neighbors
            i_nb = [i-1 i+1];
            j_nb = [j-1 j+1];            
            i_nb = i_nb(logical((i_nb>0) .* (i_nb<=8)));
            j_nb = j_nb(logical((j_nb>0) .* (j_nb<=16)));
            ref_ch_vals = [tmp1(i,[j_nb]) tmp1(i_nb,[j])'];
            out_data(i,j) = tmp1(i,j) - mean(ref_ch_vals);                        
        end
    end  
    new_temp(:,k) = out_data(:);
end

%% TEMP STUFF FOR PLOTTING CKA FOR HDOF PAPER

res=[b1';b2'];
m = mean(res,1);
figure;bar(0:5,m')
