% main code for the hand project
%
% this includes the data recently collected with multipe sequences of hand
% movements

% overall, the methods to do are:
% covariance matrix and reimann classifiers of the hand actions
% covariance matrix and then using a GRU for low-D representation
% maybe a variational autoencoder for classification? Time-series
% traveling waves and seeing differences
% travling waves with a transformer


%% SESSION DATA FOR HAND EXPERIMENTS B3

clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
cd(root_path)

%day1
session_data(1).Day = '20230510';
session_data(1).folders = {'114718','115447','120026','120552','121111','121639',...
    '122957','123819','124556','125329','130014',...
    '130904'};
session_data(1).folder_type={'I','I','I','I','I','I','O','O','O','O',...
    'O','B'};
session_data(1).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am'};

%day2
session_data(2).Day = '20230511';
session_data(2).folders = {'113750','114133','114535','115215','115650','120107',...
    '120841','121228','121645','122024',...
    '122813','123125','123502'};
session_data(2).folder_type={'I','I','I','I','I','I','O','O','O','O',...
    'B','B','B'};
session_data(2).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am','am'};

%day3
session_data(3).Day = '20230518';
session_data(3).folders = {'114942','115609','120009','120434','120825','121322',...
    '121645',...
    '122444','122837','123147','123506'...
    '124013','124254','124552'};
session_data(3).folder_type={'I','I','I','I','I','I','I','O','O','O','O'...
    'B','B','B'};
session_data(3).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am','am','am'};


save session_data_B3_Hand session_data

%% PERFORMANCE IMAGINED - ONLINE- BATCH FOR B3 HAND EXPERIMENTS

clc;clear
close all
clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3_Hand
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=5;
plot_true=true;
acc_batch_days_overall=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    %disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% cross_val classification accuracy for imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid);
    % save the data
    filename = ['condn_data_Hand_B3_ImaginedTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = ...
        accuracy_imagined_data_Hand_B3(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_imagined)))])
        xticks(1:12)
        yticks(1:12)
        xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
        yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
    end
    acc_imagined_days(:,i) = diag(acc_imagined);


    %%%%%% get classification accuracy for online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_online = accuracy_online_data_Hand(files,12);
    if plot_true
        figure;imagesc(acc_online)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_online)))])
        xticks(1:12)
        yticks(1:12)
        xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
        yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
    end
    acc_online_days(:,i) = diag(acc_online);


    %%%%%% classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_batch = accuracy_online_data_Hand(files,12);
    if plot_true
        figure;imagesc(acc_batch)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_batch)))])
        xticks(1:12)
        yticks(1:12)
        xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
        yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})       

    end
    acc_batch_days(:,i) = diag(acc_batch);
    acc_batch_days_overall(:,:,i)=acc_batch;
end




%% STEP 1: LOOK AT COVARIANCE MATRIX

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


