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







