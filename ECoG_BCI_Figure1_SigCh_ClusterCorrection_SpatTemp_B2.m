%% ERPs for B2 from sessions of online data  

clc;clear
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')

filepath ='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
cd(filepath)
folders = {'20210324'};%

% get the file names 
files=[];
for i=1:length(folders)
    full_path = fullfile(filepath,folders{i},'DiscreteArrow');
    tmp = findfiles('.mat',full_path,1)';
    for j=1:length(tmp)
        if ~isempty(regexp(tmp{j},'Data'))
            files=[files;tmp(j)];
        end
    end
end

files_motor = files(1:72);
files_tong = files(73:end);
[D1,D2,D3,D4,idx1,idx2,idx3,idx4] = load_erp_data_online_B2(files_motor);
[D5,D6,D7,D8,idx1,idx2,idx3,idx4] = load_erp_data_online_B2(files_tong);

% plot ERPs with all the imagined data 
load('ECOG_Grid_8596-002131.mat')


% plot the ERPs with bootstrapped C.I. shading
chMap=ecog_grid;


% plot the ERPs with bootstrapped C.I. shading
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
for i = 1:size(D1,1)
    [x y] = find(chMap==i);
    if x == 1
        axes(ha(y));
        %subplot(8, 16, y)
    else
        s = 16*(x-1) + y;
        axes(ha(s));
        %subplot(8, 16, s)
    end
    hold on
    erps =  squeeze(D1(i,:,:));
    
    chdata = erps;
    % zscore the data to the first 6 time-bins
    tmp_data=chdata(1:6,:);
    m = mean(tmp_data,1);% each trial 
    s = std(tmp_data,1);
    %m = mean(tmp_data(:));% overall trials
    %s = std(tmp_data(:));
    chdata = (chdata -m)./s;
    
    % get the confidence intervals
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(D1,2);
    [fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
        ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);
    hold on
    plot(m,'b')
    %plot(mb(25,:),'--b')
    %plot(mb(975,:),'--b')
    %hline(0)
    
    % shuffle the data for null confidence intervals
    tmp_mean=[];
    for j=1:1000
        %tmp = circshift(chdata,randperm(size(chdata,1),1));
        tmp = chdata;
        tmp(randperm(numel(chdata))) = tmp;
        tmp_data=tmp(1:6,:);
        m = mean(tmp_data(:));
        s = std(tmp_data(:));
        tmp = (tmp -m)./s;
        tmp_mean(j,:) = mean(tmp,2);
    end
    
    tmp_mean = sort(tmp_mean);
    %plot(tmp_mean(25,:),'--r')
    %plot(tmp_mean(975,:),'--r')
    [fillhandle,msg]=jbfill(tt,tmp_mean(25,:),tmp_mean(975,:)...
        ,[0.7 0.3 0.3],[0.7 0.3 0.3],1,.2);
    
    
    % statistical test
    % if the mean is outside confidence intervals in state 3
    m = mean(chdata,2);
    idx=13:23;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end

    [pfdr,~] = fdr(1-pval,0.05);
    % fdr approach
    res = sum((1-pval)<=pfdr);
    % nominal approach
    res=sum((1-pval)<=0.05);
    if res>=5
        suc=1;
    else
        suc=0;
    end
    
    % beautify
    ylabel (num2str(i))
    axis tight
    ylim([-2 2])    
    %set(gca,'LineWidth',1)
    %vline([time(2:4)])   
    h=vline(tim);
    %set(h,'LineWidth',1)
    set(h,'Color','k')
    h=hline(0);
    set(h,'LineWidth',1.5)    
    if i~=107
        yticklabels ''
        xticklabels ''
    else
        %xticks([tim])
        %xticklabels({'S1','S2','S3','S4'})
    end
    
    if suc==1
        box on
        set(gca,'LineWidth',2)
        set(gca,'XColor','g')
        set(gca,'YColor','g')
    end
    d = d+1;
end

% save
ERP_Data{1} = D1;
ERP_Data{2} = D2;
ERP_Data{3} = D3;
ERP_Data{4} = D4;
ERP_Data{5} = D5;
ERP_Data{6} = D6;
ImaginedMvmt = {'Rt. Thumb','Leg','Lt Thumb','Head','Tongue','Lips'};
save ERP_Data_20210324_beta_B2 -v7.3


ImaginedMvmt = {'Rt. Thumb','Leg','Lt Thumb','Head','Tongue','Lips'};

% get the average activity in M1 hand knob channels
hand_elec = [6 10 116 99 102 104 ];
roi_mean=[];
roi_dist_mean=[];
idx=13:22;
for i=1:length(ERP_Data)
    disp(i)
    data = ERP_Data{i};
    data = data(hand_elec,idx,:);
    data = squeeze(mean(data,2)); % time
    data = squeeze(mean(data,1)); % channels
    data = data(:);
    roi_mean(i) = mean(data);
%     if i==19
%         roi_mean(i)=0.9;
%     end
    roi_dist_mean(:,i) = sort(bootstrp(1000,@mean,data));
end
figure;bar(roi_mean)

y = roi_mean;
y=y';
errY(:,1) = roi_dist_mean(500,:)-roi_dist_mean(25,:);
errY(:,2) = roi_dist_mean(975,:)-roi_dist_mean(500,:);
figure;
barwitherr(errY, y);
xticks(1:6)
set(gcf,'Color','w')
set(gca,'FontSize',16)
set(gca,'LineWidth',1)
xticklabels(ImaginedMvmt)

%% SPATIOTEMPORAL CLUS CORRECTION ON THE BINNED DATA WITH ARTIFACT CORRECTION
% ERPs for B2 from sessions of online data  

clc;clear
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')


filepath ='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
cd(filepath)

%load ERP_Data_20210324_B2 
load ERP_Data_20210324_hg_B2
ImaginedMvmt = {'Rt. Thumb','Leg','Lt Thumb','Head','Tongue','Lips'};
D{1}=D1;
D{2}=D2;
D{3}=D3;
D{4}=D4;
D{5}=D5;
D{6}=D6;

% number of movements 
num_mvmts=[];
for i=1:length(D)
    num_mvmts(i) = size(D{i},3);
end
num_mvmts

load('ECOG_Grid_8596-002131.mat')
grid_layout=ecog_grid;
chMap=grid_layout;

neighb=zeros(size(grid_layout));
for i=1:numel(grid_layout)
    [x y]=find(grid_layout == i);
    Lx=[x-1 x+1];
    Ly=[y-1 y+1];
    Lx=Lx(logical((Lx<=size(grid_layout,1)) .* (Lx>0)));
    Ly=Ly(logical((Ly<=size(grid_layout,2)) .* (Ly>0)));
    temp=grid_layout(Lx,Ly);
    ch1=[grid_layout(x,Ly)';grid_layout(Lx,y);];
    neighb(i,ch1)=1;
end
figure;
imagesc(neighb)



idx = [1:length(ImaginedMvmt)];
%idx =  [1,10,30,25,20,28];
sig_ch_all=zeros(idx(end),128);
loop_iter=750;
tfce_flag=false;
for i=1:length(idx)        
    data = D{idx(i)}; % shld be time X channels X trials
    data = permute(data,[2,1,3]);
    t_scores=[];tboot=(zeros(size(data,2),size(data,1),loop_iter));%channels X time X loops
    p_scores=[];pboot=(zeros(size(data,2),size(data,1),loop_iter));    
    parfor ch=1:numel(chMap)
        disp(['movement ' num2str(i) ', Channel ' num2str(ch)])
        chdata = squeeze((data(:,ch,:)));

        % z-score to first 6 bins per trial
        m = mean(chdata(1:6,:),1);
        s = std(chdata(1:6,:),1);
        chdata = (chdata-m)./s;

        %z-score to first 6 bins averaged across all trials
        %tmp_data = chdata(1:6,:);
        %m=mean(tmp_data(:));
        %s=std(tmp_data(:));
        %chdata = (chdata-m)./s;

        % bad trial removal
        tmp_bad=zscore(chdata')';
        artifact_check = logical(abs(tmp_bad)>3.0);
        chdata(artifact_check)=NaN;        

        [h,p,ci,stats] = ttest(chdata');
        t = stats.tstat;
        if tfce_flag
            t(p>0.05)=0; % only if TFCE
        end
        t_scores(ch,:) = t;
        p_scores(ch,:) = p;
    

        % get the null t-statistics at each time-point thru bootstrap
        a = chdata';
        anew=a-nanmean(a); % centering
        asize=size(anew);
        for loop=1:loop_iter
            a1= anew(randi(asize(1),[asize(1) 1]),:); % sample with replacement
            [h0 p0 ci stats0]=ttest(a1); % run the t-test
            t0=stats0.tstat;
            if tfce_flag
                t0(p0>0.05)=0; % only if TFCE
            end
            tboot(ch,:,loop)=t0;
            pboot(ch,:,loop)=p0;
        end
    end

    % TFCE
    if tfce_flag
        E=1;H=2;dh=0.2;
        [tfce_score,~] = limo_tfce(2,t_scores,neighb,1,E,H,dh);
        [tfce_score_boot,~] = limo_tfce(2,tboot,neighb,1,E,H,dh);

        % get the null distribution of tfce score from each bootstrap
        tfce_boot=[];
        for loop=1:size(tfce_score_boot,3)
            a=squeeze(tfce_score_boot(:,:,loop));
            tfce_boot(loop) = max(a(:));
        end

        % threshold the true tfce scores with the null distribution
        tfce_boot=sort(tfce_boot);tfce_score1=tfce_score;
        thresh = tfce_boot(round(0.95*length(tfce_boot)));
        tfce_score1(tfce_score1<thresh)=0;
        figure;
        subplot(3,1,1)
        tt=linspace(-3,4,size(t_scores,2));
        imagesc(tt,1:128,t_scores);
        title('Uncorrected for multiple comparisons')
        subplot(3,1,2)
        imagesc(tt,1:128,tfce_score1)
        ylabel('Channels')
        xlabel('Time')
        title('Spatiotemp. multiple comparison corrected')

        % plot the significant channels
        a=tfce_score1;
        aa=sum(a(:,3000:6000),2);
        sig_ch_idx = find(aa>0);
        sig_ch = zeros(numel(chMap),1);
        sig_ch(sig_ch_idx)=1;
        sig_ch_all(i,:) = sig_ch;
        subplot(3,1,3);
        imagesc(sig_ch(chMap))
        title('Sig channels 0 to 3s')
        sgtitle(ImaginedMvmt{i})
        axis off
        set(gcf,'Color','w')

    else
        % 2D spatiotemporal cluster correction
        LIMO.data.chanlocs=[];
        LIMO.data.neighbouring_matrix=neighb;
        [mask,cluster_p,max_th] = ...
            limo_clustering((t_scores.^2),p_scores,...
            (tboot.^2),pboot,LIMO,2,0.05,0);
        figure;subplot(3,1,1)
        tt=linspace(-2,3,size(t_scores,2));
        imagesc(tt,1:128,t_scores);
        title('Uncorrected for multiple comparisons')    
        ylabel('Channels')
        xlabel('Time')
        subplot(3,1,2)
        t_scores1=t_scores;
        t_scores1(mask==0)=0;
        imagesc(tt,1:128,t_scores1);
        title('Corrected for multiple comparisons')        
        ylabel('Channels')
        xlabel('Time')
        a=mask;
        aa=sum(a(:,11:25),2);
        sig_ch_idx = find(aa>0);
        sig_ch = zeros(numel(grid_layout),1);
        sig_ch(sig_ch_idx)=1;
        sig_ch_all(i,:) = sig_ch;
        subplot(3,1,3);
        imagesc(sig_ch(grid_layout))
        title('Sig channels 0 to 3s')
        sgtitle(ImaginedMvmt{i})
        axis off
        set(gcf,'Color','w')
        colorbar
    end
end
% 
save B2_hG_Imagined_SpatTemp_New_New_ArtfCorr sig_ch_all -v7.3

%% LOOKING AT OVERALL SPATIAL MAPS PER EFFECTOR

clc;clear
close all
load B2_hG_Imagined_SpatTemp_New_New_ArtfCorr

ImaginedMvmt = {'Rt. Thumb','Leg','Lt Thumb','Head','Tongue','Lips'};

load('ECOG_Grid_8596-002131.mat')

imaging_B2;close all
chMap =  ecog_grid;

% rt hand, both as image and as electrode size
rt_hand = sig_ch_all([1],:);
rt_hand = mean(rt_hand,1);
rt_hand = rt_hand*10;
figure;imagesc(rt_hand(chMap));
colormap parula
plot_elec_wts_B2(rt_hand,cortex,elecmatrix,chMap)
title('Right Hand')

% lt hand
lt_hand = sig_ch_all(3,:);
lt_hand = mean(lt_hand,1);
lt_hand = (lt_hand)*10;
figure;imagesc(lt_hand(chMap));
colormap parula
plot_elec_wts_B2(lt_hand,cortex,elecmatrix,chMap)
title('Left Hand')

% distal
distal = sig_ch_all([2],:);
distal = mean(distal,1);
distal = (distal)*10;
figure;imagesc(distal(chMap));
colormap parula
plot_elec_wts_B2(distal,cortex,elecmatrix,chMap)
title('Distal')

% Face (heads/lips/tongue)
face = sig_ch_all([4 5:6 ],:);
face = mean(face,1);
face = (face)*10;
figure;imagesc(face(chMap)/10);
colormap parula
caxis([0 1])
plot_elec_wts_B2(face,cortex,elecmatrix,chMap)
title('Face')

% plot as filled circles
clear wts;
wts(1,:) = rt_hand;
wts(2,:) = lt_hand;
wts(3,:) = distal;
wts(4,:) = face;
%wts_alpha(1,:)=rt_hand./10;
% scale between 50 and 250
wts=50 +  (300-50)*( (wts - min(wts(:)))./( max(wts(:))-min(wts(:))));
for j=1:size(wts,1)
    wts1=wts(j,:);
    figure
    ha=tight_subplot(8,16);
    d = 1;
    set(gcf,'Color','w')
    set(gcf,'WindowState','maximized')
    %alp = wts_alpha(j,:);
    %alp(alp==0)=0.05;
    for ch=1:length(wts1)

        [x y] = find(chMap==ch);
        if x == 1
            axes(ha(y));
            %subplot(8, 16, y)
        else
            s = 16*(x-1) + y;
            axes(ha(s));
            %subplot(8, 16, s)
        end

        if wts1(ch)==50
            col = 'k';
        else
            col = 'r';
        end

        plot(0,0,'.','MarkerSize',wts1(ch),'Color',col);
        %scatter(0,0,wts1(ch),col,'filled')
        %alpha(alp(ch));
        
        axis off
    end
end




%% OLD STUFF ON HOW TO GET THE ERP DATA FROM BINNED DATA 

filepath ='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
cd(filepath)
folders = {'20210331'};%

% get the file names 
files=[];
for i=1:length(folders)
    full_path = fullfile(filepath,folders{i},'DiscreteArrow');
    tmp = findfiles('.mat',full_path,1)';
    for j=1:length(tmp)
        if ~isempty(regexp(tmp{j},'Data'))
            files=[files;tmp(j)];
        end
    end
end

files_motor = files(1:72);
files_tong = files(73:end);
[D1,D2,D3,D4,idx1,idx2,idx3,idx4] = load_erp_data_online_B2(files);
%[D5,D6,D7,D8,idx1,idx2,idx3,idx4] = load_erp_data_online_B2(files_tong);

% plot ERPs with all the imagined data 
load('ECOG_Grid_8596-002131.mat')


% plot the ERPs with bootstrapped C.I. shading
chMap=ecog_grid;


% plot the ERPs with bootstrapped C.I. shading
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
for i = 1:size(D1,1)
    [x y] = find(chMap==i);
    if x == 1
        axes(ha(y));
        %subplot(8, 16, y)
    else
        s = 16*(x-1) + y;
        axes(ha(s));
        %subplot(8, 16, s)
    end
    hold on
    erps =  squeeze(D1(i,:,:));
    
    chdata = erps;
    % zscore the data to the first 6 time-bins
    tmp_data=chdata(1:6,:);
    m = mean(tmp_data(:));
    s = std(tmp_data(:));
    chdata = (chdata -m)./s;
    
    % get the confidence intervals
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(D1,2);
    [fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
        ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);
    hold on
    plot(m,'b')
    %plot(mb(25,:),'--b')
    %plot(mb(975,:),'--b')
    %hline(0)
    
    % shuffle the data for null confidence intervals
    tmp_mean=[];
    for j=1:1000
        %tmp = circshift(chdata,randperm(size(chdata,1),1));
        tmp = chdata;
        tmp(randperm(numel(chdata))) = tmp;
        tmp_data=tmp(1:6,:);
        m = mean(tmp_data(:));
        s = std(tmp_data(:));
        tmp = (tmp -m)./s;
        tmp_mean(j,:) = mean(tmp,2);
    end
    
    tmp_mean = sort(tmp_mean);
    %plot(tmp_mean(25,:),'--r')
    %plot(tmp_mean(975,:),'--r')
    [fillhandle,msg]=jbfill(tt,tmp_mean(25,:),tmp_mean(975,:)...
        ,[0.7 0.3 0.3],[0.7 0.3 0.3],1,.2);
    
    
    % statistical test
    % if the mean is outside confidence intervals in state 3
    m = mean(chdata,2);
    idx=13:23;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end

    [pfdr,~] = fdr(1-pval,0.05);
    % fdr approach
    res = sum((1-pval)<=pfdr);
    % nominal approach
    res=sum((1-pval)<=0.05);
    if res>=5
        suc=1;
    else
        suc=0;
    end
    
    % beautify
    ylabel (num2str(i))
    axis tight
    ylim([-2 2])    
    %set(gca,'LineWidth',1)
    %vline([time(2:4)])   
    h=vline(tim);
    %set(h,'LineWidth',1)
    set(h,'Color','k')
    h=hline(0);
    set(h,'LineWidth',1.5)    
    if i~=107
        yticklabels ''
        xticklabels ''
    else
        %xticks([tim])
        %xticklabels({'S1','S2','S3','S4'})
    end
    
    if suc==1
        box on
        set(gca,'LineWidth',2)
        set(gca,'XColor','g')
        set(gca,'YColor','g')
    end
    d = d+1;
end

% save
ERP_Data{1} = D1;
ERP_Data{2} = D2;
ERP_Data{3} = D3;
ERP_Data{4} = D4;
ERP_Data{5} = D5;
ERP_Data{6} = D6;
save ERP_Data_20210324_B2 -v7.3
ImaginedMvmt = {'Rt. Thumb','Leg','Lt Thumb','Head','Tongue','Lips'};

% get the average activity in M1 hand knob channels
hand_elec = [6 10 116 99 102 104 ];
roi_mean=[];
roi_dist_mean=[];
idx=13:22;
for i=1:length(ERP_Data)
    disp(i)
    data = ERP_Data{i};
    data = data(hand_elec,idx,:);
    data = squeeze(mean(data,2)); % time
    data = squeeze(mean(data,1)); % channels
    data = data(:);
    roi_mean(i) = mean(data);
%     if i==19
%         roi_mean(i)=0.9;
%     end
    roi_dist_mean(:,i) = sort(bootstrp(1000,@mean,data));
end
figure;bar(roi_mean)

y = roi_mean;
y=y';
errY(:,1) = roi_dist_mean(500,:)-roi_dist_mean(25,:);
errY(:,2) = roi_dist_mean(975,:)-roi_dist_mean(500,:);
figure;
barwitherr(errY, y);
xticks(1:6)
set(gcf,'Color','w')
set(gca,'FontSize',16)
set(gca,'LineWidth',1)
xticklabels(ImaginedMvmt)


%% ERPs B2 with higher samplign rate (doesnt seem to work)

clc;clear
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')

filepath ='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
cd(filepath)
folders = {'20210324'};

% get the file names 
files=[];
for i=1:length(folders)
    full_path = fullfile(filepath,folders{i},'DiscreteArrow');
    tmp = findfiles('.mat',full_path,1)';
    for j=1:length(tmp)
        if ~isempty(regexp(tmp{j},'Data'))
            files=[files;tmp(j)];
        end
    end
end

ImaginedMvmt = {'Right Thumb','Leg','Left Thumb','Head'};



% load the ERP data for each target
ERP_Data={};
for i=1:length(ImaginedMvmt)
    ERP_Data{i}=[];
end

% TIMING INFORMATION FOR THE TRIALS
Params.InterTrialInterval = 1; % rest period between trials 
Params.InstructedDelayTime = 1; % only arrow environment
Params.CueTime = 1; % target appears
Params.ImaginedMvmtTime = 8; % Go time 

% low pass filter of raw
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',5,'PassbandRipple',0.2, ...
    'SampleRate',1e3);


% 
% % log spaced hg filters
% Params.Fs = 1000;
% Params.FilterBank(1).fpass = [70,77];   % high gamma1
% Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
% Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
% Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
% Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
% Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
% Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
% Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
% Params.FilterBank(end+1).fpass = [0.5,4]; % delta
% Params.FilterBank(end+1).fpass = [13,19]; % beta1
% Params.FilterBank(end+1).fpass = [19,30]; % beta2
% 
% % compute filter coefficients
% for i=1:length(Params.FilterBank),
%     [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
%     Params.FilterBank(i).b = b;
%     Params.FilterBank(i).a = a;
% end

for i=1:length(files)
    disp(i/length(files)*100)
    load(files{i});
    features  = TrialData.BroadbandData;
    %features = cell2mat(features');
    Params = TrialData.Params;

    fs = TrialData.Params.UpdateRate;
    kinax = TrialData.TaskState;
    state1 = find(kinax==1);
    state2 = find(kinax==2);
    state3 = find(kinax==3);
    state4 = find(kinax==4);
    tmp_data = features(:,state3);
    idx1= ones(length(state1),1);
    idx2= 2*ones(length(state2),1);
    idx3= 3*ones(length(state3),1);
    idx4= 4*ones(length(state4),1);   
    fidx=9:16;%filters idx, 9-15 is hg, 1 is delta and 4:5 is beta

    %%%%% state 1 
    % extract state 1 data hG and resample it to 1s
    features1 = cell2mat(features(state1)');
    filtered_data=[];
    k=1;
    for ii=fidx
        filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
            Params.FilterBank(ii).b, ...
            Params.FilterBank(ii).a, ...
            features1)));
        k=k+1;
    end    
    if length(size(filtered_data))>2
        features1 = squeeze(mean(filtered_data,3));
    else
        features1 = filtered_data;
    end    

    % interpolation
    if size(features1,1)~=1000
%         tb = [1:size(features1,1)]*1e-3;
%         t = [1:1000]*1e-3;

        tb = [1:size(features1,1)]*1e-3;
        t = [1:1000]*1e-3;
        tb = tb*t(end)/tb(end);
        features1 = interp1(tb,features1,t);
    end

    %%%%% state 2 
    % extract state 1 data hG and resample it to 1s
    features2 = cell2mat(features(state2)');
    filtered_data=[];
    k=1;
    for ii=fidx
        filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
            Params.FilterBank(ii).b, ...
            Params.FilterBank(ii).a, ...
            features2)));
        k=k+1;
    end    
    if length(size(filtered_data))>2
        features2 = squeeze(mean(filtered_data,3));
    else
        features2 = filtered_data;
    end
    % interpolation
    if size(features2,1)~=1000
        tb = [1:size(features2,1)]*1e-3;
        t = [1:1000]*1e-3;
        tb = tb*t(end)/tb(end);
        features2 = interp1(tb,features2,t);       
    end

    %%%%% state 3
    % extract state 1 data hG and resample it to 3s
    features3 = cell2mat(features(state3)');
    filtered_data=[];
    k=1;
    for ii=fidx
        filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
            Params.FilterBank(ii).b, ...
            Params.FilterBank(ii).a, ...
            features3)));
        k=k+1;
    end       
    if length(size(filtered_data))>2
        features3 = squeeze(mean(filtered_data,3));
    else
        features3 = filtered_data;
    end
    % interpolation
    if size(features3,1)~=3000
        tb = [1:size(features3,1)]*1e-3;
        t = [1:3000]*1e-3;
        tb = tb*t(end)/tb(end);
        features3 = interp1(tb,features3,t);
    end

    %%%%% state 4
    % extract state 1 data hG and resample it to 1s
    features4 = cell2mat(features(state4)');
    filtered_data=[];
    k=1;
    for ii=fidx
        filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
            Params.FilterBank(ii).b, ...
            Params.FilterBank(ii).a, ...
            features4)));
        k=k+1;
    end        
    if length(size(filtered_data))>2
        features4 = squeeze(mean(filtered_data,3));
    else
        features4 = filtered_data;
    end
    % interpolation
    if size(features4,1)~=1000
        tb = [1:size(features4,1)]*1e-3;
        t = [1:1000]*1e-3;
        tb = tb*t(end)/tb(end);
        features4 = interp1(tb,features4,t);
    end
    
    % now stitch the raw data back together
    data = [features1;features2;features3;features4];    
    data = smoothdata(data,'movmean',25);
    figure;imagesc(data)
    title(num2str(i))

    % now z-score to the first 1s of data
    m = mean(data(1:1000,:));
    s = std(data(1:1000,:));
    data = (data-m)./s;

    targetID = TrialData.TargetID;
    tmp = ERP_Data{targetID};
    tmp = cat(3,tmp,data);
    ERP_Data{targetID} = tmp;
end

%save high_res_erp_beta_data_4Dir -v7.3
%save high_res_erp_delta_data_4Dir -v7.3
save high_res_erp_hg_data_4Dir -v7.3


% plot the ERPs for hg for instance
load('ECOG_Grid_8596-002131.mat')
chMap=ecog_grid;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([1000 1000 3000 1000]);
D1 = ERP_Data{2};
D1 = permute(D1,[2 1 3]);
for i = 1:size(D1,1)
    [x y] = find(chMap==i);
    if x == 1
        axes(ha(y));
        %subplot(8, 16, y)
    else
        s = 16*(x-1) + y;
        axes(ha(s));
        %subplot(8, 16, s)
    end
    hold on
    erps =  squeeze(D1(i,:,:));    
    chdata = erps;    
    m=mean(chdata);
    [c,s,l]=pca(chdata);
    chdata = m + (s(:,2:end)*c(:,2:end)');

    % baseline to first 1000ms
    tmp = chdata(1:1000,:);
    m = mean(tmp,1);
    s = std(tmp);
    chdata = (chdata-m)./s;
    
    % get the confidence intervals
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(D1,2);
    [fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
        ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);
    hold on
    plot(m,'b')
    %plot(mb(25,:),'--b')
    %plot(mb(975,:),'--b')
    %hline(0)
    
    % shuffle the data for null confidence intervals
%     tmp_mean=[];
%     for j=1:1000
%         %tmp = circshift(chdata,randperm(size(chdata,1),1));
%         tmp = chdata;
%         tmp(randperm(numel(chdata))) = tmp;
%         tmp_data=tmp(1:8,:);
%         m = mean(tmp_data(:));
%         s = std(tmp_data(:));
%         tmp = (tmp -m)./s;
%         tmp_mean(j,:) = mean(tmp,2);
%     end
%     
%     tmp_mean = sort(tmp_mean);
%     %plot(tmp_mean(25,:),'--r')
%     %plot(tmp_mean(975,:),'--r')
%     [fillhandle,msg]=jbfill(tt,tmp_mean(25,:),tmp_mean(975,:)...
%         ,[0.7 0.3 0.3],[0.7 0.3 0.3],1,.2);
%     
%     
%     % statistical test
%     % if the mean is outside confidence intervals in state 3
%     m = mean(chdata,2);
%     idx=13:27;
%     mstat = m((idx));
%     pval=[];
%     for j=1:length(idx)
%         pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
%     end
%     
%     res=sum((1-pval)<=0.05);
%     if res>=7
%         suc=1;
%     else
%         suc=0;
%     end
%     
%     % beautify
%     ylabel (num2str(i))
%     axis tight
%     ylim([-2 2])    
%     %set(gca,'LineWidth',1)
%     %vline([time(2:4)])   
%     h=vline(tim);
%     %set(h,'LineWidth',1)
%     set(h,'Color','k')
%     h=hline(0);
%     set(h,'LineWidth',1.5)    
%     if i~=102
%         yticklabels ''
%         xticklabels ''
%     else
%         %xticks([tim])
%         %xticklabels({'S1','S2','S3','S4'})
%     end
%     
%     if suc==1
%         box on
%         set(gca,'LineWidth',2)
%         set(gca,'XColor','g')
%         set(gca,'YColor','g')
%     end
    d = d+1;
    ylim([-2 2])
end
