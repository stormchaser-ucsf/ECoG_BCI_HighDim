%% ERPs of imagined actions higher sampling rate (MAIN B1)
% using hG and LMP, beta etc.

clc;clear
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')
%addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4\limo_cluster_functions')



root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20211201','20211203','20211206','20211208','20211215','20211217',...
    '20220126','20220223','20220225'};

cd(root_path)

files=[];
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
        files = [files;findfiles('',filepath)'];
    end
end

ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Rotate Left Wrist','Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};



% load the ERP data for each target
ERP_Data={};
for i=1:length(ImaginedMvmt)
    ERP_Data{i}=[];
end

% TIMING INFORMATION FOR THE TRIALS
Params.InterTrialInterval = 3; % rest period between trials 
Params.InstructedDelayTime = 4; % text appears telling subject which action to imagine
Params.CueTime = 2; % A red square; subject has to get ready
Params.ImaginedMvmtTime = 4; % A green square, subject has actively imagine the action

% low pass filter of raw
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',5,'PassbandRipple',0.2, ...
    'SampleRate',1e3);
% 
% % 
% % % 
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
% % 
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
    features = cell2mat(features');
    Params = TrialData.Params;




    %get hG through filter bank approach
    filtered_data=zeros(size(features,1),size(features,2),1);
    k=1;
    for ii=1% 9:16is hG, 4:5 is beta, 1 is delta
        filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
            Params.FilterBank(ii).b, ...
            Params.FilterBank(ii).a, ...
            features)));
        k=k+1;
    end
    %tmp_hg = squeeze(mean(filtered_data.^2,3));
    tmp_hg = squeeze(mean(filtered_data,3));

    % low pass filter the data or low pass filter hG data
    %features1 = [randn(4000,128);features;randn(4000,128)];
%     features1 = [std(tmp_hg(:))*randn(4000,128) + mean(tmp_hg);...
%         tmp_hg;...
%         std(tmp_hg(:))*randn(4000,128) + mean(tmp_hg)];
%     tmp_hg = ((filtfilt(lpFilt,features1)));
%     %tmp_hg = abs(hilbert(filtfilt(lpFilt,features1)));
%     tmp_hg = tmp_hg(4001:end-4000,:);

    task_state = TrialData.TaskState;
    idx=[];
    for ii=1:length(task_state)
        tmp = TrialData.BroadbandData{ii};
        idx = [idx;task_state(ii)*ones(size(tmp,1),1)];
    end    
    
    % z-score to 1s before the get ready symbol
    fidx2 =  find(idx==2);fidx2=fidx2(1);
    fidx2 = fidx2+[-1000:0];
    m = mean(tmp_hg(fidx2,:));
    s = std(tmp_hg(fidx2,:));
    fidx = [fidx2 fidx2(end)+[1:7000]];

    tmp_hg_epoch  = tmp_hg(fidx,:);
    tmp_hg_epoch = (tmp_hg_epoch-m)./s;

    % downsample to 200Hz
    %tmp_lp = resample(tmp_lp,200,800);
%     tmp_hg_epoch1=[];
%     for j=1:size(tmp_hg_epoch,2);
%         tmp_hg_epoch1(:,j) = decimate(tmp_hg_epoch(:,j),5);
%     end

    features = tmp_hg_epoch;

    for j=1:length(ImaginedMvmt)
        if strcmp(ImaginedMvmt{j},TrialData.ImaginedAction)
            tmp = ERP_Data{j};
            tmp = cat(3,tmp,features);
            ERP_Data{j}=tmp;
            break
        end
    end
end

%save high_res_erp_beta_imagined_data -v7.3
%save high_res_erp_LMP_imagined_data -v7.3
%save high_res_erp_imagined_data -v7.3
%save high_res_delta_erp_imagined_data -v7.3
%save high_res_erp_hgLFO_imagined_data -v7.3
save high_res_erp_beta_imagined_data -v7.3


% get the number of epochs used
ep=[];
for i=1:length(ERP_Data)
    tmp = ERP_Data{i};
    ep(i) = size(tmp,3);
end
figure;stem(ep)
xticks(1:32)
xticklabels(ImaginedMvmt)





%% plot ERPs at all channels with tests for significance
idx = [1:30];
%idx =  [1,10,30,25,20,28];
data = ERP_Data{1};
chMap=TrialData.Params.ChMap;
%ch=3;
sig_ch=zeros(30,128);
for i=1:length(idx)    
    figure
    ha=tight_subplot(8,16);
    d = 1;
    set(gcf,'Color','w')
    set(gcf,'WindowState','maximized')
    data = ERP_Data{idx(i)};
    for ch=1:numel(chMap)
        disp(['movement ' num2str(i) ' channel ' num2str(ch)])

        [x y] = find(chMap==ch);
        if x == 1
            axes(ha(y));
            %subplot(8, 16, y)
        else
            s = 16*(x-1) + y;
            axes(ha(s));
            %subplot(8, 16, s)
        end
        hold on

        chdata = squeeze((data(:,ch,:)));
        
        % bad trial removal
        tmp_bad=zscore(chdata')';
        artifact_check=((tmp_bad)>=5) + (tmp_bad<=-4);
        good_idx = find(sum(artifact_check)==0);
        chdata = chdata(:,good_idx);      

        % significance testing via temporal cluster correction
        %[t,pval]=temporal_clust_1D(chdata',0.05);


        % confidence intervals around the mean
        m=mean(chdata,2);
        opt=statset('UseParallel',true);
        mb = sort(bootstrp(1000,@mean,chdata','Options',opt));
        tt=linspace(-1,7,size(data,1));
        %figure;        
        [fillhandle,msg]=jbfill(tt,(mb(25,:)),(mb(975,:))...
        ,[0.5 0.5 0.5],[0.5 0.5 0.5],1,.4);
        hold on        
        plot(tt,(m),'k','LineWidth',1)
        axis tight
        %plot(tt,mb(25,:),'--k','LineWidth',.25)
        %plot(tt,mb(975,:),'--k','LineWidth',.25)
        % beautify
        ylabel(num2str(ch))        
        ylim([-1.5 2])
        yticks ''
        xticks ''
        vline([0 2 6],'r')
        hline(0)
        hline([0.5, -0.5])
        axis tight

        % channel significance: via a simple one-sample t-test
        %pval=[];
        %for j=1:size(chdata,1)
        %    [h p tb st] = ttest(chdata(j,:));            
        %    pval(j) = p;
        %end
        %[pfdr, pval1]=fdr(pval,0.05);pfdr


        
        % channel significance: if mean is outside the 95% boostrapped C.I. for
        % any duration of time
%         tmp = mb(:,1:1000);
%         tmp = tmp(:);
%         pval=[];sign_time=[];
%         for j=3001:6000
%             if m(j)>0
%                 ptest = (sum(m(j) >= tmp(:)))/length(tmp);
%                 sign_time=[sign_time 1];
%             else
%                 ptest = (sum(m(j) <= tmp(:)))/length(tmp);
%                 sign_time=[sign_time -1];
%             end
%             ptest = 1-ptest;
%             pval = [pval ptest];   
% 
%         end
%         [pfdr, pval1]=fdr(pval,0.05);pfdr;
%         %pfdr=0.0005;
%         pval(pval<=pfdr) = 1;
%         pval(pval~=1)=0;
%         m1=m(3001:6000);
%         tt1=tt(3001:6000);
%         idx1=find(pval==1);
        %plot(tt1(idx1),m1(idx1),'b')

        %sum(pval/3000)

        
% 
        %if sum(pval>0)
%         if ~isempty(pval) && sum(t(3000:7000)) >0 
%             %if sum(pval.*sign_time)>0
%                 box_col = 'g';
%                 sig_ch(i,ch)=1;
%             %else
%             %    box_col = 'b';
%             %    sig_ch(i,ch)=-1;
%             %end
%             box on
%             set(gca,'LineWidth',2)
%             set(gca,'XColor',box_col)
%             set(gca,'YColor',box_col)
%         end        
    end    
    sgtitle(ImaginedMvmt(idx(i)))
    %filename = fullfile('F:\DATA\ecog data\ECoG BCI\Results\ERPs Imagined Actions\delta',ImaginedMvmt{idx(i)});
    %saveas(gcf,filename)
    %set(gcf,'PaperPositionMode','auto')
    %print(gcf,filename,'-dpng','-r500')
end
%save ERPs_sig_ch_beta -v7.3
save ERPs_sig_ch_beta -v7.3
%save ERPs_sig_ch_hg -v7.3

% temp scatter
tmp=chdata(3600,:);
idx=ones(size(tmp))+0.1*randn(size(tmp));
figure;plot(idx,tmp,'.','MarkerSize',18,'Color',[.5 .5 .5 .5])
xlim([0.5 1.5])
hold on
plot(mean(tmp),'.r','MarkerSize',40)
hline(0,'k')
set(gcf,'Color','w')
ylabel('Z-score')
set(gca,'FontSize',14)
xlabel ''
xticks ''

%% no plotting just compute channels with tests for significance
% SINGLE CHANNEL TEMPORAL CLUSTER CORRECTION OF THE ERP
idx = [1:30];
%idx =  [1,10,30,25,20,28];
chMap=TrialData.Params.ChMap;
%ch=3;
sig_ch=zeros(30,128);
for i=1:length(idx)        
    data = ERP_Data{idx(i)};
    t_scores=[];
    for ch=1:numel(chMap)
        disp(['movement ' num2str(i) ' channel ' num2str(ch)])

        chdata = squeeze((data(:,ch,:)));

        % bad trial removal
        tmp_bad=zscore(chdata')';
        artifact_check=((tmp_bad)>=4) + (tmp_bad<=-4);
        good_idx = find(sum(artifact_check)==0);
        chdata = chdata(:,good_idx);

        % significance testing via temporal cluster correction
        a=chdata';
        [t,pval]=temporal_clust_1D(a,0.05);

        if ~isempty(pval) && sum(t(3000:7000))>0
            sig_ch(i,ch)=1;
        end
    end
    figure;
    tmp=sig_ch(i,:);
    imagesc(tmp(chMap))
    title(ImaginedMvmt{i})
end
% 
% tmp=sig_ch(i,:);
% figure;imagesc(tmp(chMap))
% title(ImaginedMvmt{i})


%% Clustering tests for significance B1 (2D, TFCE etc). 

% B1 grid layout
tic
chMap=TrialData.Params.ChMap;
grid_layout = chMap;
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



idx = [1:30];
%idx =  [1,10,30,25,20,28];
chMap=TrialData.Params.ChMap;
%ch=3;
sig_ch_all=zeros(30,128);
loop_iter=750;
tfce_flag=false;
for i=1:length(idx)    
    data = ERP_Data{idx(i)};
    t_scores=[];tboot=zeros(128,8001,loop_iter);
    p_scores=[];pboot=zeros(128,8001,loop_iter);
    parfor ch=1:numel(chMap)
        disp(['movement ' num2str(i) ', Channel ' num2str(ch)])
        chdata = squeeze((data(:,ch,:)));

        % bad trial removal
        tmp_bad=zscore(chdata')';
        %artifact_check = (abs(tmp_bad))<=3.0;
        artifact_check= (tmp_bad(2000:7000,:)>=3.5) + ...
            (tmp_bad(2000:7000,:)<=-3.5);
        %good_idx = find(sum(artifact_check)==0);
        good_idx = find( (sum(artifact_check)/size(artifact_check,1)) <=0.01);
        chdata = chdata(:,good_idx);

        % artfiact correction at each time point
%         t=[];p=[];artifact_check=logical(artifact_check);
%         for ii=1:size(chdata,1)
%             idx= artifact_check(ii,:);
%             [hh,pp,ci,stats]=ttest(chdata(ii,idx));
%             t(ii)=stats.tstat;
%             p(ii)=pp;
%         end
    
        
        % get the true t-statistic at each time-point
        [h,p,ci,stats] = ttest(chdata');
        t = stats.tstat;
        if tfce_flag
            t(p>0.05)=0; % only if TFCE
        end
        t_scores(ch,:) = t;
        p_scores(ch,:) = p;
    

        % get the null t-statistics at each time-point thru bootstrap
        a = chdata';
        anew=a-mean(a); % set the mean to zero
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
        tt=linspace(-3,4,size(t_scores,2));
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
        colorbar
    end
end
% 
% figure;
% for ii=3e3:size(aa,2)
%     a=aa(:,ii);
%     imagesc(a(chMap))
%     title(num2str(ii))
%     caxis([-5 5])
%     colorbar
%     pause(0.005)
% end



toc

save hg_LFO_Imagined_SpatTemp_New sig_ch_all -v7.3


% 
% tmp=sig_ch(i,:);
% figure;imagesc(tmp(chMap))
% title(ImaginedMvmt{i})

%% (MAIN) Clustering tests for significance B1 (2D, TFCE etc) WITH ARTIFACT CORRECTION. 


% B1 grid layout
tic
chMap=TrialData.Params.ChMap;
grid_layout = chMap;
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

% 
% % downsample the data to 50Hz for the specific movements 
% idx = [1:30];
% for i=1:length(idx)
%     data = ERP_Data{idx(i)};    
%     % downsample 
%     tmp_data=[];
%     for j=1:size(data,3)
%         tmp =  squeeze(data(:,:,j));
%         tmp = resample(tmp,50,1e3);
%         tmp_data(:,:,j) = tmp;
%     end
%     ERP_Data{idx(i)} = tmp_data;
% end



idx = [1:30];
%idx =  [1,10,30,25,20,28];
chMap=TrialData.Params.ChMap;
%ch=3;
sig_ch_all=zeros(30,128);
loop_iter=750;
tfce_flag=false;
for i=1:length(idx)    
    data = ERP_Data{idx(i)}; % time X channels X trials
    t_scores=[];tboot=zeros(128,size(data,1),loop_iter);%channels X time X loops
    p_scores=[];pboot=zeros(128,size(data,1),loop_iter);
    parfor ch=1:numel(chMap)
        disp(['movement ' num2str(i) ', Channel ' num2str(ch)])
        chdata = squeeze((data(:,ch,:)));

        % bad trial removal
        tmp_bad=zscore(chdata')';
        artifact_check = logical(abs(tmp_bad)>3.0);
        chdata(artifact_check)=NaN;    

        % smooth the data
        %for j=1:size(chdata,2)
        %    chdata(:,j) = smooth(chdata(:,j),10);
        %end
        

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
        tt=linspace(-3,5,size(t_scores,2));
        imagesc(tt,1:128,t_scores);
        title('Uncorrected for multiple comparisons')
        subplot(3,1,2)
        imagesc(tt,1:128,tfce_score1)
        ylabel('Channels')
        xlabel('Time')
        title('Spatiotemp. multiple comparison corrected')

        % plot the significant channels
        a=tfce_score1;
        aa=sum(a(:,150:350),2);
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
            (tboot.^2),pboot,LIMO,2,0.05,1);
        figure;subplot(3,1,1)
        tt=linspace(-3,5,size(t_scores,2));
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
        colorbar
    end
end

imaging_B1;close all
plot_elec_wts(sig_ch*6,cortex,elecmatrix,chMap)
   
figure;
imagesc(tt,1:128,abs(t_scores1));
set(gcf,'Color','w')
xlim([-3 3])

% plotting sig channels with size determined by t-scores
aa=  sum(abs(t_scores1(:,3000:5000)),2);
ii = find(aa>0);
aa(ii)=log10(aa(ii));
plot_elec_wts(2*(aa),cortex,elecmatrix,chMap)

% 
% figure;
% for ii=3e3:size(aa,2)
%     a=aa(:,ii);
%     imagesc(a(chMap))
%     title(num2str(ii))
%     caxis([-5 5])
%     colorbar
%     pause(0.005)
% end



toc

%save hg_Imagined_SpatTemp_New_New_ArtfCorr_DownSampled sig_ch_all -v7.3

% PLOTTING A FEW EXEMPLAR ERPS with CI
%load high_res_erp_hgLFO_imagined_data
ch = [97 106 25 100 103 31];
opt=statset('UseParallel',false);
for i=1:length(ch)
    figure;hold on
    chdata = squeeze((data(:,ch(i),:)));

    % bad trial removal
    tmp_bad=zscore(chdata')';
    artifact_check = logical(abs(tmp_bad)>3.0);
    chdata(artifact_check)=NaN;

    % plot the mean, confidence interval and the time-periods when
    % significant

    m=nanmean(chdata,2);    
    mb = sort(bootstrp(1000,@nanmean,chdata','Options',opt));    
    tt=linspace(-3,5,size(chdata,1));    
    [fillhandle,msg]=jbfill(tt,(mb(25,:)),(mb(975,:))...
        ,[0.2 0.2 0.8],[0.2 0.2 0.8],1,.25);
    hold on
    plot(tt,(m),'b','LineWidth',1)
    axis tight
    vline([-2 0],'k')
    hline(0,'k')
    xlim([-2.5 3])

    ts = t_scores1(ch(i),:);
    ts(abs(ts)>0)=1;
    idx = [0 diff(ts)];
    idx = find(idx~=0);   

    

    for j=1:2:length(idx)
        h1=hline(-0.5,'-g');
        set(h1,'LineWidth',3)
        set(h1,'Color',[0 .5 0 1])
        set(h1,'XData',[tt(idx(j)) tt(idx(j+1))])
    end


    title(num2str(ch(i)))
    set(gcf,'Color','w')
end


% plot the cluster as a map
figure;
for i=3460:size(t_scores1,2)
    tmp = t_scores1(:,i);
    imagesc(tmp(chMap));
    title(num2str((i)))
    caxis([-4 4])
    pause(0.05)
end



%% LOOKING AT OVERALL SPATIAL MAPS PER EFFECTOR

clc;clear
close all

addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')


load hg_LFO_Imagined_SpatTemp_New_New_ArtfCorr

ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Rotate Left Wrist','Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};

load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20230421\HandOnline\144820\BCI_Fixed\Data0001.mat')

imaging_B1;close all
chMap =  TrialData.Params.ChMap;

% rt hand, both as image and as electrode size
rt_hand = sig_ch_all([1:9 ],:);
rt_hand = mean(rt_hand,1);
rt_hand = rt_hand*10;
figure;imagesc(rt_hand(chMap));
colormap parula
plot_elec_wts(rt_hand,cortex,elecmatrix,chMap)
title('Right Hand')

% lt hand
lt_hand = sig_ch_all(10:14,:);
lt_hand = mean(lt_hand,1);
lt_hand = (lt_hand)*10;
figure;imagesc(lt_hand(chMap));
colormap parula
plot_elec_wts(lt_hand,cortex,elecmatrix,chMap)
title('Left Hand')

% rt proximal
rt_proximal = sig_ch_all([21 23 25],:);
rt_proximal = mean(rt_proximal,1);
rt_proximal = (rt_proximal)*10;
figure;imagesc(rt_proximal(chMap));
colormap parula
caxis([0 1])
plot_elec_wts(rt_proximal,cortex,elecmatrix,chMap)
title('Rt Prox')

% lt proximal
%lt_proximal = sig_ch_all([22 22 26],:);
lt_proximal = sig_ch_all([21:26],:);
lt_proximal = mean(lt_proximal,1);
lt_proximal = (lt_proximal)*10;
figure;imagesc(lt_proximal(chMap));
colormap parula
caxis([0 1])
plot_elec_wts(lt_proximal,cortex,elecmatrix,chMap)
title('Left Prox')

% distal
distal = sig_ch_all([27 28],:);
distal = mean(distal,1);
distal = (distal)*10;
figure;imagesc(distal(chMap));
colormap parula
plot_elec_wts(distal,cortex,elecmatrix,chMap)
title('Distal')

% Face (heads/lips/tongue)
face = sig_ch_all([20 29 30 ],:);
face = mean(face,1);
face = (face)*10;
figure;imagesc(face(chMap)/10);
colormap parula
caxis([0 1])
plot_elec_wts(face,cortex,elecmatrix,chMap)
title('Face')

% plot as filled circles

clear wts;
wts(1,:) = rt_hand;
wts(2,:) = lt_hand;
wts(3,:) = lt_proximal;
wts(4,:) = distal;
wts(5,:) = face;
wts_alpha(1,:)=rt_hand./10;
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
        alpha(alp(ch));
        
        axis off
    end
end

%% JACCARD DISTANCE ON SPATIAL MAPS

clc;clear
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')
%addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4\limo_cluster_functions')
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210201\Robot3DArrow\140901\BCI_Fixed\Data0004.mat')
chMap = TrialData.Params.ChMap;


load hg_LFO_Imagined_SpatTemp_New_New_ArtfCorr

ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Rotate Left Wrist','Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};

D=zeros(length(ImaginedMvmt));
k=(1/4)*ones(2);
for i=1:length(ImaginedMvmt)
    a=sig_ch_all(i,:);
    %a=conv2(a(chMap),k,'same');
    a=a(:);
    for j=i+1:length(ImaginedMvmt)
        b=sig_ch_all(j,:);
        %b=conv2(b(chMap),k,'same');
        b=b(:);
        x=[a';b'];
        x= pdist(x,'jaccard');
        %x=sqrt(sum(sum(abs(a-b).^1)));
        if isnan(x)
            x=0.005;
        end
        D(i,j) = x;
        D(j,i) = x;
    end
end
figure;imagesc(D)
xticks(1:length(ImaginedMvmt))
yticks(1:length(ImaginedMvmt))
xticklabels(ImaginedMvmt)
yticklabels(ImaginedMvmt)

Z = linkage(D,'ward');
figure;dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')

% distance between within effector to between effector movements
within_effector_hands = [squareform(D(1:9,1:9))'];
between_effector1 = D(1:9,10:18); % to left hand 
between_effector1=(between_effector1(:));
between_effector2 = D(1:9,20:end); % to all other movements 
between_effector2=(between_effector2(:));

% bootstrap difference of means
[p ]=bootstrap_ttest(within_effector_hands,between_effector1,2,1e3)
[p ]=bootstrap_diff_mean(within_effector_hands,between_effector1,1e3)
[p ]=bootstrap_diff_mean(between_effector1,between_effector2,1e3)

if length(between_effector2) > length(within_effector_hands)
    within_effector_hands(end+1:length(between_effector2)) = NaN;
    between_effector1(end+1:length(between_effector2)) = NaN;
else
    between_effector(end+1:length(within_effector_hands)) = NaN;
end

tmp = [within_effector_hands between_effector1 between_effector2];
p = ranksum(within_effector_hands,between_effector1)
p = ranksum(within_effector_hands,between_effector2)
p = ranksum(between_effector1,between_effector2)
figure;boxplot(tmp,'notch','on')

% plot mean with 95% confidence intervals 
y = nanmean(tmp);
yboot = sort(bootstrp(1000,@nanmean,tmp));
neg = yboot(500,:)-yboot(25,:);
pos =  yboot(975,:)-yboot(500,:);
x=1:3;
figure;hold on
errorbar(x,y,neg,pos,'LineStyle','none','LineWidth',1,'Color','k');
plot(x,y,'o','MarkerSize',15,'Color','k','LineWidth',1)
xlim([0.5 3.5])
ylim([0.5 0.85])
xticks(1:3)
xticklabels({'Within Rt. Hand Mvmts','b/w Rt. Hand and Lt. Hand','b/w Rt. Hand and all other'})
set(gcf,'Color','w')
ylabel('Jaccard spatial similarity coefficient')


%% PERFORMING CLASSIFICATION TRIAL LEVEL OR LOW LEVEL REPRESENTATION 
% USING ALL FEATURES FROM THE PROCESSED DATA


clc;clear
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

% load all the hG features first
load high_res_erp_hgLFO_imagined_data

% load the features, trial level 
chMap=TrialData.Params.ChMap;
hg_features={};
for i=1:length(ImaginedMvmt)
    disp(['Getting features for Mvmt ' num2str(i)]);

    data = ERP_Data{(i)};  
    ch_feat=[];
    for ch=1:numel(chMap)        
        chdata = squeeze((data(:,ch,:)));

         % bad trial removal
        tmp_bad=zscore(chdata')';
        artifact_check = logical(abs(tmp_bad)>3.0);
        chdata(artifact_check)=NaN;     

        % get the neural features, between 3500 and 4500 samples
        xx = nanmean(chdata(3500:4000,:),1);
        xx(isnan(xx)) = nanmedian(xx);
        ch_feat(ch,:) = xx;
    end
    hg_features{i}=ch_feat;
end

% Mahalanobis distance matrix
D=zeros(length(hg_features));
for i=1:length(hg_features)
    A=hg_features{i};
    % get rid of bad trials
     a=zscore(median(A,1));
     good_idx= find(abs(a)<=3);
     A = A(:,good_idx);

    for j=i+1:length(hg_features)
        B=hg_features{j};

        % get rid of bad trials
        b=zscore(median(B,1));
        good_idx= find(abs(a)<=3);
        B = B(:,good_idx);

        % get the mahal distance
        D(i,j) = mahal2(A',B',2);
        D(j,i) = D(i,j);
    end
end
figure;imagesc(D)
Z = linkage(squareform(D),'complete');
figure;
dendrogram(Z)

% 
% tmp=sig_ch(i,:);
% figure;imagesc(tmp(chMap))
% title(ImaginedMvmt{i})

%% SPATIO TEMP TEST SAME AS AOBOVE BUT TEST OF MEDIAN BEING ZERO 


% B1 grid layout
tic
chMap=TrialData.Params.ChMap;
grid_layout = chMap;
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



idx = [1:30];
%idx =  [1,10,30,25,20,28];
chMap=TrialData.Params.ChMap;
%ch=3;
sig_ch_all=zeros(30,128);
loop_iter=750;
tfce_flag=false;
for i=1:length(idx)    
    data = ERP_Data{idx(i)};
    t_scores=[];tboot=zeros(128,8001,loop_iter);
    p_scores=[];pboot=zeros(128,8001,loop_iter);
    parfor ch=1:numel(chMap)
        disp(['movement ' num2str(i) ', Channel ' num2str(ch)])
        chdata = squeeze((data(:,ch,:)));

       [t,p] = signrank_2D(chdata);
        
        if tfce_flag
            t(p>0.05)=0; % only if TFCE
        end

        t_scores(ch,:) = t;
        p_scores(ch,:) = p;

        % get the null t-statistics at each time-point thru bootstrap
        a = chdata';
        anew=a-median(a); % set the median to zero
        asize=size(anew);
        for loop=1:loop_iter
            a1= anew(randi(asize(1),[asize(1) 1]),:); % sample with replacement
            [t0,p0] = signrank_2D(a1');            
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
            limo_clustering(abs(t_scores.^2),p_scores,...
            abs(tboot.^2),pboot,LIMO,2,0.05,0);
        figure;subplot(3,1,1)
        tt=linspace(-3,4,size(t_scores,2));
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
        colorbar
    end
end
% 
% figure;
% for ii=3e3:size(aa,2)
%     a=aa(:,ii);
%     imagesc(a(chMap))
%     title(num2str(ii))
%     caxis([-5 5])
%     colorbar
%     pause(0.005)
% end



%% (MAIN) PLOTTING DIFFERENCES BETWEEN MOVEMENTS AT THE SINGLE CHANNEL LEVEL
% with temporal clustering of ANOVA F-values 


clc;clear
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')


addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')


cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210201\Robot3DArrow\140901\BCI_Fixed\Data0004.mat')
chMap= TrialData.Params.ChMap;

% neighbors
grid_layout = chMap;
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


% load the hG LFO ERP Data
load high_res_erp_hgLFO_imagined_data



data_overall=[];
idx = [19 2 3]; % both middle, rt index, rt middle

% downsample the data to 50Hz for the specific movements 
for i=1:length(idx)
    data = ERP_Data{idx(i)};    
    % downsample 
    tmp_data=[];
    for j=1:size(data,3)
        tmp =  squeeze(data(:,:,j));
        tmp = resample(tmp,50,1e3);
        tmp_data(:,:,j) = tmp;
    end
    ERP_Data{idx(i)} = tmp_data;
end

ch=106; % face is 101, hand is 103 for thumb, index and middle
figure;hold on
set(gcf,'Color','w')
%cmap={'r','g','b','y','m'};
cmap={'m','b','k'};
opt=statset('UseParallel',true);
for i=1:length(idx)
    data = ERP_Data{idx(i)};    
    data=squeeze(data(:,ch,:));

    % bad trial removal
    %tmp_bad=zscore(data')';
    %artifact_check = logical(abs(tmp_bad)>3.0);
    %data(artifact_check)=NaN;
    %[h,p,ci,stats] = ttest(chdata');

    data=data';
    %smooth data
    parfor j=1:size(data,1)
        data(j,:) = smooth(data(j,:),10);
    end

    data_overall(i,:,:)=data';

    m=nanmean(data,1);
    mb = sort(bootstrp(1000,@nanmean,data,'Options',opt));
    tt=linspace(-3,5,size(data,2));
    [fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
        ,cmap{i},cmap{i},1,.2);
    hold on
    plot(tt,m,'Color',cmap{i},'LineWidth',1)
end
h=vline([0]);
h.LineWidth=1.5;
h.Color='r';
h=vline([5]);
h.LineWidth=1.5;
h.Color='g';
hline(0,'k')
xlim([-1 3])

% run the ANOVA test at each time-point and do temporal clustering
%dependent variable -> erp
%independent variable -> movement type
%fitrm

Fvalues=[];pval=[];
parfor i=1:size(data_overall,2)
    disp(i)
    a = (squeeze(data_overall(1:3,i,:)))';
    %tmp_bad = zscore(a);
    %artifact_check = logical(abs(tmp_bad)>3);
    %a(artifact_check)=NaN;   
    a=a(:);
    erps=a;
    mvmt = num2str([ones(size(a,1)/3,1);2*ones(size(a,1)/3,1);3*ones(size(a,1)/3,1)]);
    subj = table([1],'VariableNames',{'Subject'});
    data = table(mvmt,erps);
    rm=fitrm(data,'erps~mvmt');
    ranovatbl = anova(rm);
    Fvalues(i) = ranovatbl{2,6};
    pval(i) = ranovatbl{2,7};
end

% get the boot values
Fvalues_boot=[];p_boot=[];
parfor i=1:size(data_overall,2)
    disp(i)
    a = (squeeze(data_overall(1:3,i,:)))';
    %tmp_bad = zscore(a);
    %artifact_check = logical(abs(tmp_bad)>3);
    %a(artifact_check)=NaN;

    % center the data
    a = a-nanmean(a);

    % now sample with replacement for each column to create new data
    % matrix over 1000 iterations
    Fvalues_tmp=[];ptmp=[];
    for iter=1:750      
        a_tmp=[];
        for j=1:size(a,2)
            a1 = randi([1,size(a,1)],size(a,1),1);
            a_tmp(:,j) = a(a1,j);
        end
        a_tmp=a_tmp(:);
        erps=a_tmp;
        mvmt = num2str([ones(size(erps,1)/3,1);...
            2*ones(size(erps,1)/3,1);3*ones(size(erps,1)/3,1)]);
        data = table(mvmt,erps);
        rm=fitrm(data,'erps~mvmt');
        ranovatbl = anova(rm);
        Fvalues_tmp(iter) = ranovatbl{2,6};
        ptmp(iter) =  ranovatbl{2,7};
    end
    Fvalues_boot(i,:) = Fvalues_tmp;
    p_boot(i,:) = ptmp;
end

% 
% %%% using just simple regular anova instead of repeated measures
% Fvalues=[];pval=[];
% parfor i=1:size(data_overall,2)
%     disp(i)
%     a = (squeeze(data_overall(1:3,i,:)))';
%     tmp_bad = zscore(a);
%     artifact_check = logical(abs(tmp_bad)>3);
%     a(artifact_check)=NaN;   
%     [P,ANOVATAB,STATS]  = anova1(a,[],'off');   
%     Fvalues(i) = ANOVATAB{2,5};
%     pval(i) = ANOVATAB{2,6};
% end
% %figure;plot(Fvalues)
% %figure;plot(pval);hline(0.05)
% 
% % get the boot values using simple anova1
% Fvalues_boot=[];p_boot=[];
% parfor i=1:size(data_overall,2)
%     disp(i)
%     a = (squeeze(data_overall(1:3,i,:)))';
%     tmp_bad = zscore(a);
%     artifact_check = logical(abs(tmp_bad)>3);
%     a(artifact_check)=NaN;
% 
%     % center the data
%     a = a-nanmean(a);
% 
%     % now sample with replacement for each column to create new data
%     % matrix over 1000 iterations
%     Fvalues_tmp=[];ptmp=[];
%     for iter=1:750      
%         a_tmp=[];
%         for j=1:size(a,2)
%             a1 = randi([1,size(a,1)],size(a,1),1);
%             a_tmp(:,j) = a(a1,j);
%         end
% 
%         [P,ANOVATAB,STATS]  = anova1(a_tmp,[],'off');
%         Fvalues_tmp(iter) = ANOVATAB{2,5};
%         ptmp(iter)  = ANOVATAB{2,6};
%     end
%     Fvalues_boot(i,:) = Fvalues_tmp;
%     p_boot(i,:) = ptmp;
% end


% perform limo clustering, 1 in first dimension , and iterations in last
% dimension
Fvalues_boot = reshape(Fvalues_boot,[1 size(Fvalues_boot)]);
p_boot = reshape(p_boot,[ 1 size(p_boot)]);

 LIMO.data.chanlocs=[];
        LIMO.data.neighbouring_matrix=neighb;
[mask,cluster_p,max_th] = ...
    limo_clustering(Fvalues,pval,Fvalues_boot,p_boot,...
    LIMO,3,0.05,0);
mask(mask>0)=1;
%figure;plot(pval);
%hold on; plot(mask/2)
%plot(tt,mask,'r','LineWidth',2)
ts=mask;
idx1 = [0 diff(ts)];
idx1 = find(idx1~=0);
if rem(length(idx1),2)==1
    idx1= [idx1 length(ts)];
end


for j=1:2:length(idx1)
    h1=hline(-0.5,'-g');
    set(h1,'LineWidth',3)
    set(h1,'Color',[0 .5 0 1])
    set(h1,'XData',[tt(idx1(j)) tt(idx1(j+1))])
end
xlim([-1 3])


% just plotting channel 106
imaging_B1;
sig_ch=zeros(128,1);
sig_ch(111)=8;
plot_elec_wts(sig_ch,cortex,elecmatrix,chMap)



%% MORE PLOTTING STUFF


% plotting sig channels on by one
for i=1:length(ImaginedMvmt)
    tmp = sig_ch(i,:);
    figure;imagesc(abs(tmp(chMap)));
    title(ImaginedMvmt{i})    
end

%plot brain plots of sig. ch for hg in tongue, rt bicep, rt thumb, lt
%thumb,head and beta for left leg
imaging_B1;close all
idx =  [1,10,30,25,20,27,28,21];
for i=1:length(idx)
    tmp = sig_ch(idx(i),:);
    figure;imagesc(tmp(chMap))
    sgtitle(ImaginedMvmt{idx(i)})
    plot_sig_channels_binary_B1(tmp,cortex,elecmatrix,chMap);
    sgtitle(ImaginedMvmt{idx(i)})
end

figure
c_h = ctmr_gauss_plot(cortex,elecmatrix(ch,:),...
    Wa1(1:286),'lh');
temp=Wa1;
temp=temp./(max(abs(temp)));
chI = find(temp~=0);
for j=1:length(chI)
    ms = abs(temp(chI(j))) * (12-2) + 2;
    e_h = el_add(elecmatrix(chI(j),:), 'color', 'b','msize',abs(ms));
end
set(gcf,'Color','w')
set(gca,'FontSize',20)
view(-94,30)


%%%% ROI SPECIFIC ERPS FOR SELECTION ACTIONS %%%%%%
%idx = [3,8,10,19,20,25,28,30];
%idx = [1,10,20,25,30]; % have to play around with this
%idx= [ 20 29 30]; % face
%idx = [1 3  19 ];%hand
idx= [15 16 17]% limbs
%idx=[ 10 12 16 ];% left hand
%cmap = turbo(length(idx));
cmap = brewermap(length(idx),'Set1');
s1 = [106 97 103 100 116 115];
m1 = [25 31 3 9 27];
rol = [42 15 59 36 32 63];
pmd = [79 67 34 91 90 51];
pmv= [53 2 13 ];
channels = {s1,m1,rol,pmd,pmv};
% plotting
channels=1:128;
% for beta, hand: 96, limbs: 127,79,49, face: 82
% for delta, face: 116, 104, 36, limbs: 119, 95, 35

for i=1:length(channels)    
    figure;
    hold on
    ch = channels(i);
    % first plot the ERPs
    for j=1:length(idx)
        tmp_erp = ERP_Data{idx(j)};        
        tmp = tmp_erp(:,ch,:);
        tmp = squeeze(mean(tmp,2))';        
        tmp=detrend(tmp')'; 
        tmp=tmp+0.2;
%         if j==2
%             tmp=tmp-0.2;
%         end
%         if j==2
%             tmp = tmp+0.4;
%         else
%             tmp = tmp+0.2;
%         end

%         if j==1
%             tmp = tmp-0.25;
%         end
%         if j==3
%             tmp = tmp-0.15;
%         end
        m = smooth(mean(tmp,1),300);
        %mb = sort(bootstrp(1000,@mean,tmp)); % bootstrap
        s = std(tmp,1)/sqrt(size(tmp,1)); % standard error
        s = smooth(s',300);
        tt=(0:size(tmp,2)-1) + 3000;        
        plot(tt,m,'Color',cmap(j,:),'LineWidth',2)        
        %  [fillhandle,msg]=jbfill(tt,(m-s)',(m+s)'...
        %      ,cmap(j,:),cmap(j,:),1,.2);
        hold on        
    end        
    set(gcf,'Color','w')
    % now plot the C.I.
    for j=1:length(idx)
        tmp_erp = ERP_Data{idx(j)};        
        tmp = tmp_erp(:,ch,:);
        tmp = squeeze(mean(tmp,2))';        
        tmp=detrend(tmp')';
        tmp=tmp+0.2;
%         if j==2
%             tmp=tmp-0.2;
%         end
%         if j==2
%             tmp = tmp+0.4;
%         else
%             tmp = tmp+0.2;
%         end
%         if j==1
%             tmp = tmp-0.25;
%         end
%         if j==3
%             tmp = tmp-0.15;
%         end
        m = smooth(mean(tmp,1),300);
        %mb = sort(bootstrp(1000,@mean,tmp)); % bootstrap
        s = std(tmp,1)/sqrt(size(tmp,1)); % standard error
        s = smooth(s',300);
        tt=(0:size(tmp,2)-1) + 3000;        
        %plot(tt,m,'Color',cmap(j,:),'LineWidth',2)        
          [fillhandle,msg]=jbfill(tt,(m-s)',(m+s)'...
              ,cmap(j,:),cmap(j,:),1,.1);
        hold on        
    end 
    axis tight
    legend(ImaginedMvmt(idx))       
    hline(0,'k')           
    xlim([2000 7200]+3000)
    title(num2str(i))
    xticks(4000:2000:11000)
    yticks(-.6:.4:3.4)
    ylim([-.4 1.1])
    vline([1000 3000 7000]+3000,'r')
    %waitforbuttonpress
    %close
end

%good channels for face- 40, 121, 19,126,109,30
% good channes for hand - 116, 115, 97, 51, 46, 31, 26, 25
% good channles for limbs - 126, 124, 116, 103, 29 (-0.2 mag)

% plot chMAP of the specific channels
imaging_B1;
grid_ecog = [];
for i=1:16:128
    grid_ecog =[grid_ecog; i:i+15];
end
grid_ecog=flipud(grid_ecog);
figure;
ch=10;
[x y] = find(chMap==ch);
ch1 = grid_ecog(x,y);
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix(1:end,:),'color','w','msize',2);
e_h = el_add(elecmatrix(ch1,:),'color','r','msize',10);
set(gcf,'Color','w')
view(-99,21)



b= abs(sin(2*pi*1/100*(1:100)))+0.1*randn(1,100);
b= (conv(b,b,'same')) + randn(1,100);
figure;plot(b)
[r,lags]=xcorr(b,'unbiased');
figure;stem(lags,(r))


%%%%% ROI specific activity in the various regions
chmap=TrialData.Params.ChMap;
hand_elec=[97 106 25 3 ];
%hand_elec =[22	5	20 111	122	117 105	120	107]; %pmv
%hand_elec=[99	101	121	127	105	120];%pmd
% hand_elec = [49	64	58	59
% 54	39	47	42
% 18	1	8	15];% pmv
% get the average activity for each imagined action with C.I.
roi_mean=[];
roi_dist_mean=[];
task_state = TrialData.TaskState;
%idx = [ find(task_state==3)];
idx=3001:4000;
pval=[];pval1=[];pval2=[];
%idx=idx(5:15);
for i=1:length(ImaginedMvmt)
    disp(i)
    data = ERP_Data{i};

    % boot
    tmp = data(1:1000,hand_elec,:);
    tmp = squeeze(mean(tmp,1));

    % real
    data = data(idx,hand_elec,:);
    data = squeeze(mean(data,1)); % time
    %data = squeeze(mean(data,1)); % channels
    data = data(:);
   
    roi_mean(i) = mean(data);
%     if i==19
%         roi_mean(i)=0.9;
%     end
    roi_dist_mean(:,i) = sort(bootstrp(1000,@mean,data));

    % do some stats
    pval(i) = 1-((sum((mean(data)) >= tmp(:)))/length(tmp(:)));
    if mean(roi_dist_mean(:,i)) >0
        pval1(i) = 1-(sum((roi_dist_mean(:,i))>0))/length(roi_dist_mean(:,i));
    elseif mean(roi_dist_mean(:,i)) <=0
        pval1(i) = 1-(sum((roi_dist_mean(:,i))<=0))/length(roi_dist_mean(:,i));
    end
    pval2(i) = signrank(data);
end
figure;bar(roi_mean)

y = roi_mean;
y=y';
errY(:,1) = roi_dist_mean(500,:)-roi_dist_mean(25,:);
errY(:,2) = roi_dist_mean(975,:)-roi_dist_mean(500,:);
figure;
barwitherr(errY, y);
hold on
xticks(1:30)
set(gcf,'Color','w')
set(gca,'FontSize',16)
set(gca,'LineWidth',1)
xticklabels(ImaginedMvmt)
hold on
[pfdr,pmask]=fdr(pval1,0.01);
idx = pval1<=pfdr;
idx = find(idx==1);
%vline(idx,'r');
for i=1:length(idx)
    plot(idx(i),-0.25,'*r','MarkerSize',20);
end


subplot(2,1,2) % have to run the above code again to get roi specific activity
barwitherr(errY, y);
xticks(1:30)
set(gcf,'Color','w')
set(gca,'FontSize',16)
set(gca,'LineWidth',1)
xticklabels(ImaginedMvmt)


% plotting spatial map comparing right and left hand movements in sig.
% channels (1:9, 10:18)
rt_channels = sum(abs(sig_ch(1:9,:)));
lt_channels = sum(abs(sig_ch(10:18,:)));
figure;imagesc(rt_channels(chMap));caxis([0 8])
figure;imagesc(lt_channels(chMap));caxis([0 8])

figure;stem(abs(rt_channels))
hold on
stem(abs(lt_channels))

% PmD test
pmd=[65	85	83	68
37	56	48	43
53	55	52	35
2	10	21	30];pmd=pmd(:);
lt_pmd = (abs(sig_ch(10:18,pmd)));
rt_pmd = (abs(sig_ch(1:9,pmd)));
[sum(lt_pmd(:)) sum(rt_pmd(:))]


% cosine distance between cortical network
D=zeros(30);
for i=1:size(sig_ch,1)
    A=sig_ch(i,:);
    for j=i+1:size(sig_ch,1)
        B=sig_ch(j,:);
        D(i,j) = pdist([(A);(B)],'jaccard');
        D(j,i) = D(i,j);
    end
end
% plotting hierarhical similarity
Z = linkage(D,'ward');
figure;dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')

%%% building a MLP classifier based on average activity across channels
% 2.3s to 3.3s after go cue, averaged over that one second window
res_overall=[];
condn_data=[];  % features by samples
Y=[];
%using average activity plus a smoothing parameter
for i=1:length(ERP_Data)
    tmp = ERP_Data{i};
    % heavily smooth each channel's single trial activity
    for j=1:size(tmp,2)
        for k=1:size(tmp,3)
            tmp(:,j,k) = smooth(tmp(:,j,k),500);
        end
    end
    m = squeeze(mean(tmp(3000:4500,:,:),1));
    s = squeeze(std(tmp(3000:4500,:,:),1));
    condn_data=cat(2,condn_data,[m;s]);
    Y = [Y;i*ones(size(m,2)*1,1)];
end

for iter=1:20
    disp(iter)

    N=condn_data;

    % partition into training and testing
    idx = randperm(size(condn_data,2),round(0.8*size(condn_data,2)));
    YTrain = Y(idx);
    NTrain = N(:,idx);
    I = ones(size(condn_data,2),1);
    I(idx)=0;
    YTest = Y(logical(I));
    NTest = N(:,logical(I));

    T1=YTrain;
    T = zeros(size(T1,1),30);
    for i=1:30
        [aa bb]=find(T1==i);
        %T(aa(1):aa(end),i)=1;
        T(aa,i)=1;
    end

    clear net
    net = patternnet([64]) ;
    net.performParam.regularization=0.2;
    net.divideParam.trainRatio = 0.85;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.0;
    net = train(net,NTrain,T','UseGPU','yes');

    % test on held out data
    out = net(NTest);
    D=zeros(30);
    for i=1:length(YTest)
        [aa bb]=max(out(:,i));
        D(YTest(i),bb)=D(YTest(i),bb)+1;
    end
    for i=1:size(D,1)
        D(i,:)= D(i,:)/sum(D(i,:));
    end
    %figure;stem(diag(D))
    %xticks(1:30)
    %xticklabels(ImaginedMvmt)
    res_overall(iter,:)=diag(D);
    res_overall_map(iter,:,:) = D;
end
figure;stem(mean(res_overall,1))
xticks(1:30)
xticklabels(ImaginedMvmt)
hline(1/30)
figure;
imagesc(squeeze(mean(res_overall_map,1)))

% use a classifier based on 

%%%% dPCA analyses on the imagined movement data.. ROI X Time X Mvmt-Type
%avg activity: N x C x L x T
% Channels X Conditions X Laterality X Time
M1 =[27	9	3
26	31	25
116	103	106
115	100	97];M1=M1(:);
pmd=[94	91	79
73	90	67
61	51	34
40	46	41];pmd=pmd(:);
pmv=[96	84	76	95
92	65	85	83
62	37	56	48
45	53	55	52];pmv=pmv(:);
m1_ant=[19	2	10	21
24	13	6	4
124	126	128	119
102	109	99	101];m1_ant=m1_ant(:);
central=[33	49	64	58
50	54	39	47
28	18	1	8
5	20	14	11];central=central(:);

%condn_idx = [1 23  3  24 ];
condn_idx = [1 25  10  26 ];

Data_left=[];Data_right=[];
for i=1:length(condn_idx)
    tmp = ERP_Data{condn_idx(i)};
    tmp=squeeze(mean(tmp,3));
    %tmp = tmp(:,M1);
    if i>2
        Data_left=cat(3,Data_left,tmp');
    else
        Data_right=cat(3,Data_right,tmp');
    end
end
clear Data
Data(:,:,1,:) = permute(Data_right,[1 3 2]);
Data(:,:,2,:) = permute(Data_left,[1 3 2]);
firingRatesAverage=Data;

% plot all right/left finger ERPs
idx = [10:18];
ch=106;
figure;
hold on
col = parula(length(idx));
for i=1:length(idx)
    data = ERP_Data{idx(i)};
    chdata = squeeze((data(:,ch,:)));
    m=mean(chdata,2);
    tt=linspace(-1,7,size(data,1));
    plot(tt,m,'LineWidth',1,'Color',col(i,:));
end
vline([0 2 6],'r')
set(gcf,'Color','w')
set(gca,'FontSize',12)
legend(ImaginedMvmt(idx))


% sig channel from the ERP analyses
% head 
tmp = sig_ch(27:28,:);
tmp=mean(abs(tmp),1);
tmp(tmp~=0)=1;
figure;imagesc(abs(tmp(chMap)))
set(gcf,'Color','w')
axis off
set(gcf,'Color','w')
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh')
e_h = el_add(elecmatrix(find(tmp~=0),:),'color','b')
set(gcf,'Color','w')

% getting a map of significant channels based on an ANOVA model. % Rt
% single digit, Lt single digit, Head, Lips/Tongue, Rt Proximal (Bi/Tri),
% Lt Proximal (Bi/Tri), Distal (Lt leg)


% get the data
% right sindle digit
rt_idx = [10:18];rt_data=[];
for i=1:length(rt_idx)
    rt_data(i,:,:,:) = ERP_Data{rt_idx(i)};
end
rt_data = squeeze(mean(rt_data,1));
% left sindle digit
lt_idx = [10:18];lt_data=[];
for i=1:length(lt_idx)
    lt_data(i,:,:,:) = ERP_Data{lt_idx(i)};
end
lt_data = squeeze(mean(lt_data,1));
% head
head = ERP_Data{20};
% lips tong
lips_tong=[];
lips_tong = cat(4,lips_tong,ERP_Data{29},ERP_Data{30},ERP_Data{20});
lips_tong = permute(lips_tong,[4 1 2 3]);
lips_tong = squeeze(mean(lips_tong,4));
% rt prximal
rt_prox=[];
rt_prox = cat(4,rt_prox,ERP_Data{21},ERP_Data{22},...
    ERP_Data{23},ERP_Data{24},ERP_Data{25},ERP_Data{26});
rt_prox = permute(rt_prox,[4 1 2 3]);
rt_prox = squeeze(mean(rt_prox,4));
% lt proximal
lt_prox=[];
lt_prox = cat(4,lt_prox,ERP_Data{24},ERP_Data{24});
lt_prox = squeeze(mean(lt_prox,4));
% legs
legs=[];
legs = cat(4,legs,ERP_Data{27},ERP_Data{28});
legs = permute(legs,[4 1 2 3]);
legs = squeeze(mean(legs,4));

% run the test at each channel, avg activity 3200-5000 time-pts
anova_sig_ch_pval=[];
time_idx=3200:5000;
for i=1:128
    anova_data = [squeeze(mean(rt_data(time_idx,i,:)))...
        squeeze(mean(lt_data(time_idx,i,:)))...
        squeeze(mean(head(time_idx,i,:)))...
        squeeze(mean(lips_tong(time_idx,i,:)))...
        squeeze(mean(rt_prox(time_idx,i,:)))...
        squeeze(mean(lt_prox(time_idx,i,:)))...
        squeeze(mean(legs(time_idx,i,:)))];
    [p,tbl,stats]=anova1(anova_data,[],'off');
    anova_sig_ch_pval(i)=p;
end
sum(anova_sig_ch_pval<=0.05)

% anova on movements within a body part set e.g., all rt hand movements on
% a sample by sample basis
anova_sig_ch_pval=[];
time_idx=4000:5500;
for i=1:128
    disp(i)
    parfor time_idx=3000:5000
        anova_data = squeeze(mean(rt_data(:,time_idx,i,:),2))';
        tmp=zscore(anova_data);
        bad_idx = abs(tmp)>2.0;
        anova_data(logical(bad_idx))=NaN;
        [p,tbl,stats]=anova1(anova_data,[],'off');
        anova_sig_ch_pval(i,time_idx)=p;
    end
end
anova_sig_ch_pval=anova_sig_ch_pval(:,3000:end);
[pfdr,pvals]=fdr(anova_sig_ch_pval(:),0.05);
sum(anova_sig_ch_pval(:)<=0.05)
%sum(anova_sig_ch_pval(:)<=pfdr)
%tmp = anova_sig_ch_pval<=pfdr;
tmp = anova_sig_ch_pval<=1e-4;
figure;imagesc(tmp)
tmp1=sum(tmp');
tmp1(tmp1>0)=1;
figure;stem(tmp1)
figure;imagesc(tmp1(chMap))

% anova on movements within a body part set e.g., all rt hand movements on
% averaged over time
anova_sig_ch_pval=[];
time_idx=3200:5000;
for i=1:128    
    anova_data = squeeze(mean(rt_data(:,time_idx,i,:),2))';
    [p,tbl,stats]=anova1(anova_data,[],'off');
    anova_sig_ch_pval(i)=p;
end
[pfdr,pvals]=fdr(anova_sig_ch_pval(:),0.05);
sum(anova_sig_ch_pval(:)<=0.05)
tmp = anova_sig_ch_pval<=0.05;
figure;imagesc(tmp(chMap))
axis off
set(gcf,'Color','w')

% same as above but with bad trial removal
anova_sig_ch_pval=[];
time_idx=3200:5000;
for i=1:128    
    anova_data = squeeze(mean(rt_data(:,time_idx,i,:),2))';
    tmp=zscore(anova_data);
    bad_idx = abs(tmp)>2.0;
    anova_data(logical(bad_idx))=NaN;
    [p,tbl,stats]=anova1(anova_data,[],'off');
    anova_sig_ch_pval(i)=p;
end
[pfdr,pvals]=fdr(anova_sig_ch_pval(:),0.01);
sum(anova_sig_ch_pval(:)<=0.01)
tmp = anova_sig_ch_pval<=0.05;
figure;imagesc(tmp(chMap))
axis off
set(gcf,'Color','w')
   


% delta -> covers rt shoulder, lt shoulder and leg but also in same hand
% knob regions
% now look at Mahab distance using this new data from 3000-6000 samples
% mean for each trial, std across trials
D=[];
for i=1:length(ERP_Data)
    A=ERP_Data{i};
    disp(i)
    for j=i:length(ERP_Data)
        B=ERP_Data{j};
        if i==j
            D(i,j)=0;
        else
            a=squeeze(mean(A(3000:6000,:,:),1))';
            % artifact correction
            m = zscore(mean(a,2));
            idx = find(abs(m)>3);
            I=ones(size(a,1),1);
            I(idx)=0;
            a=a(logical(I),:);

            b=squeeze(mean(B(3000:6000,:,:),1))';
            % artifact correction
            m = zscore(mean(b,2));
            idx = find(abs(m)>3);
            I2=ones(size(b,1),1);
            I2(idx)=0;
            b=b(logical(I),:);

            a=A(3500:5000,:,logical(I));
            clear a1 b1
            for ii=1:size(a,3)
                a1(:,:,ii) = resample(a(:,:,ii),1,5);
            end
            a=a1;
            a=permute(a,[2 1 3]);
            a=a(:,:)';

            b=B(3500:5000,:,logical(I2));
            for ii=1:size(b,3)
                b1(:,:,ii) = resample(b(:,:,ii),1,5);
            end
            b=b1;
            b=permute(b,[2 1 3]);
            b=b(:,:)';

            D(i,j) = mahal2(a,b,2);
            D(j,i) = D(i,j);
        end
    end
end
figure;imagesc((D))



Z = linkage(D,'ward');
figure;dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')

% mahab distance after smoothing hG activity 

 

 % saving dpng images for all erps in various bands
 filepath='F:\DATA\ecog data\ECoG BCI\Results\ERPs Imagined Actions\beta';
 D=dir(filepath); 
 for i=3:length(D)
     disp(i-2)
     filename = fullfile(D(i).folder,D(i).name);
     openfig(filename);
     set(gcf,'WindowState','maximized')
     filename_to_save = fullfile(D(i).folder,D(i).name(1:end-4));
     set(gcf,'PaperPositionMode','auto')
     print(gcf,filename_to_save,'-dpng','-r500')
     close all
 end


 a = [0.27 0.35 0.37 0.45];
 stat = mean(a);
 a = a-mean(a)+0.25;
 boot=[];
 for i=1:10000
     idx = randi(length(a),length(a),1);
     boot(i) = mean(a(idx));
 end
figure;hist(boot,6)
vline(stat)
sum(boot>stat)/length(boot)
