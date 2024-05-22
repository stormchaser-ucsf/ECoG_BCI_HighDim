%% ERPs of imagined actions higher sampling rate (MAIN)
% using hG and LMP, beta etc.

clc;clear
if ~isunix
    addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
    addpath('C:\Users\nikic\Documents\MATLAB')
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
    addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
    addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')
    cd 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3'
else
    cd('/home/reza/Repositories/ECoG_BCI_HighDim')
    addpath(genpath('/home/reza/Repositories/ECoG_BCI_HighDim'))
    addpath('/home/reza/Repositories/limo_tools')
    addpath('/home/reza/Repositories/limo_tools/limo_cluster_functions')
end



root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
foldernames = {'20230111','20230118','20230119','20230125','20230126',...
    '20230201','20230203'};
cd(root_path)




files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'ImaginedMvmtDAQ')
    D=dir(folderpath);

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
    'Right Knee','Left Knee',...
    'Right Ankle','Left Ankle',...
    'Lips','Tongue'};



% load the ERP data for each target
ERP_Data={};
for i=1:length(ImaginedMvmt)
    ERP_Data{i}=[];
end

% TIMING INFORMATION FOR THE TRIALS
Params.InterTrialInterval = 2; % rest period between trials
Params.InstructedDelayTime = 2; % text appears telling subject which action to imagine
Params.CueTime = 2; % A red square; subject has to get ready
Params.ImaginedMvmtTime = 4; % A green square, subject has actively imagine the action

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
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.BroadbandData;
        features = cell2mat(features');
        Params = TrialData.Params;


        % artifact correction is there is too much noise in raw signal
        if sum(abs(features(:))>5)/length(features(:))*100 < 5 ...
                && size(features,2)==256




            %get hG through filter bank approach
            filtered_data=zeros(size(features,1),size(features,2),1);
            k=1;
            for ii=1%9:16 is hG, 4:5 is beta, 1 is delta
                filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
                    Params.FilterBank(ii).b, ...
                    Params.FilterBank(ii).a, ...
                    features)));
                k=k+1;
            end
            %tmp_hg = squeeze(mean(filtered_data.^2,3));
            tmp_hg = squeeze(mean(filtered_data,3));   

%             % low pass filter the data or low pass filter hG data
%             %features1 = [randn(4000,128);features;randn(4000,128)];
%             features1 = [std(tmp_hg(:))*randn(4000,256) + mean(tmp_hg);...
%                 tmp_hg;...
%                 std(tmp_hg(:))*randn(4000,256) + mean(tmp_hg)];
%             tmp_hg = ((filtfilt(lpFilt,features1)));
%             %tmp_hg = abs(hilbert(filtfilt(lpFilt,features1)));
%             tmp_hg = tmp_hg(4001:end-4000,:);
            

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
            var_exists=true;
            try
                tmp_action_name = TrialData.ImaginedAction;
            catch
                var_exists=false;
            end

            if var_exists && size(features,2)==256
                for j=1:length(ImaginedMvmt)
                    if strcmp(ImaginedMvmt{j},TrialData.ImaginedAction)
                        tmp = ERP_Data{j};
                        tmp = cat(3,tmp,features);
                        ERP_Data{j}=tmp;
                        break
                    end
                end
            end
        end
    end
end

%save high_res_erp_beta_imagined_data -v7.3
%save high_res_erp_LMP_imagined_data -v7.3
save B3_delta_high_res_erp_imagined_data -v7.3

% get the number of epochs used
ep=[];
for i=1:length(ERP_Data)
    tmp = ERP_Data{i};
    ep(i) = size(tmp,3);
end
figure;stem(ep)
xticks(1:32)
xticklabels(ImaginedMvmt)

%% SAVING THE DATA FILES
mkdir('C:\Data from F drive\B3 data\delta')
cd('C:\Data from F drive\B3 data\delta')
for i=1:length(ImaginedMvmt)
    disp(['Saving Movement ' num2str(i)])
    data=ERP_Data{i};
    data = single(data);
    filename = ([ImaginedMvmt{i} '_Delta.mat']);
    save(filename,'data','-v7.3')
end

%%  plot ERPs and see individual channels, with stats
data = ERP_Data{21};
figure;
for ch=1:size(data,2)

    clf
    set(gcf,'Color','w')
    set(gcf,'WindowState','maximized')
    title(num2str(ch))

    chdata = squeeze((data(:,ch,:)));

    % bad trial removal
    tmp_bad=zscore(chdata')';
    artifact_check=((tmp_bad)>=5) + (tmp_bad<=-4);
    good_idx = find(sum(artifact_check)==0);
    chdata = chdata(:,good_idx);


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
    %yticks ''
    %xticks ''
    vline([0 2 6],'r')
    hline(0)
    axis tight

    % channel significance: if mean is outside the 95% boostrapped C.I. for
    % any duration of time
    tmp = mb(:,1:1000);
    tmp = tmp(:);
    pval=[];sign_time=[];
    for j=3001:6000
        if m(j)>0
            ptest = (sum(m(j) >= tmp(:)))/length(tmp);
            sign_time=[sign_time 1];
        else
            ptest = (sum(m(j) <= tmp(:)))/length(tmp);
            sign_time=[sign_time -1];
        end
        ptest = 1-ptest;
        pval = [pval ptest];

    end
    [pfdr, pval1]=fdr(pval,0.05);pfdr;
    %pfdr=0.0005;
    pval(pval<=pfdr) = 1;
    pval(pval~=1)=0;
    m1=m(3001:6000);
    tt1=tt(3001:6000);
    idx1=find(pval==1);
    plot(tt1(idx1),m1(idx1),'b')

    %sum(pval/3000)


    %
    if sum(pval)>300 % how many sig. samples do you want for it to be 'sig'
        if sum(pval.*sign_time)>0
            box_col = 'g';
        else
            box_col = 'b';
        end
        box on
        set(gca,'LineWidth',2)
        set(gca,'XColor',box_col)
        set(gca,'YColor',box_col)
    end

    waitforbuttonpress



end


%% plot ERPs at all channels with tests for significance
idx = [9,10,20,21,27,29,31,32 ];
bad_ch = [108,113,118];
load('ECOG_Grid_8596_000063_B3.mat')
chMap=ecog_grid;
sig_ch=zeros(32,256);
for i=1:length(idx)
    figure
    ha=tight_subplot(size(chMap,1),size(chMap,2));
    d = 1;
    set(gcf,'Color','w')
    set(gcf,'WindowState','maximized')
    data = ERP_Data{idx(i)};
    for ch=1:size(data,2)

        if sum(ch==bad_ch) == 0

            disp(['movement ' num2str(i) ' channel ' num2str(ch)])

            [x y] = find(chMap==ch);
            if x == 1
                axes(ha(y));
                %subplot(8, 16, y)
            else
                s = 23*(x-1) + y;
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
            %hline([0.5, -0.5])
            axis tight


            % channel significance: if mean is outside the 95% boostrapped C.I. for
            % any duration of time
            tmp = mb(:,1:1000);
            tmp = tmp(:);
            pval=[];sign_time=[];
            for j=3001:6000
                if m(j)>0
                    ptest = (sum(m(j) >= tmp(:)))/length(tmp);
                    sign_time=[sign_time 1];
                else
                    ptest = (sum(m(j) <= tmp(:)))/length(tmp);
                    sign_time=[sign_time -1];
                end
                ptest = 1-ptest;
                pval = [pval ptest];

            end
            [pfdr, pval1]=fdr(pval,0.05);pfdr;
            pfdr=0.0005;
            pval(pval<=pfdr) = 1;
            pval(pval~=1)=0;
            m1=m(3001:6000);
            tt1=tt(3001:6000);
            idx1=find(pval==1);
            plot(tt1(idx1),m1(idx1),'b')

            %sum(pval/3000)


            %
            if sum(pval)>300
                if sum(pval.*sign_time)>0
                    box_col = 'g';
                    sig_ch(i,ch)=1;
                else
                    box_col = 'b';
                    sig_ch(i,ch)=-1;
                end
                box on
                set(gca,'LineWidth',2)
                set(gca,'XColor',box_col)
                set(gca,'YColor',box_col)
            end
        end
    end
    sgtitle(ImaginedMvmt(idx(i)))
    %     filename = fullfile('F:\DATA\ecog data\ECoG BCI\Results\ERPs Imagined Actions\delta',ImaginedMvmt{idx(i)});
    %     saveas(gcf,filename)
    %     set(gcf,'PaperPositionMode','auto')
    %     print(gcf,filename,'-dpng','-r500')
end
%save ERPs_sig_ch_beta -v7.3
save ERPs_sig_ch_LMP -v7.3
%save ERPs_sig_ch_hg -v7.3

%% %% Clustering tests for significance B3 (2D, TFCE etc) WITH ARTIFACT CORRECTION. 


% B3 grid layout
% filepath='/media/reza/ResearchDrive/B3 Data for ERP Analysis';
% cd(filepath)
% load('ECOG_Grid_8596_000067_B3.mat')    
% chMap=ecog_grid;
% grid_layout = chMap;

if isunix
    filepath='/media/reza/ResearchDrive/B3 Data for ERP Analysis/delta';
    cd(filepath)
    load('ECOG_Grid_8596_000067_B3.mat')
    chMap=ecog_grid;
    grid_layout = chMap;
end

% rename all the channels after accounting for 108,113,118
for i=109:112
    [x, y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-1;
end

for i=114:117
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-2;
end

for i=119:256
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-3;
end

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

tic
load('ImaginedMvmt.mat')
idx = [1:length(ImaginedMvmt)];
%idx =  [1,10,30,25,20,28];
sig_ch_all=zeros(32,253);
loop_iter=750;
tfce_flag=false;

% parallel cluster
clus = parcluster;
clus.NumWorkers = 16;
par_clus = clus.parpool(16)

for i=1:length(idx)    
    %filename = [ImaginedMvmt{i} '.mat'];
     filename = [ImaginedMvmt{i} '_Delta.mat'];
    
    load(filename)
    data = double(data);
    
    t_scores=[];tboot=(zeros(253,8001,loop_iter));
    p_scores=[];pboot=(zeros(253,8001,loop_iter));
    % remove the bad channels
    bad_ch = [108 113 118];
    good_ch_idx=ones(256,1);
    good_ch_idx(bad_ch)=0;
    good_ch_idx=logical(good_ch_idx);
    data=data(:,good_ch_idx,:);
    parfor ch=1:numel(chMap)
        disp(['movement ' num2str(i) ', Channel ' num2str(ch)])
        chdata = squeeze((data(:,ch,:)));

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
        tt=linspace(-3,5,size(t_scores,2));
        imagesc(tt,1:253,t_scores);
        title('Uncorrected for multiple comparisons')    
        ylabel('Channels')
        xlabel('Time')
        subplot(3,1,2)
        t_scores1=t_scores;
        t_scores1(mask==0)=0;
        imagesc(tt,1:253,t_scores1);
        title('Corrected for multiple comparisons')        
        ylabel('Channels')
        xlabel('Time')
        a=mask;
        aa=sum(a(:,3000:6000),2);
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

save B3_delta_Imagined_SpatTemp_New_New_ArtfCorr sig_ch_all -v7.3


%% (MAIN) PLOTTING DIFFERENCES BETWEEN MOVEMENTS AT THE SINGLE CHANNEL LEVEL
% with temporal clustering of ANOVA F-values 


clc;clear
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')


addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
load('ECOG_Grid_8596_000063_B3.mat')
chMap=ecog_grid;
grid_layout=chMap;

% rename all the channels after accounting for 108,113,118
for i=109:112
    [x, y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-1;
end

for i=114:117
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-2;
end

for i=119:256
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-3;
end

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
%load high_res_erp_hgLFO_imagined_data

load B3_delta_high_res_erp_imagined_data

ERP_Data_bkup=ERP_Data;



data_overall=[];
%idx = [19 2 3]; % both middle, rt index, rt middle for hg
idx = [1 4 9]; % both middle, rt thumb, rt middle for hg

ERP_Data=  ERP_Data_bkup;

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

ch=25; % face is 101, hand is 103 for thumb, index and middle
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
% 
% Fvalues=[];pval=[];
% parfor i=1:size(data_overall,2)
%     disp(i)
%     a = (squeeze(data_overall(1:3,i,:)))';
%     %tmp_bad = zscore(a);
%     %artifact_check = logical(abs(tmp_bad)>3);
%     %a(artifact_check)=NaN;   
%     a=a(:);
%     erps=a;
%     mvmt = num2str([ones(size(a,1)/3,1);2*ones(size(a,1)/3,1);3*ones(size(a,1)/3,1)]);
%     subj = table([1],'VariableNames',{'Subject'});
%     data = table(mvmt,erps);
%     rm=fitrm(data,'erps~mvmt');
%     ranovatbl = anova(rm);
%     Fvalues(i) = ranovatbl{2,6};
%     pval(i) = ranovatbl{2,7};
% end
% 
% % get the boot values
% Fvalues_boot=[];p_boot=[];
% parfor i=1:size(data_overall,2)
%     disp(i)
%     a = (squeeze(data_overall(1:3,i,:)))';
%     %tmp_bad = zscore(a);
%     %artifact_check = logical(abs(tmp_bad)>3);
%     %a(artifact_check)=NaN;
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
%         a_tmp=a_tmp(:);
%         erps=a_tmp;
%         mvmt = num2str([ones(size(erps,1)/3,1);...
%             2*ones(size(erps,1)/3,1);3*ones(size(erps,1)/3,1)]);
%         data = table(mvmt,erps);
%         rm=fitrm(data,'erps~mvmt');
%         ranovatbl = anova(rm);
%         Fvalues_tmp(iter) = ranovatbl{2,6};
%         ptmp(iter) =  ranovatbl{2,7};
%     end
%     Fvalues_boot(i,:) = Fvalues_tmp;
%     p_boot(i,:) = ptmp;
% end


%%% using just simple regular anova instead of repeated measures
Fvalues=[];pval=[];
parfor i=1:size(data_overall,2)
    disp(i)
    a = (squeeze(data_overall(1:end,i,:)))';
    tmp_bad = zscore(a);
    artifact_check = logical(abs(tmp_bad)>3);
    a(artifact_check)=NaN;   
    [P,ANOVATAB,STATS]  = anova1(a,[],'off');   
    Fvalues(i) = ANOVATAB{2,5};
    pval(i) = ANOVATAB{2,6};
end
%figure;plot(Fvalues)
%figure;plot(pval);hline(0.05)

% get the boot values using simple anova1
Fvalues_boot=[];p_boot=[];
parfor i=1:size(data_overall,2)
    disp(i)
    a = (squeeze(data_overall(1:end,i,:)))';
    tmp_bad = zscore(a);
    artifact_check = logical(abs(tmp_bad)>3);
    a(artifact_check)=NaN;

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

        [P,ANOVATAB,STATS]  = anova1(a_tmp,[],'off');
        Fvalues_tmp(iter) = ANOVATAB{2,5};
        ptmp(iter)  = ANOVATAB{2,6};
    end
    Fvalues_boot(i,:) = Fvalues_tmp;
    p_boot(i,:) = ptmp;
end


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


%% PLOTTING SIG CHANNELS ON  BRAIN


clc;clear
imaging_B3;close all
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')


load B3_hG_Imagined_SpatTemp_New_ArtfCorr %hG
%load B3_delta_Imagined_SpatTemp_New_ArtfCorr %delta
%load B3_beta_Imagined_SpatTemp_New_ArtfCorr %beta

load('ECOG_Grid_8596_000063_B3.mat')
chMap=ecog_grid;

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
    'Right Knee','Left Knee',...
    'Right Ankle','Left Ankle',...
    'Lips','Tongue'};

% rename all the channels after accounting for 108,113,118
grid_layout = chMap;
for i=109:112
    [x, y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-1;
end

for i=114:117
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-2;
end

for i=119:256
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-3;
end
chMap = grid_layout;

% plotting individual maps
for i=1:size(sig_ch_all,1)
    tmp=sig_ch_all(i,:);
    %figure;imagesc(tmp(chMap));
    plot_elec_wts_B3(tmp*5,cortex,elecmatrix,chMap)
    title(ImaginedMvmt{i})
    view(-100,10)
end

% rt hand, both as image and as electrode size
rt_hand = sig_ch_all(1:9,:);
rt_hand = mean(rt_hand,1);
rt_hand = rt_hand*5;
figure;imagesc(rt_hand(chMap));
%colormap parula
plot_elec_wts_B3(rt_hand,cortex,elecmatrix,chMap)
title('Right Hand')

% lt hand
lt_hand = sig_ch_all(10:18,:);
lt_hand = mean(lt_hand,1);
lt_hand = (lt_hand)*5;
figure;imagesc(lt_hand(chMap));
%colormap parula
plot_elec_wts_B3(lt_hand,cortex,elecmatrix,chMap)
title('Left Hand')

% rt proximal
rt_proximal = sig_ch_all([21 23 25],:);
rt_proximal = mean(rt_proximal,1);
rt_proximal = (rt_proximal)*5;
figure;imagesc(rt_proximal(chMap));
colormap parula
plot_elec_wts_B3(rt_proximal,cortex,elecmatrix,chMap)
title('Rt Prox')

% lt proximal
lt_proximal = sig_ch_all([22 22 26],:);
%lt_proximal = sig_ch_all([21:26],:);
lt_proximal = mean(lt_proximal,1);
lt_proximal = (lt_proximal)*5;
figure;imagesc(lt_proximal(chMap));
%colormap parula
plot_elec_wts_B3(lt_proximal,cortex,elecmatrix,chMap)
title('Left Prox')

% distal
distal = sig_ch_all([27:30],:);
distal = mean(distal,1);
distal = (distal)*5;
figure;imagesc(distal(chMap));
colormap parula
plot_elec_wts_B3(distal,cortex,elecmatrix,chMap)
title('Distal')

% Face (heads/lips/tongue)
face = sig_ch_all([20 31 32 ],:);
face = mean(face,1);
face = (face)*5;
figure;imagesc(face(chMap));
%colormap parula
plot_elec_wts_B3(face,cortex,elecmatrix,chMap)
title('Face')



clear wts;
wts(1,:) = rt_hand;
wts(2,:) = lt_hand;
wts(3,:) = lt_proximal;
wts(4,:) = distal;
wts(5,:) = face;
wts(6,:) = rt_proximal;

%wts_alpha(1,:)=rt_hand./5;
% scale between 30 and 200
wts=50 +  (150-50)*( (wts - min(wts(:)))./( max(wts(:))-min(wts(:))));
for j=1:size(wts,1)
    wts1=wts(j,:);
    figure

    ha=tight_subplot(size(chMap,1),size(chMap,2));
    d = 1;
    set(gcf,'Color','w')
    set(gcf,'WindowState','maximized')
    %alp = wts_alpha(j,:);
    for ch=1:length(wts1)

        [x y] = find(chMap==ch);
        if x == 1
            axes(ha(y));
            %subplot(8, 16, y)
        else
            s = 23*(x-1) + y;
            axes(ha(s));
            %subplot(8, 16, s)
        end
        hold on

        if wts1(ch)==50
            col = 'k';
        else
            col = 'r';
        end

        plot(0,0,'.','Color',col,'MarkerSize',wts1(ch));
        %scatter(0,0,wts1(ch),col,'filled')
        %alpha(alp(ch));
        axis off
    end
end

% doing some stats on the weights
for i=1:size(wts,1)
    wts(i,:) = wts(i,:)./norm(wts(i,:));
end

xx  = wts*wts';

xx=wts(1:2,:);
boot_val=[];
for iter=1:1000
    a=xx(:);
    idx=randperm(numel(a));
    a=a(idx);
    a=reshape(a,size(xx));
    boot_val(iter) = corr(a(1,:)',a(2,:)');
end

%% Jaccard distances



clc;clear
%imaging_B3;close all
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')


load B3_hG_Imagined_SpatTemp_New_ArtfCorr %hG
%load B3_delta_Imagined_SpatTemp_New_ArtfCorr %delta
%load B3_beta_Imagined_SpatTemp_New_ArtfCorr %beta

load('ECOG_Grid_8596_000063_B3.mat')
chMap=ecog_grid;

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
    'Right Knee','Left Knee',...
    'Right Ankle','Left Ankle',...
    'Lips','Tongue'};

% rename all the channels after accounting for 108,113,118
grid_layout = chMap;
for i=109:112
    [x, y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-1;
end

for i=114:117
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-2;
end

for i=119:256
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-3;
end
chMap = grid_layout;


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


%%%% FINER JACCARD %%%%%
% within right hand (1)
rt_idx=[1:9 ];
within_effector_hands = [squareform(D(rt_idx,rt_idx))'];

% b/w rt to lt. hand (2)
lt_idx = [10:18];
between_effector1 = D(1:9,lt_idx); 
between_effector1=(between_effector1(:));

% b/w rt to rt proximal movements (or all proximal) (3)
rt_prox_idx=[21:26];
between_effector2 = D(1:9,rt_prox_idx); 
between_effector2=(between_effector2(:));

% b/w rt to lt proximal movements (4)
lt_prox_idx=[22 24 26];
between_effector3 = D(1:9,lt_prox_idx);
between_effector3=(between_effector3(:));

% b/w rt to distal (5)
distal_idx=[27:30];
between_effector4 = D(1:9,distal_idx); 
between_effector4=(between_effector4(:));

% b/w rt to face (6)
face_idx=[20 31 32];
between_effector5 = D(1:9,face_idx);
between_effector5=(between_effector5(:));

% bootstrap difference of means
[p ]=bootstrap_ttest(within_effector_hands,between_effector5,2,1e3)

if length(between_effector1) > length(within_effector_hands)
    within_effector_hands(end+1:length(between_effector1)) = NaN;
    between_effector2(end+1:length(between_effector1)) = NaN;
    between_effector3(end+1:length(between_effector1)) = NaN;
    between_effector4(end+1:length(between_effector1)) = NaN;
    between_effector5(end+1:length(between_effector1)) = NaN;
else
    between_effector(end+1:length(within_effector_hands)) = NaN;
end

tmp = [within_effector_hands between_effector1 between_effector2...
    between_effector3 between_effector4 between_effector5];
p = ranksum(within_effector_hands,between_effector1)
p = ranksum(within_effector_hands,between_effector2)
p = ranksum(within_effector_hands,between_effector3)
p = ranksum(within_effector_hands,between_effector4)
p = ranksum(within_effector_hands,between_effector5)
figure;boxplot(tmp,'notch','off')

% plot mean with 95% confidence intervals 
y = nanmean(tmp);
yboot = sort(bootstrp(1000,@nanmean,tmp));
neg = yboot(500,:)-yboot(25,:);
pos =  yboot(975,:)-yboot(500,:);
x=1:6;
figure;hold on
errorbar(x,y,neg,pos,'LineStyle','none','LineWidth',1,'Color','k');
plot(x,y,'o','MarkerSize',15,'Color','k','LineWidth',1)
xlim([0.5 6.5])
ylim([0.3 1])
xticks(1:6)
xticklabels({'Within Rt. Hand Mvmts','b/w Rt. Hand and Lt. Hand',...
    'b/w Rt. Hand and Rt Prox','b/w Rt. Hand and Lt Prox',...
    'b/w Rt. Hand and Distal','b/w Rt. Hand and Face'})
set(gcf,'Color','w')
ylabel('Jaccard spatial similarity coefficient')

%%%% MAIN plot mean with 95% confidence intervals just for the main movements
if ~exist('tmp_bkup')
    tmp_bkup=tmp;
    tmp = tmp(:,[1 2 3 5 6]);
end
y = nanmean(tmp);
yboot = sort(bootstrp(1000,@nanmean,tmp));
neg = yboot(500,:)-yboot(25,:);
pos =  yboot(975,:)-yboot(500,:);
x=1:5;
figure;hold on
errorbar(x,y,neg,pos,'LineStyle','none','LineWidth',1,'Color','k');
plot(x,y,'o','MarkerSize',12,'Color','k','LineWidth',1)
xlim([0.5 5.5])
ylim([0.4 0.7])
xticks(1:5)
xticklabels({'Within Rt. Hand Mvmts','b/w Rt. Hand and Lt. Hand',...
    'b/w Rt. Hand and Proximal','b/w Rt. Hand and Distal',...
    'b/w Rt. Hand and Orofacial'})
set(gcf,'Color','w')
ylabel('Jaccard spatial similarity coefficient')

[p ]=bootstrap_ttest(tmp(:,1),tmp(:,2),2,2e3)
[p ]=bootstrap_ttest(tmp(:,1),tmp(:,3),2,2e3)
[p ]=bootstrap_ttest(tmp(:,1),tmp(:,4),2,2e3)
[p ]=bootstrap_ttest(tmp(:,1),tmp(:,5),2,1e4)

% rank sum tests
[p ]=ranksum(tmp(:,1),tmp(:,2))
[p ]=ranksum(tmp(:,1),tmp(:,3))
[p ]=ranksum(tmp(:,1),tmp(:,4))
[p ]=ranksum(tmp(:,1),tmp(:,5))

figure;boxplot(tmp,'notch','on')


%%%%% COARSE JACCCARD %%%%%
% distance between within effector to between effector movements
within_effector_hands = [squareform(D(1:9,1:9))'];
between_effector1 = D(1:9,10:18);
between_effector1=(between_effector1(:));
between_effector2 = D(1:9,20:end);
between_effector2=(between_effector2(:));

% bootstrap difference of means
[p ]=bootstrap_ttest(within_effector_hands,between_effector1,2,2e3)
[p ]=bootstrap_ttest(within_effector_hands,between_effector2,2,2e3)
[p ]=bootstrap_ttest(between_effector1,between_effector2,2,2e3)

if length(between_effector2) > length(within_effector_hands)
    within_effector_hands(end+1:length(between_effector2)) = NaN;
    between_effector1(end+1:length(between_effector2)) = NaN;
else
    between_effector(end+1:length(within_effector_hands)) = NaN;
end

tmp = [within_effector_hands between_effector1 between_effector2];
[p,h,stats] = ranksum(within_effector_hands,between_effector1)
[p,h,stats] = ranksum(within_effector_hands,between_effector2)
[p,h,stats] = ranksum(between_effector1,between_effector2)
figure;boxplot(tmp,'notch','on')

% plot mean with 95% confidence intervals 
y = nanmedian(tmp);
yboot = sort(bootstrp(1000,@nanmedian,tmp));
neg = y-yboot(25,:);
pos =  yboot(975,:)-y;
x=1:3;
figure;hold on
errorbar(x,y,neg,pos,'LineStyle','none','LineWidth',1,'Color','k');
plot(x,y,'o','MarkerSize',15,'Color','k','LineWidth',1)
xlim([0.5 3.5])
ylim([0.35 0.67])
xticks(1:3)
xticklabels({'Within Rt. Hand Mvmts','b/w Rt. Hand and Lt. Hand','b/w Rt. Hand and all other'})
set(gcf,'Color','w')
ylabel('Jaccard spatial similarity coefficient')


%% (MAIN) PLOTTING DIFFERENCES BETWEEN MOVEMENTS AT THE SINGLE CHANNEL LEVEL
% with temporal clustering of ANOVA F-values 


clc;clear
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')


addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')


cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
load('C:\Data from F drive\B3 data\B3_high_res_erp_imagined_data')


load('ECOG_Grid_8596_000067_B3.mat')    
chMap=ecog_grid;
grid_layout = chMap;

% rename all the channels after accounting for 108,113,118
for i=109:112
    [x, y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-1;
end

for i=114:117
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-2;
end

for i=119:256
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-3;
end

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

% downsample the data to 50Hz 
idx=1:length(ImaginedMvmt);
for i=1:length(idx)
    data = ERP_Data{idx(i)};    
    bad_ch = [108 113 118];
    good_ch_idx=ones(256,1);
    good_ch_idx(bad_ch)=0;
    good_ch_idx=logical(good_ch_idx);
    data=data(:,good_ch_idx,:);
    % downsample 
    tmp_data=[];
    for j=1:size(data,3)
        tmp =  squeeze(data(:,:,j));
        tmp = resample(tmp,50,1e3);
        tmp_data(:,:,j) = tmp;
    end
    ERP_Data{idx(i)} = tmp_data;
end


%ch=103;% rt hand
ch=219; % face
figure;hold on
set(gcf,'Color','w')
%cmap={'r','g','b','y','m'};
cmap={'m','b','k'};
opt=statset('UseParallel',true);
data_overall=[];
idx = [10 12 16];
for ch=1:253
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

        %data_overall(i,:,:)=data';

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
    title(num2str(ch))
    waitforbuttonpress
    clf
end


% run the ANOVA test at each time-point and do temporal clustering
% dependent variable -> erp
% independent variable -> movement type
% fitrm
% 
% Fvalues=[];
% parfor i=1:size(data_overall,2)
%     disp(i)
%     a = (squeeze(data_overall(1:3,i,:)))';
%     tmp_bad = zscore(a);
%     artifact_check = logical(abs(tmp_bad)>3);
%     a(artifact_check)=NaN;   
%     a=a(:);
%     erps=a;
%     mvmt = num2str([ones(size(a,1)/3,1);2*ones(size(a,1)/3,1);3*ones(size(a,1)/3,1)]);
%     subj = table([1],'VariableNames',{'Subject'});
%     data = table(mvmt,erps);
%     rm=fitrm(data,'erps~mvmt');
%     ranovatbl = anova(rm);
%     Fvalues(i) = ranovatbl{2,6};
% end
% 
% % get the boot values
% Fvalues_boot=[];
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
%     Fvalues_tmp=[];
%     for iter=1:750      
%         a_tmp=[];
%         for j=1:size(a,2)
%             a1 = randi([1,size(a,1)],size(a,1),1);
%             a_tmp(:,j) = a(a1,j);
%         end
%         a_tmp=a_tmp(:);
%         erps=a_tmp;
%         mvmt = num2str([ones(size(erps,1)/3,1);...
%             2*ones(size(erps,1)/3,1);3*ones(size(erps,1)/3,1)]);
%         data = table(mvmt,erps);
%         rm=fitrm(data,'erps~mvmt');
%         ranovatbl = anova(rm);
%         Fvalues_tmp(iter) = ranovatbl{2,6};
%     end
%     Fvalues_boot(i,:) = Fvalues_tmp;
% end


%%% using just simple regular anova instead of repeated measures
Fvalues=[];pval=[];
parfor i=1:size(data_overall,2)
    disp(i)
    a = (squeeze(data_overall(1:3,i,:)))';
    tmp_bad = zscore(a);
    artifact_check = logical(abs(tmp_bad)>3);
    a(artifact_check)=NaN;   
    [P,ANOVATAB,STATS]  = anova1(a,[],'off');   
    Fvalues(i) = ANOVATAB{2,5};
    pval(i) = ANOVATAB{2,6};
end
%figure;plot(Fvalues)
%figure;plot(pval);hline(0.05)

% get the boot values using simple anova1
Fvalues_boot=[];p_boot=[];
parfor i=1:size(data_overall,2)
    disp(i)
    a = (squeeze(data_overall(1:3,i,:)))';
    tmp_bad = zscore(a);
    artifact_check = logical(abs(tmp_bad)>3);
    a(artifact_check)=NaN;

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

        [P,ANOVATAB,STATS]  = anova1(a_tmp,[],'off');
        Fvalues_tmp(iter) = ANOVATAB{2,5};
        ptmp(iter)  = ANOVATAB{2,6};
    end
    Fvalues_boot(i,:) = Fvalues_tmp;
    p_boot(i,:) = ptmp;
end


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
plot(tt,mask,'r','LineWidth',2)
