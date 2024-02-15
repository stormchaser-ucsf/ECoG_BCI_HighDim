%% ERPs of imagined actions higher sampling rate (MAIN)
% using hG and LMP, beta etc.

clc;clear
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')



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


% B1 grid layout
tic
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



idx = [1:length(ImaginedMvmt)];
%idx =  [1,10,30,25,20,28];
sig_ch_all=zeros(32,253);
loop_iter=750;
tfce_flag=false;
for i=1:length(idx)    
    load('C:\Data from F drive\B3 data\B3_high_res_erp_imagined_data.mat',...
        'ERP_Data')
    data = ERP_Data{idx(i)};
    clear ERP_Data
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
            (tboot.^2),pboot,LIMO,2,0.01,0);
        figure;subplot(3,1,1)
        tt=linspace(-3,4,size(t_scores,2));
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

