%% ERPs
% load a particular day'a data.
% Normalize the length to be constant across trials
% plot ERPs by target


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
foldernames = {'20230111','20230118','20230119','20230125'};
cd(root_path)

files=[];
for i=length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        if exist(filepath)
            files = [files;findfiles('',filepath)'];
        end
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    data=[];
    features  = TrialData.SmoothedNeuralFeatures;
    features = cell2mat(features);
    features = features(769:896,:);
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
    
    % interpolate
    tb = (1/fs)*[1:size(tmp_data,2)];
    t=(1/fs)*[1:10];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');
    
    % now stick all the data together
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];

    % now get the ERPs
    if TrialData.TargetID == TrialData.SelectedTargetID
        if TrialData.TargetID == 1
            D1 = cat(3,D1,data);            
        elseif TrialData.TargetID == 2
            D2 = cat(3,D2,data);
        elseif TrialData.TargetID == 3
            D3 = cat(3,D3,data);
        elseif TrialData.TargetID == 4
            D4 = cat(3,D4,data);
        elseif TrialData.TargetID == 5
            D5 = cat(3,D5,data);
        elseif TrialData.TargetID == 6
            D6 = cat(3,D6,data);
        end        
    end    
end

% plot the ERPs now
chMap=TrialData.Params.ChMap;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
for i = 1:size(D1,1)
    [x y] = find(chMap==d);
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
    plot(erps, 'color', [0.4 0.4 0.4 ]);
    ylabel (num2str(i))
    axis tight
    ylim([-2 2])
    plot(mean(erps,2),'r','LineWidth',1.5)
    %set(gca,'LineWidth',1)
    %vline([time(2:4)])   
    h=vline(tim);
    %set(h,'LineWidth',1)
    set(h,'Color','b')
    h=hline(0);
    set(h,'LineWidth',1.5)    
    if i~=102
        yticklabels ''
        xticklabels ''
    else
        %xticks([tim])
        %xticklabels({'S1','S2','S3','S4'})
    end
    d = d+1;
end


%% ERP B3 FROM THE ONLINE EXPERIMENT (MAIN)
% do it for a few days for all 7 actions


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\';
foldernames = {'20231006'};
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))

files=[];
for i=length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'HandOnline');
    D=dir(folderpath);
    for j=3:length(D)        
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        if exist(filepath)
            files = [files;findfiles('',filepath)'];
        end
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
D8=[];
D9=[];
D10=[];
D11=[];
D12=[];
time_to_target=zeros(2,7);
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    features = cell2mat(features);
    %1537:1792 is hG, 257:512 is delta and 1025:1280 is beta
    features = features(1537:1792,:); 
    %features = features(513:640,:);
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
    
    % interpolate
    tb = (1/fs)*[1:size(tmp_data,2)];
    t=(1/fs)*[1:10];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');
    
    % now stick all the data together
    %trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];
    
    % correction
%     if length(state1)<8
%         data  =[data(:,1) data];
%     end
    
    % store the time to target data and number of trials 
    %time_to_target(1,TrialData.TargetID) = time_to_target(1,TrialData.TargetID)+trial_dur;    
    %time_to_target(2,TrialData.TargetID) = time_to_target(2,TrialData.TargetID)+1;
    

    % now get the ERPs
    %if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=5
        if TrialData.TargetID == 1
            D1 = cat(3,D1,data);            
        elseif TrialData.TargetID == 2
            D2 = cat(3,D2,data);
        elseif TrialData.TargetID == 3
            D3 = cat(3,D3,data);
        elseif TrialData.TargetID == 4
            D4 = cat(3,D4,data);
        elseif TrialData.TargetID == 5
            D5 = cat(3,D5,data);
        elseif TrialData.TargetID == 6
            D6 = cat(3,D6,data);
        elseif TrialData.TargetID == 7
            D7 = cat(3,D7,data);
        elseif TrialData.TargetID == 8
            D8 = cat(3,D8,data);
        elseif TrialData.TargetID == 9
            D9 = cat(3,D9,data);
        elseif TrialData.TargetID == 10
            D10 = cat(3,D10,data);
        elseif TrialData.TargetID == 11
            D11 = cat(3,D11,data);
        elseif TrialData.TargetID == 12
            D12 = cat(3,D12,data);
        end        
    %end    
end

% prints the average time to target here
%time_to_target(1,:)./time_to_target(2,:)

% load the data into a common struct
clear D
D{1} =  D1;
D{2} =  D2;
D{3} =  D3;
D{4} =  D4;
D{5} =  D5;
D{6} =  D6;
D{7} =  D7;
D{8} =  D8;
D{9} =  D9;
D{10} =  D10;
D{11} =  D11;
D{12} =  D12;


% plot the ERPs with bootstrapped C.I. shading
bad_ch = [108,113,118];
load('ECOG_Grid_8596_000063_B3.mat')
chMap=ecog_grid;
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
mvmts = {'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'};
sig_ch = zeros(256,length(D));
for iter=1:length(D)
    figure
    ha=tight_subplot(size(chMap,1),size(chMap,2));
    d = 1;
    set(gcf,'Color','w')
    set(gcf,'WindowState','maximized')    
    data  = D{iter};
    tmp_sig=zeros(256,1);
    for i = 1:size(data,1)
        if sum(i==bad_ch) == 0

            [x y] = find(chMap==i);
            if x == 1
                axes(ha(y));
                %subplot(8, 16, y)
            else
                s = 23*(x-1) + y;
                axes(ha(s));
                %subplot(8, 16, s)
            end           
            hold on
            erps =  squeeze(data(i,:,:));

            chdata = erps;
            % zscore the data to the first 8 time-bins
            tmp_data=chdata(1:8,:);
            m = mean(tmp_data(:));
            s = std(tmp_data(:));
            chdata = (chdata -m)./s;

            % get the confidence intervals
            m = mean(chdata,2);
            mb = sort(bootstrp(1000,@mean,chdata'));
            tt=1:size(data,2);
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
                tmp_data=tmp(1:8,:);
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
            idx=11:20;
            mstat = m((idx));
            pval=[];
            for j=1:length(idx)
                pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
            end

            [pfdr, pval1]=fdr(1-pval,0.05);pfdr;
            res=sum((1-pval)<=pfdr);
            if res>=2
                suc=1;
            else
                suc=0;
            end

            % beautify
            ylabel (num2str(i))
            axis tight
            ylim([-2.5 4])
            %disp(max(chdata(:)))
            %set(gca,'LineWidth',1)
            %vline([time(2:4)])
            h=vline(tim);
            %set(h,'LineWidth',1)
            set(h,'Color','k')
            h=hline(0);
            set(h,'LineWidth',1.5)
            if i~=104
                yticklabels ''
                xticklabels ''
            else
                %xticks([tim])
                %xticklabels({'S1','S2','S3','S4'})
                %yticks([-2:1:5])
            end

            if suc==1
                box on
                set(gca,'LineWidth',2)
                set(gca,'XColor','g')
                set(gca,'YColor','g')
                tmp_sig(i)=1;
            end
            d = d+1;
        end
    end    
    sgtitle(mvmts{iter});
    filename = fullfile('F:\DATA\ecog data\ECoG BCI\Results\B3 results\ERPs Hand\Day 7',mvmts{iter});
    saveas(gcf,filename)
    set(gcf,'PaperPositionMode','auto')
    set(gcf, 'Renderer', 'painters');
    print(gcf,filename,'-dpng','-r300')
    sig_ch(:,iter)=tmp_sig;
end

% 
% % plotting significant channels 
% channel_layout = [];
% for i=1:23:253
%     channel_layout = [channel_layout;i:i+22];
% end
% channel_layout = fliplr(channel_layout);
% imaging_B3;
% close all
% for i=1:size(sig_ch,2)
%     figure;
%     set(gcf,'Color','w')
%     c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
%     e_h = el_add(elecmatrix, 'color', 'w','msize',2);
%     tmp_ch =  sig_ch(:,i);
%     for j=1:length(tmp_ch)
%         [x,y] = find(ecog_grid==j);
%         query_ch = channel_layout(x,y);
%         if tmp_ch(j)==1
%             e_h = el_add(elecmatrix(query_ch,:), 'color', 'b','msize',6);
%         end
%     end
%     title(mvmts{i})
%     view(-93.4,0.673)
%     set(gcf,'Position',[680,342,791,636])
%     filename = fullfile('F:\DATA\ecog data\ECoG BCI\Results\B3 results\ERPs Arrow\beta',[mvmts{i} ' brain']);            
%     print(gcf,filename,'-dpng','-r300')
% end

%[az,el]=view %gets the view
% pos = get(gcf,'Position') % gets the position


%% ERPS WITH channel STATS
% find trials where it takes less than 2s for the correct action to be
% decoded


clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20201218','20210108','20210115','20210128','20210201','20210212','20210219','20210226',...
    '20210305','20210312','20210319','20210402','20210326','20210409'};
cd(root_path)

files=[];
for i=1%length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow')
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
time_to_target=zeros(2,6);
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    features = cell2mat(features);
    features = features(769:end,:); % hG
    %features = features(513:640,:); %beta
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
    
    % interpolate
    tb = (1/fs)*[1:size(tmp_data,2)];
    t=(1/fs)*[1:10];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');
    
    % now stick all the data together
    trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];
    
    % correction if there is an error thrown somewhere 
    if length(state1)<8
        data  =[data(:,1) data];
    end
%     
    % store the time to target data
    time_to_target(2,TrialData.TargetID) = time_to_target(2,TrialData.TargetID)+1;
    if trial_dur<=4
        time_to_target(1,TrialData.TargetID) = time_to_target(1,TrialData.TargetID)+1;
    end

    % now get the ERPs
    if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=4
        if TrialData.TargetID == 1
            D1 = cat(3,D1,data);            
        elseif TrialData.TargetID == 2
            D2 = cat(3,D2,data);
        elseif TrialData.TargetID == 3
            D3 = cat(3,D3,data);
        elseif TrialData.TargetID == 4
            D4 = cat(3,D4,data);
        elseif TrialData.TargetID == 5
            D5 = cat(3,D5,data);
        elseif TrialData.TargetID == 6
            D6 = cat(3,D6,data);
        end        
    end    
end


time_to_target(1,:)./time_to_target(2,:)

% plot the ERPs now
chMap=TrialData.Params.ChMap;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
for i = 1:size(D1,1)
    [x y] = find(chMap==d);
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
    plot(erps, 'color', [0.4 0.4 0.4 ]);
    ylabel (num2str(i))
    axis tight
    ylim([-2 2])
    plot(mean(erps,2),'r','LineWidth',1.5)
    %set(gca,'LineWidth',1)
    %vline([time(2:4)])   
    h=vline(tim);
    %set(h,'LineWidth',1)
    set(h,'Color','b')
    h=hline(0);
    set(h,'LineWidth',1.5)    
    if i~=102
        yticklabels ''
        xticklabels ''
    else
        %xticks([tim])
        %xticklabels({'S1','S2','S3','S4'})
    end
    d = d+1;
end
% 
% 
% bootstrapped confidnce intervals for a particular channel
% chdata = ch100;
% 
% zscore the data to the first 8 time-bins
% tmp_data=chdata(1:8,:);
% m = mean(tmp_data(:));
% s = std(tmp_data(:));
% chdata = (chdata -m)./s;
%  
% 
% m = mean(chdata,2);
% mb = sort(bootstrp(1000,@mean,chdata'));
% figure;
% hold on
% plot(m,'b')
% plot(mb(25,:),'--b')
% plot(mb(975,:),'--b')
% hline(0)
% 
% % shuffle the data and see the results
% tmp_mean=[];
% for i=1:1000
%     %tmp = circshift(chdata,randperm(size(chdata,1),1));
%     tmp = chdata;
%     tmp(randperm(numel(chdata))) = tmp;
%     tmp_data=tmp(1:8,:);
%     m = mean(tmp_data(:));
%     s = std(tmp_data(:));
%     tmp = (tmp -m)./s;
%     tmp_mean(i,:) = mean(tmp,2);
% end
% 
% tmp_mean = sort(tmp_mean);
% plot(tmp_mean(25,:),'--r')
% plot(tmp_mean(975,:),'--r')



% plot the ERPs with bootstrapped C.I. shading
chMap=TrialData.Params.ChMap;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
for i = 1:size(D5,1)
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
    erps =  squeeze(D5(i,:,:));
    
    chdata = erps;
    % zscore the data to the first 8 time-bins
    tmp_data=chdata(1:8,:);
    m = mean(tmp_data(:));
    s = std(tmp_data(:));
    chdata = (chdata -m)./s;
    
    % get the confidence intervals
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(erps,1);
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
        tmp_data=tmp(1:8,:);
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
    idx=14:25;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end
    
    res=sum((1-pval)<=0.05);
    if res>=floor(length(idx)/2)
        suc=1;
    else
        suc=0;
    end
    
    % beautify
    ylabel (num2str(i))
    axis tight
    ylim([-2 5])     
    %set(gca,'LineWidth',1)
    %vline([time(2:4)])   
    h=vline(tim);
    %set(h,'LineWidth',1)
    set(h,'Color','k')
    h=hline(0);
    set(h,'LineWidth',1.5)    
    if i~=102
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





%% putting it all together ERPs with stats across days

clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20201218','20210115','20210128','20210201','20210212','20210219','20210226',...
    '20210305','20210312','20210319','20210402','20210326','20210409','20210416'};
cd(root_path)

T1=[];
T2=[];
T3=[];
T4=[];
T5=[];
T6=[];
time_to_target_overall=[];
for i=length(foldernames)
    disp(['processing folder ' num2str(i) ' of ' num2str(length(foldernames))])
    
    % get all the files
    files=[];
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
    
    % now load all the data
    [D1,D2,D3,D4,D5,D6,time_to_target] = load_erp_data(files);
    time_to_target = time_to_target(1,:)./time_to_target(2,:);
    time_to_target_overall(i,:) = time_to_target;
    % chmap file
    load(files{1})
    chMap=TrialData.Params.ChMap;
    
    % get the mask of sig. channels for each movement type  and save ERPs
    clear res1 res2 res3 res res5 res6
    res1 = erp_stats(D1,chMap,foldernames{i},'T1',0);
    res2 = erp_stats(D2,chMap,foldernames{i},'T2',0);
    res3 = erp_stats(D3,chMap,foldernames{i},'T3',0);
    res4 = erp_stats(D4,chMap,foldernames{i},'T4',0);
    res5 = erp_stats(D5,chMap,foldernames{i},'T5',0);
    res6 = erp_stats(D6,chMap,foldernames{i},'T6',0);
    
    % collate across days
    T1(i,:,:) = res1;
    T2(i,:,:) = res2;
    T3(i,:,:) = res3;
    T4(i,:,:) = res4;
    T5(i,:,:) = res5;
    T6(i,:,:) = res6;
    
end

% save the results
save ERP_data

% plotting the sig channels over days using all the days
figure
ha=tight_subplot(7,2);
for i=1:14
    axes(ha(i))
    if i==4 || i==6 || i==8 || i==14
        box off
        axis off
    else
        imagesc(squeeze(T1(i,:,:)));
        colormap parula
        caxis([0 1])
        xticks ''
        yticks ''
        box on
    end
end
set(gcf,'Color','w')

% plotting the sig channels over days using only useful days
idx = [1:3 5 7 9:13];
data = T6(idx,:,:);
figure
ha=tight_subplot(5,2);
for i=1:10
    axes(ha(i))
    imagesc(squeeze(data(i,:,:)));
    colormap parula
    caxis([0 1])
    xticks ''
    yticks ''
    box on
    
end
set(gcf,'Color','w')

figure
ha=tight_subplot(3,2);
axes(ha(1))
imagesc(squeeze(sum(T1(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
axes(ha(2))
imagesc(squeeze(sum(T2(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
axes(ha(3))
imagesc(squeeze(sum(T3(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
axes(ha(4))
imagesc(squeeze(sum(T4(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
axes(ha(5))
imagesc(squeeze(sum(T5(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
axes(ha(6))
imagesc(squeeze(sum(T6(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
set(gcf,'Color','w')



figure;imagesc(squeeze(sum(T1,1)));colormap bone
caxis([0 13])
figure;imagesc(squeeze(sum(T2,1)));colormap bone
caxis([0 13])
figure;imagesc(squeeze(sum(T3,1)));colormap bone
caxis([0 13])
figure;imagesc(squeeze(sum(T4,1)));colormap bone
caxis([0 13])
figure;imagesc(squeeze(sum(T5,1)));colormap bone
caxis([0 13])
figure;imagesc(squeeze(sum(T6,1)));colormap bone
caxis([0 13])


%% %% ERPS WITH channel STATS NEW IMAGINED ACTIONS
% find trials where it takes less than 2s for the correct action to be
% decoded


clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20210915'};
cd(root_path)

files=[];
for i=length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=6:length(D)-1
        folderpath,D(j).name
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
time_to_target=zeros(2,6);
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    features = cell2mat(features);
    features = features(769:end,:);
    %features = features(513:640,:);
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
    
    % interpolate
    tb = (1/fs)*[1:size(tmp_data,2)];
    t=(1/fs)*[1:10];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');
    
    % now stick all the data together
    trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];
    
    % correction
    if length(state1)<8
        data  =[data(:,1) data];
    end
    
    % store the time to target data
    time_to_target(2,TrialData.TargetID) = time_to_target(2,TrialData.TargetID)+1;
    if trial_dur<=3
        time_to_target(1,TrialData.TargetID) = time_to_target(1,TrialData.TargetID)+1;
    end

    % now get the ERPs
    if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=3
        if TrialData.TargetID == 1
            D1 = cat(3,D1,data);            
        elseif TrialData.TargetID == 2
            D2 = cat(3,D2,data);
        elseif TrialData.TargetID == 3
            D3 = cat(3,D3,data);
        elseif TrialData.TargetID == 4
            D4 = cat(3,D4,data);
        elseif TrialData.TargetID == 5
            D5 = cat(3,D5,data);
        elseif TrialData.TargetID == 6
            D6 = cat(3,D6,data);
        end        
    end    
end


time_to_target(1,:)./time_to_target(2,:)


% plot the ERPs with bootstrapped C.I. shading
chMap=TrialData.Params.ChMap;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
for i = 1:size(D2,1)
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
    erps =  squeeze(D2(i,:,:));
    
    chdata = erps;
    % zscore the data to the first 8 time-bins
    tmp_data=chdata(1:8,:);
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
        tmp_data=tmp(1:8,:);
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
    idx=13:27;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end
    
    res=sum((1-pval)<=0.05);
    if res>=7
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
    if i~=102
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

%% putting it all together ERPs with stats across days

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20201218','20210115','20210128','20210201','20210212','20210219','20210226',...
    '20210305','20210312','20210319','20210402','20210326','20210409','20210416','20210915'};
cd(root_path)
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')

T1=[];
T2=[];
T3=[];
T4=[];
T5=[];
T6=[];
time_to_target_overall=[];
for i=length(foldernames)
    disp(['processing folder ' num2str(i) ' of ' num2str(length(foldernames))])
    
    % get all the files
    files=[];
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
    
    % now load all the data
    [D1,D2,D3,D4,D5,D6,time_to_target] = load_erp_data(files);
    time_to_target = time_to_target(1,:)./time_to_target(2,:);
    time_to_target_overall(i,:) = time_to_target;
    % chmap file
    load(files{1})
    chMap=TrialData.Params.ChMap;
    
    % get the mask of sig. channels for each movement type  and save ERPs
    clear res1 res2 res3 res res5 res6
    res1 = erp_stats(D1,chMap,foldernames{i},'T1',1);
    res2 = erp_stats(D2,chMap,foldernames{i},'T2',0);
    res3 = erp_stats(D3,chMap,foldernames{i},'T3',0);
    res4 = erp_stats(D4,chMap,foldernames{i},'T4',0);
    res5 = erp_stats(D5,chMap,foldernames{i},'T5',0);
    res6 = erp_stats(D6,chMap,foldernames{i},'T6',0);
    
    % collate across days
    T1(i,:,:) = res1;
    T2(i,:,:) = res2;
    T3(i,:,:) = res3;
    T4(i,:,:) = res4;
    T5(i,:,:) = res5;
    T6(i,:,:) = res6;
    
end

% save the results
save ERP_data

% plotting the sig channels over days using all the days
figure
ha=tight_subplot(7,2);
for i=1:14
    axes(ha(i))
    if i==4 || i==6 || i==8 || i==14
        box off
        axis off
    else
        imagesc(squeeze(T1(i,:,:)));
        colormap parula
        caxis([0 1])
        xticks ''
        yticks ''
        box on
    end
end
set(gcf,'Color','w')

% plotting the sig channels over days using only useful days
idx = [1:3 5 7 9:13];
data = T1(idx,:,:);
figure
ha=tight_subplot(5,2);
for i=1:10
    axes(ha(i))
    imagesc(squeeze(data(i,:,:)));
    colormap parula
    caxis([0 1])
    xticks ''
    yticks ''
    box on
    
end
set(gcf,'Color','w')

figure
ha=tight_subplot(3,2);
axes(ha(1))
imagesc(squeeze(sum(T1(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
axes(ha(2))
imagesc(squeeze(sum(T2(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
axes(ha(3))
imagesc(squeeze(sum(T3(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
axes(ha(4))
imagesc(squeeze(sum(T4(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
axes(ha(5))
imagesc(squeeze(sum(T5(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
axes(ha(6))
imagesc(squeeze(sum(T6(idx,:,:),1)));colormap bone; caxis([0 10])
xticks ''
yticks ''
set(gcf,'Color','w')



figure;imagesc(squeeze(sum(T1,1)));colormap bone
caxis([0 13])
figure;imagesc(squeeze(sum(T2,1)));colormap bone
caxis([0 13])
figure;imagesc(squeeze(sum(T3,1)));colormap bone
caxis([0 13])
figure;imagesc(squeeze(sum(T4,1)));colormap bone
caxis([0 13])
figure;imagesc(squeeze(sum(T5,1)));colormap bone
caxis([0 13])
figure;imagesc(squeeze(sum(T6,1)));colormap bone
caxis([0 13])


%% PLOTTING ERPS ON SINGLE CHANNEL AS COMPARED TO POOLED DATA 





clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20210915'};
cd(root_path)

files=[];
for i=length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        folderpath,D(j).name
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D1pool=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
D7pool=[];
time_to_target=zeros(2,7);
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    features = cell2mat(features);
    features = features(769:end,:);
    %features = features(513:640,:);
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
    
    % interpolate
    tb = (1/fs)*[1:size(tmp_data,2)];
    t=(1/fs)*[1:10];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');
    
    % now stick all the data together
    trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];
    
    % correction
    if size(data,2)<40
        len = size(data,2);
        lend = 40-len;
        data = [data(:,1:lend) data];
    end
    %
    %     % store the time to target data
    %     time_to_target(2,TrialData.TargetID) = time_to_target(2,TrialData.TargetID)+1;
    %     if trial_dur<=2
    %         time_to_target(1,TrialData.TargetID) = time_to_target(1,TrialData.TargetID)+1;
    %     end
    
    % now get the ERPs
    if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=10
        if TrialData.TargetID == 1
            D1 = cat(3,D1,data);
        elseif TrialData.TargetID == 2
            D2 = cat(3,D2,data);
        elseif TrialData.TargetID == 3
            D3 = cat(3,D3,data);
        elseif TrialData.TargetID == 4
            D4 = cat(3,D4,data);
        elseif TrialData.TargetID == 5
            D5 = cat(3,D5,data);
        elseif TrialData.TargetID == 6
            D6 = cat(3,D6,data);            
        elseif TrialData.TargetID == 7
            D7 = cat(3,D7,data);
        end
    end
    
    
    
    % do it now for pooled data
    temp=data;
    new_temp=[];
    [xx yy] = size(TrialData.Params.ChMap);
    for k=1:size(temp,2)
        tmp1 = temp(1:128,k);tmp1 = tmp1(TrialData.Params.ChMap);
        %tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
        %tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
        pooled_data=[];
        for ii=1:2:xx
            for j=1:2:yy
                delta = (tmp1(ii:ii+1,j:j+1));delta=mean(delta(:)); % this is actualy hg
                % beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                % hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                %pooled_data = [pooled_data; delta; beta ;hg];
                pooled_data = [pooled_data; delta; ];
            end
        end
        new_temp= [new_temp pooled_data];
    end
    temp_data=new_temp;
    data = temp_data;
    
    if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=10
        if TrialData.TargetID == 1
            D1pool = cat(3,D1pool,data);
        end
    end
    
    if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=10
        if TrialData.TargetID == 7
            D1pool = cat(3,D1pool,data);
        end
    end
    
end

tmp = squeeze(D1pool(29,:,:));
figure;plot(tmp,'Color',[.5 .5 .5])
hold on
plot(mean(tmp,2),'b','LineWidth',1)

tmp = squeeze(D1(3,:,:));
figure;plot(tmp,'Color',[.5 .5 .5])
hold on
plot(mean(tmp,2),'b','LineWidth',1)


% plot ERPs single trial for target 1 for example
chMap=TrialData.Params.ChMap;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
for i = 1:size(D1,1)
    [x y] = find(chMap==d);
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
    plot(erps, 'color', [0.4 0.4 0.4 ]);
    ylabel (num2str(i))
    axis tight
    ylim([-2 2])
    plot(mean(erps,2),'r','LineWidth',1.5)
    %set(gca,'LineWidth',1)
    %vline([time(2:4)])   
    h=vline(tim);
    %set(h,'LineWidth',1)
    set(h,'Color','b')
    h=hline(0);
    set(h,'LineWidth',1.5)    
    if i~=102
        yticklabels ''
        xticklabels ''
    else
        %xticks([tim])
        %xticklabels({'S1','S2','S3','S4'})
    end
    d = d+1;
end



% plot the ERPs with bootstrapped C.I. shading
chMap=TrialData.Params.ChMap;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
tim(3)=25;
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
    % zscore the data to the first 5 time-bins 
    tmp_data=chdata(1:length(idx1),:);
    m = mean(tmp_data(:));
    s = std(tmp_data(:));
    chdata = (chdata -m)./s;
    
 
    
    
    % get the confidence intervals and plot
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(D1,2);
    
    % plotting confidence intervals
    %[fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
    %    ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);
    
    % plotting the raw ERPs
    plot(chdata,'Color',[.5 .5 .5 .5],'LineWidth',1)    
    hold on
    plot(m,'b','LineWidth',1)
    %plot(mb(25,:),'--b')
    %plot(mb(975,:),'--b')
    %hline(0)
    
    
    
    
    
    % shuffle the data for null confidence intervals
    tmp_mean=[];
    for j=1:1000
        %tmp = circshift(chdata,randperm(size(chdata,1),1));
        tmp = chdata;
        tmp(randperm(numel(chdata))) = tmp;
        tmp_data=tmp(1:8,:);
        m = mean(tmp_data(:));
        s = std(tmp_data(:));
        tmp = (tmp -m)./s;
        tmp_mean(j,:) = mean(tmp,2);
    end
    
    % plot null interval
    tmp_mean = sort(tmp_mean);
    %plot(tmp_mean(25,:),'--r')
    %plot(tmp_mean(975,:),'--r')
    [fillhandle,msg]=jbfill(tt,tmp_mean(25,:),tmp_mean(975,:)...
        ,[0.7 0.3 0.3],[0.7 0.3 0.3],1,.2);
    
        
    % statistical test
    % if the mean is outside confidence intervals in state 3
    m = mean(chdata,2);
    idx=13:25;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end
    
    res=sum((1-pval)<=0.01);
    if res>=9
        suc=1;
    else
        suc=0;
    end
    
    % beautify
    ylabel (num2str(i))
    axis tight
    ylim([-3 6])    
%     if max(chdata(:))>2
%         ylim([-2 max(chdata(:))]);
%     end
    %set(gca,'LineWidth',1)
    %vline([time(2:4)])   
    h=vline(tim(1:2));
    %set(h,'LineWidth',1)
    set(h,'Color','k')
    h=hline(0);
    set(h,'LineWidth',1.5)    
    if i~=102
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


% plot individual channel
erps =  squeeze(D1(25,:,:));
chdata = erps;
% zscore the data to the first 5 time-bins
tmp_data=chdata(1:length(idx1),:);
m = mean(tmp_data(:));
s = std(tmp_data(:));
chdata = (chdata -m)./s;
% now plot
figure;hold on
tt = (1/5)*[0:size(chdata,1)-1];
timm = tim*(1/5);
timm = timm/max(tt);
tt = tt./max(tt);
plot(tt,chdata,'Color',[.5 .5 .5 ],'LineWidth',1);
plot(tt,mean(chdata,2),'LineWidth',2,'Color','b')
hline(0)
axis tight
h=vline(timm(1:2));
set(h,'Color','k')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
xlabel('Normalized trial length (percent)')
ylabel('z-score')
title('Channel 25 Day 2')
xticks ''
xlabel ''
    

%% FOR grant ERPs
% plot single trial ERPs at an example channel, for right hand and then
% also for left hand. Plot both erps with CI as well as ERPs single trial
% denoised. Do it on the spatial avergaed grid 


clc;clear






