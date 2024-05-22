
clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')

foldername='20210615';
foldernames=dir(fullfile(root_path,foldername,'Robot3DArrow'));
foldernames=foldernames(3:end);

%non_pooling=[5,8,11]

pooling=[];non_pooling=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path,foldername,...
        'Robot3DArrow',foldernames(i).name,'BCI_Fixed');

    if exist(folderpath,'dir') > 0
        files=findfiles('',folderpath);
        if ~isempty(files)
            load(files{1})
            if TrialData.Params.ChPooling ==1
                pooling = [pooling i];
            elseif TrialData.Params.ChPooling ==0
                non_pooling = [non_pooling i];
            end
        end
    end
end


pooling_files=[];
for i=1:length(pooling)
      folderpath = fullfile(root_path,foldername,...
        'Robot3DArrow',foldernames(pooling(i)).name,'BCI_Fixed');
      
      pooling_files =[pooling_files;findfiles('',folderpath)'];
end

acc_pooling = accuracy_online_data(pooling_files);
figure;imagesc(acc_pooling*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_pooling,1)
    for k=1:size(acc_pooling,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_pooling(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_pooling(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head','Lips','Tongue','Both middle'})
yticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head','Lips','Tongue','Both middle'})
title([num2str(mean(diag(acc_pooling*100))) '% accuracy'])

non_pooling_files=[];
for i=1:length(non_pooling)
      folderpath = fullfile(root_path,foldername,...
        'Robot3DArrow',foldernames(non_pooling(i)).name,'BCI_Fixed');
      
      non_pooling_files =[non_pooling_files;findfiles('',folderpath)'];
end

acc_non_pooling = accuracy_online_data(non_pooling_files);
figure;imagesc(acc_non_pooling*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_non_pooling,1)
    for k=1:size(acc_non_pooling,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_non_pooling(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_non_pooling(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head','Lips','Tongue','Both middle'})
yticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head','Lips','Tongue','Both middle'})
title([num2str(mean(diag(acc_non_pooling*100))) '% accuracy'])


% plot individual block examples
pooling_acc_blks=[];
for i=1:length(pooling)
    folderpath = fullfile(root_path,foldername,...
        'Robot3DArrow',foldernames(pooling(i)).name,'BCI_Fixed');
    tmp = accuracy_online_data(findfiles('',folderpath)');
    pooling_acc_blks(i) = mean(diag(tmp));
    %pooling_acc_blks =[pooling_acc_blks ;(diag(tmp))];
end
 %pooling_acc_blks=pooling_acc_blks+0.001*randn(size(pooling_acc_blks));

non_pooling_acc_blks=[];
for i=1:length(non_pooling)
    folderpath = fullfile(root_path,foldername,...
        'Robot3DArrow',foldernames(non_pooling(i)).name,'BCI_Fixed');
    tmp = accuracy_online_data(findfiles('',folderpath)');
    non_pooling_acc_blks(i) = mean(diag(tmp));
    %non_pooling_acc_blks =[non_pooling_acc_blks; (diag(tmp))];
end

if length(pooling_acc_blks)>length(non_pooling_acc_blks)
    non_pooling_acc_blks(end+1:length(pooling_acc_blks)) = NaN;
else
    pooling_acc_blks(end+1:length(non_pooling_acc_blks)) = NaN;
end

figure;boxplot(100*[non_pooling_acc_blks' pooling_acc_blks'])
xticks(1:2)
xticklabels({'No Pooling','Pooling'})
set(gcf,'Color','w')
box off
ylabel('Accuracy')
yticks([30:20:90])


 [P,H,STATS] = ranksum(non_pooling_acc_blks,pooling_acc_blks,'method','approximate')



%% PLOTTING DIFFERENCES IN POOLED VS UNPOOLED SMOOTHED DATA - variance stats
% PLOTTING ERPS ON SINGLE CHANNEL AS COMPARED TO POOLED DATA

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
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
    features  = TrialData.NeuralFeatures;
    features = cell2mat(features);
    features = features(769:end,:);

    features1  = TrialData.SmoothedNeuralFeatures;
    features1 = cell2mat(features1);
    features1 = features1(769:end,:);



    %features = features(513:640,:);
    fs = TrialData.Params.UpdateRate;
    kinax = TrialData.TaskState;
    state1 = find(kinax==1);
    state2 = find(kinax==2);
    state3 = find(kinax==3);
    state4 = find(kinax==4);
    tmp_data = features(:,state3);
    tmp_data_p = features1(:,state3);
    idx1= ones(length(state1),1);
    idx2= 2*ones(length(state2),1);
    idx3= 3*ones(length(state3),1);
    idx4= 4*ones(length(state4),1);

    % interpolate
    tb = (1/fs)*[1:size(tmp_data,2)];
    t=(1/fs)*[1:10];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    tmp_data_p = interp1(tb,tmp_data_p',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');

    % now stick all the data together
    trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];
    data_p = [features1(:,[state1 state2]) tmp_data_p features1(:,[state4])];

    % correction
    if size(data,2)<40
        len = size(data,2);
        lend = 40-len;
        data = [data(:,1:lend) data];
        data_p = [data_p(:,1:lend) data_p];
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
    temp=data_p;
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
            D7pool = cat(3,D7pool,data);
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


% plotting a comparison of pooled and unpooled data
tmp = D1;
% first smooth each trial
for i=1:size(tmp,3)
    for j=1:size(tmp,1)
        tmp(j,:,i) = smooth(squeeze(tmp(j,:,i)),5);
    end
end
% now spatially pool
chmap=TrialData.Params.ChMap;
ch = [100 97 103 106];
tmp_p = tmp(ch,:,:);
tmp_p = squeeze(mean(tmp_p,1));

figure;
tt = (1/5)*[0:size(D1,2)-1];
plot(tt,squeeze(D1(97,:,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
hold on
plot(tt,squeeze(mean(D1(97,:,:),3)),'Color','b','LineWidth',1)
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
timm = tim*(1/5);
hline(0,'--r')
axis tight
h=vline(timm(1),'r');
h1=vline(timm(2),'g');
%set(h,'Color','k')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
xlabel('Time (s)')
ylabel('z-score')
title('Ch Data')
%xticks ''
%xlabel ''

figure;
tt = (1/5)*[0:size(D1,2)-1];
plot(tt,tmp_p,'Color',[.5 .5 .5 .5],'LineWidth',1)
hold on
plot(tt,mean(tmp_p,2),'Color','b','LineWidth',1)
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
timm = tim*(1/5);
hline(0,'--r')
axis tight
h=vline(timm(1),'r');
h1=vline(timm(2),'g');
%set(h,'Color','k')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
xlabel('Time (s)')
ylabel('z-score')
title('Ch pooled Data')
%xticks ''
%xlabel ''

%%% OLD

% % plot the variances across channels for single trials
% var_raw=[];
% var_pool=[];
% for i=1:size(D1,3)
%     tmp_raw =  squeeze(D1(:,:,i));
%     var_raw = [var_raw var(tmp_raw(:))];
% 
%     tmp_pool =  squeeze(D1pool(:,:,i));
%     var_pool = [var_pool var(tmp_pool(:))];
% end
% 
% figure;
% boxplot([var_raw' var_pool'])
% xticklabels({'Raw','Pooled'})
% set(gca,'FontSize',12)
% set(gcf,'Color','w')
% ylabel('Variance')


%%% NEW
% plot the variances for single trials per channel 
var_raw=[];
for i=1:size(D1,1)
     tmp_raw =  squeeze(D1(i,:,:));
     var_raw = [var_raw var(tmp_raw(:))];
end
var_pool=[];
for i=1:size(D1pool,1)    
    tmp_pool =  squeeze(D1pool(i,:,:));
    var_pool = [var_pool var(tmp_pool(:))];
end

figure;
var_pool(end+1:length(var_raw)) = NaN;
boxplot([var_raw' var_pool'])
xticklabels({'Raw','Pooled'})
set(gca,'FontSize',12)
set(gcf,'Color','w')
ylabel('Variance')
box off

 [P,H,STATS] = ranksum(var_raw,var_pool)

% plot an image of hG activity in the hand knob region
tmp = squeeze(mean(D1(:,15,12:16),3));
figure;
imagesc(tmp(chmap))
axis off
set(gcf,'Color','w')

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



