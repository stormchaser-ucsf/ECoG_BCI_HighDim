%% BLOCK BY BLOCK BIT RATE CALCULATIONS ACROSS DAYS - B1
%%%%(MAIN)


clc;clear
%close all


root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath 'C:\Users\nikic\Documents\MATLAB'


%foldernames = {'20220601'};
foldernames = {'20210813','20210818','20210825','20210827','20210901','20210903',...
    '20210910','20210915','20210917','20210922','20210924'};


cd(root_path)
folders={};
br_across_days={};
time2target_days=[];
acc_days=[];
conf_matrix_overall=[];
overall_trial_accuracy=zeros(7);
for i=1:10%length(foldernames)

    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    if i==1
        idx = [1 2 3 4];
        D = D(idx);
    elseif i==2
        idx = [1 2 3 4 6 7];
        D = D(idx);
    elseif i==3
        idx = [1 2 5 6];
        D = D(idx);
    elseif i==6
        idx = [1 2 3 4 5 6];
        D = D(idx);
    elseif i==8
        idx = [1 2 3 4 5 6 7];
        D = D(idx);
    elseif i==9
        idx = [1 2 5 6 7 9 10];
        D = D(idx);
    elseif i==11
        idx = [1 2 3  5 6 9 10 11];
        D = D(idx);
    elseif i == 10
        idx = [1 2 3 4 5  7 8];
        D = D(idx);
    end
    br=[];acc=[];time2target=[];
    for j=3:length(D)
        files=[];
        filepath=fullfile(folderpath,D((j)).name,'BCI_Fixed');
        if exist(filepath)
            filepath
            files = [files;findfiles('mat',filepath)'];
            folders=[folders;filepath];
        end
        if length(files)>0
            [b,a,t,T,ov_acc] = compute_bitrate(files,7);
            %[b,a,t,T] = compute_bitrate_constTime(files,7);
            conf_matrix_overall = cat(3,conf_matrix_overall,T);
            br = [br b];
            acc = [acc mean(a)];
            time2target = [time2target; mean(t)];
            overall_trial_accuracy = overall_trial_accuracy + ov_acc;
            %[br, acc ,t] = [br compute_bitrate(files)];
        end
    end
    close all
    br_across_days{i}=br;
    time2target_days{i} = time2target(:);
    acc_days{i} = acc(:);
    %time2target_days = [time2target_days ;time2target(:)];
    %acc_days = [acc_days ;acc(:)];
end

%plot results as a scatter plot
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95');
figure;hold on
br=[];
brh=[];
%cmap = brewermap(11,'blues');
%cmap=flipud(cmap);
%cmap=cmap(1:length(br_across_days),:);
%cmap=flipud(cmap);
cmap = turbo(7);%turbo(length(br_across_days));
for i=1:7%length(br_across_days)
    tmp = br_across_days{i};
    brh = [brh tmp];
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    br(i) = median(tmp);
end
plot(br(1:end),'k','LineWidth',2)
days={'1','5','12','14','19','21','28','32','35','40','42'};
xticks(1:length(br))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('BitRate')
set(gca,'LineWidth',1)
%set(gca,'Color',[.85 .85 .85])
xlim([0 7.5])
ylim([0 3.5])
%

figure
boxplot(brh,'whisker',1.75)
set(gcf,'Color','w')
xticks(1)
xticklabels('PnP Experiment 1')
ylabel('Effective bit rate')
set(gca,'FontSize',12)
box off
xlim([.75 1.25])
ylim([0 3.5])
yticks([0:.5:3.5])
set(gca,'LineWidth',1,'TickLength',[0.025 0.025]);

figure;hold on
acc=[];
acch=[];
acc_ind_days=[];
days_track=[];
acc_med=[];
for i=1:7%length(acc_days)
    tmp  = acc_days{i};
    acc_med(i) = median(tmp);
    acc_ind_days=[acc_ind_days tmp'];
    days_track = [days_track str2num(days{i})*ones(1,length(tmp))];
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    acc(i) = median(tmp);
    acch = [acch ;tmp];
end
plot(acc,'k','LineWidth',2)
ylim([0 1])
xticks(1:length(acc))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Decoder Accuracy')
set(gca,'LineWidth',1)
xlim([0.5 7.5])
h=hline(1/7);
set(h,'LineWidth',2)
yticks([0:.2:1])

% linear regression on the median accuracy each day. 
dayss=1:length(acc_med);
x=dayss;y=acc_med;
lm = fitlm(x,y)
bhat = lm.Coefficients.Estimate;
x=[ones(length(acc_med),1) dayss(:)];
y = acc_med(:);
figure;plot(x(:,2),y,'.k','MarkerSize',25)
hold on
yhat = x*bhat;
plot(x(:,2),yhat,'k','LineWidth',1)
ylim([0 1])
set(gcf,'Color','w')
xticks(1:length(acc_med))
yticks([0:.2:1])
ylabel('Median Decoding Accuracy')
xlabel('PnP Days')
xticklabels(days)
xlim([0.5 length(acc_med)+0.5])
box off



% exponential fit 
y=acc_ind_days(1:end);
x=days_track(1:end);
t=x;
figure;plot(t,y,'ok','MarkerSize',20);hold on
f=fit(t(:),y(:),'exp1',Algorithm='Levenberg-Marquardt');
a=f.a;
b=f.b;
%c=f.c;
%d=f.d;
%yhat = a*exp(b*t) + c*exp(d*t);
yhat = a*exp(b*t) ;
plot(t,yhat,'k','LineWidth',1)
tau = -(1/b);
time_to_50per = -log(0.5)*tau;
tt=1:100;
yhat = a*exp(b*tt) ;
plot(tt,yhat,'r','LineWidth',1)
vline(round(time_to_50per))


figure;hold on
t2t=[];
t2th=[];
for i=1:7%length(time2target_days)
    tmp  = time2target_days{i};
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    t2t(i) = median(tmp);
    t2th = [t2th;tmp];
end
plot(t2t,'k','LineWidth',2)
ylim([0 3])
xticks(1:length(acc))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Mean time to Target (s)')
set(gca,'LineWidth',1)
xlim([0.5 7.5])
yticks([0:.5:3])

figure;hist(acch,10)
xlim([0 1])
vline(1/7,'r')
box off
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Decoder Accuracy')
ylabel('Count')
set(gca,'LineWidth',1)
%vline(median(acch),'k')

figure;hist(t2th,10)
xlim([0 2.25])
box off
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Time to Target')
ylabel('Count')
set(gca,'LineWidth',1)


% overall accuracy
overall_trial_accuracy_bkup=overall_trial_accuracy;
for i=1:length(overall_trial_accuracy)
    overall_trial_accuracy(i,:) = overall_trial_accuracy(i,:)./sum(overall_trial_accuracy(i,:));
end



%save bit_rate_discrete_PnP_v2 -v7.3


%
% figure;hist(time2target_days);
% figure;hist(acc_days);


%
% figure;boxplot(acc_days,'notch','off')
% figure;
% idx=ones(size(acc_days)) + 0.1*randn(size(acc_days));
% scatter(idx,acc_days,'k')


%% BLOCK BY BLOCK BIT RATE CALCULATIONS ACROSS DAYS - PNP 2 B1 
%%%%(MAIN)
% second plug and play experiment 


clc;clear

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath('C:\Users\nikic\Documents\MATLAB')

%foldernames = {'20210903'};
foldernames = {'20211013','20211015','20211020','20211022','20211027','20211029',...
    '20211103','20211105','20211117','20211119'};

% days -> 1 3 8 10 15 17 22 24 36 38

cd(root_path)
folders={};
br_across_days={};
time2target_days=[];
acc_days=[];
overall_trial_accuracy=zeros(7);
for i=1:length(foldernames)
    
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);

    if i==1
        idx=[1 2 3 4 7 8];
        D = D(idx);
    end

    if i==2
        idx=[1 2 4:length(D)];
        D = D(idx);
    end

    if i==6
        idx=[1 2 3 4 7];
        D = D(idx);
    end

    if i==7
        idx=[1 2 5 8 ];
        D = D(idx);
    end

    br=[];acc=[];time2target=[];
    for j=3:length(D)
        files=[];
        filepath=fullfile(folderpath,D((j)).name,'BCI_Fixed');
        if exist(filepath)
            filepath
            files = [files;findfiles('mat',filepath)'];
            folders=[folders;filepath];
        end
        if length(files)>0
            [b,a,t,T,ov_acc] = compute_bitrate(files,7);
            %[b,a,t] = compute_bitrate_constTime(files,7);
            br = [br b];
            acc = [acc mean(a)];
            time2target = [time2target; mean(t)];
            %[br, acc ,t] = [br compute_bitrate(files)];
            overall_trial_accuracy = overall_trial_accuracy + ov_acc;
        end
    end
    close all
    br_across_days{i}=br;
    time2target_days{i} = time2target(:);
    acc_days{i} = acc(:);
    %time2target_days = [time2target_days ;time2target(:)];
    %acc_days = [acc_days ;acc(:)];
end

% plotting as scatter plot
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95');
figure;hold on
br=[];
brh=[];
cmap = turbo(10);%turbo(length(br_across_days));
for i=1:10%length(br_across_days)
    tmp = br_across_days{i};
    brh =[brh tmp];
    %tmp=tmp(tmp>0.5);
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    br(i) = median(tmp);
end
plot(br(1:end),'k','LineWidth',2)
days={'1', '3', '8' ,'10' ,'15', '17', '22', '24', '36', '38'};
%days={'1','5','12','14','19','21','28','32','35','40','42'};
xticks(1:length(br))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('BitRate')
set(gca,'LineWidth',1)
%set(gca,'Color',[.85 .85 .85])
xlim([0 10.5])
ylim([0 3.5])

figure
boxplot(brh,'whisker',1.75)
set(gcf,'Color','w')
xticks(1)
xticklabels('PnP Experiment 2')
ylabel('Effective bit rate')
set(gca,'FontSize',12)
box off
xlim([.75 1.25])
ylim([0 3.5])
yticks([0:.5:3.5])
set(gca,'LineWidth',1,'TickLength',[0.025 0.025]);

% accuracy
figure;hold on
acc=[];
acch=[];
idxx=[];
acc_med=[];
for i=1:10%length(acc_days)
    tmp  = acc_days{i};
    acc_med(i)=median(tmp);
    %tmp=tmp(tmp>0.6);
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    acc(i) = median(tmp);
    acch = [acch ;tmp];
    idxx=[idxx;idx];
end
plot(acc,'k','LineWidth',2)
ylim([0 1])
xticks(1:length(acc))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Decoder Accuracy')
set(gca,'LineWidth',1)
xlim([0.5 10.5])
h=hline(1/7);
set(h,'LineWidth',2)
yticks([0:.2:1])

% linear regression on the median accuracy each day. 
dayss=1:length(acc_med);
x=dayss;y=acc_med;
lm = fitlm(x,y)
bhat = lm.Coefficients.Estimate;
x=[ones(length(acc_med),1) dayss(:)];
y = acc_med(:);
figure;plot(x(:,2),y,'.k','MarkerSize',25)
hold on
yhat = x*bhat;
plot(x(:,2),yhat,'k','LineWidth',1)
ylim([0 1])
set(gcf,'Color','w')
xticks(1:length(acc_med))
yticks([0:.2:1])
ylabel('Median Decoding Accuracy')
xlabel('PnP Days')
xticklabels(days)
xlim([0.5 length(acc_med)+0.5])
box off

figure;hold on
t2t=[];
t2th=[];
idxx=[];
for i=1:10%length(time2target_days)
    tmp  = time2target_days{i};
    %tmp=tmp(tmp<1.2);
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    t2t(i) = median(tmp);
    t2th = [t2th;tmp];
    idxx=[idxx;idx];
end
plot(t2t,'k','LineWidth',2)
ylim([0 3])
xticks(1:length(acc))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Time to Target (s)')
set(gca,'LineWidth',1)
xlim([0.5 10.5])

figure;hist(acch,10)
xlim([0 1])
vline(1/7,'r')
box off
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Decoder Accuracy')
ylabel('Count')
set(gca,'LineWidth',1)
%vline(median(acch),'k')

figure;hist(t2th,10)
xlim([0 2.25])
box off
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Time to Target')
ylabel('Count')
set(gca,'LineWidth',1)


% overall accuracy
overall_trial_accuracy_bkup=overall_trial_accuracy;
for i=1:length(overall_trial_accuracy)
    overall_trial_accuracy(i,:) = overall_trial_accuracy(i,:)./sum(overall_trial_accuracy(i,:));
end


%save bit_rate_discrete_PnP2_v2 -v7.3


%% (MAIN)  B1 plotting combined time to target and accuracy across both PnP experiments

clear;clc
close all
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

%a = load('bit_rate_discrete_PnP.mat');
%b = load('bit_rate_discrete_PnP2.mat');

a = load('bit_rate_discrete_PnP_v2.mat'); % fixed a bug
b = load('bit_rate_discrete_PnP2_v2.mat'); % fixed a bug


acch =[a.acch;b.acch];
figure;hist(acch,10)
xlim([0 1])

t2th =[a.t2th;b.t2th];
figure;hist(t2th,10)
xlim([0 2.5])

br =[a.br';b.br'];
figure;hist(br,10)

tmp = [(a.br) (b.br)];


br_overall=[];
br_across_days = a.br_across_days;
for i=1:7
    br_overall = [br_overall br_across_days{i}];        
end
br_across_days = b.br_across_days;
for i=1:10
    br_overall = [br_overall br_across_days{i}];        
end

% median bit rate
m = median(br_overall);
mb = sort(bootstrp(1000,@median,br_overall));
[mb(25) m mb(975)]

% mean bit rate
m = mean(br_overall);
mb = sort(bootstrp(1000,@mean,br_overall));
[mb(25) m mb(975)]


acc = median(a.acch);
accb = sort(bootstrp(1000,@median,a.acch));
[accb(25) acc accb(975)]


% mean accuracies
acc = mean(a.acch);
accb = sort(bootstrp(1000,@mean,a.acch));
[accb(25) acc accb(975)]

acc = mean(b.acch);
accb = sort(bootstrp(1000,@mean,b.acch));
[accb(25) acc accb(975)]

% mean time to target 
acc = mean(a.t2th);
accb = sort(bootstrp(1000,@mean,a.t2th));
[accb(25) acc accb(975)]

acc = mean(b.t2th);
accb = sort(bootstrp(1000,@mean,b.t2th));
[accb(25) acc accb(975)]


ov_acc = a.overall_trial_accuracy
ov_acc = (a.overall_trial_accuracy + b.overall_trial_accuracy)/2;
mean(diag(ov_acc))

% permutation test on bit rates
br1=a.brh;
br2=b.brh;

[p,h,stats]=ranksum(br1,br2)




%% (MAIN) block by block bit rate computation for B3

clc;clear



root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath 'C:\Users\nikic\Documents\MATLAB'

% take a call to whether include 20231120 -> it didnt have norm 1 durng
% online control 
foldernames = {'20231120','20231122','20231127','20231129','20231201',...
    '20231207','20231210','20231213','20231215','20231218','20231220',...
    '20231228','20231229','20240104','20240110'};


cd(root_path)
folders={};
br_across_days={};
time2target_days=[];
acc_days=[];
conf_matrix_overall=[];
overall_trial_accuracy=zeros(7);
for i=1:length(foldernames)

    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    br=[];acc=[];time2target=[];
    for j=3:length(D)
        files=[];
        filepath=fullfile(folderpath,D((j)).name,'BCI_Fixed');
        if exist(filepath)
            filepath
            files = [files;findfiles('mat',filepath)'];
            folders=[folders;filepath];
        end

        % removing bad trials
        if i==2 && j==4 %20231122, 144831
            good_files = ones(length(files),1);
            good_files([1])=0;
            files=files(logical(good_files));
        end

        if i==2 && j==8 %20231122, 153225
            good_files = ones(length(files),1);
            good_files([15])=0;
            files=files(logical(good_files));
        end


        if i==4 && j==3
            good_files = ones(length(files),1);
            good_files([1])=0;
            files=files(logical(good_files));
        end


        if i==5 && j==6
            good_files = ones(length(files),1);
            good_files([12 13])=0;
            files=files(logical(good_files));
        end

        if i==8 && j==5
            files=[];
        end

        if i==9 && j~=3
            files=[];
        end

        if i==10 && j<5
            files=[];
        end

        if i==11 && j==3 %20231220
            good_files = ones(length(files),1);
            good_files([12 ])=0;
            files=files(logical(good_files));
        end

        if i==12 && j<5 %20231228
            files=[];
        end

        if i==13 && (j==4 || j==5) %20231229
            files=[];
        end

        if i==13 && (j==6) %20231229
            good_files = ones(length(files),1);
            good_files([6])=0;
            files=files(logical(good_files));
        end

        if i==15 && j>6
            good_files = ones(length(files),1);
            good_files([7:end])=0;
            files=files(logical(good_files));
        end

        if length(files)>0
            [b,a,t,T,ov_acc] = compute_bitrate(files,7); % just compute as is
            %[b,a,t,T] = compute_bitrate_constTime(files,7); %overall time
            %[b,a,t,T] = compute_bitrate_badCh(files,7,net); %remove bad channels and simulate 
            conf_matrix_overall = cat(3,conf_matrix_overall,T);
            br = [br b];
            acc = [acc mean(a)];
            time2target = [time2target; mean(t)];
            overall_trial_accuracy = overall_trial_accuracy + ov_acc;
            %[br, acc ,t] = [br compute_bitrate(files)];
        end
    end
    close all
    br_across_days{i}=br;
    time2target_days{i} = time2target(:);
    acc_days{i} = acc(:);
    %time2target_days = [time2target_days ;time2target(:)];
    %acc_days = [acc_days ;acc(:)];
end

%plot results as a scatter plot
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95');
figure;hold on
br=[];
brh=[];
cmap = turbo(length(br_across_days));
for i=1:length(br_across_days)
    tmp = br_across_days{i};
    brh = [brh tmp];
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    br(i) = median(tmp);
end
plot(br(1:end),'k','LineWidth',2)
days={'1','3','8','10','12','18','21','24','26','29','31','39','40','46','52'};
xticks(1:length(br))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('BitRate')
set(gca,'LineWidth',1)
%set(gca,'Color',[.85 .85 .85])
xlim([0 length(br_across_days)+0.5])
ylim([0 4])
yticks([0:.5:4])
%

% plotting noise free days;
good_days=[1:6 8  10 12 ];
figure;hold on
br=[];
brh=[];
cmap = turbo(length(br_across_days));
for i=1:length(good_days)
    tmp = br_across_days{good_days(i)};
    brh = [brh tmp];
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    br(i) = median(tmp);
end
plot(br(1:end),'k','LineWidth',2)
days={'1','3','8','10','12','18','21','24','26','29','31','39','40','46'};
xticks(1:length(br))
set(gca,'XTickLabel',days(good_days))
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('BitRate')
set(gca,'LineWidth',1)
%set(gca,'Color',[.85 .85 .85])
xlim([0 length(good_days)+0.5])
ylim([0 4])
yticks([0:.5:4])
title('B3 Bit rate PnP')


figure
boxplot(brh,'whisker',1.75)
set(gcf,'Color','w')
xticks(1)
xticklabels('PnP Experiment')
ylabel('Effective bit rate')
set(gca,'FontSize',12)
box off
xlim([.75 1.25])
ylim([0 4])
yticks([0:.5:4])
set(gca,'LineWidth',1,'TickLength',[0.025 0.025]);
ylim([0 3.6])

% bit rate stats
br_mean = mean(brh);
brb = sort(bootstrp(1000,@mean,brh));
[brb(25) br_mean brb(975)]

% plotting decoder acc across days
figure;hold on
acc=[];
acch=[];
acc_good_days = acc_days(good_days);
cmap = turbo(length(acc_good_days));
acc_med=[];
for i=1:length(acc_good_days)
    tmp  = acc_good_days{i};
    acc_med(i)=median(tmp);
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    acc(i) = median(tmp);
    acch = [acch ;tmp];
end
plot(acc,'k','LineWidth',2)
ylim([0 1])
xticks(1:length(acc))
set(gca,'XTickLabel',days(good_days))
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Decoder Accuracy')
set(gca,'LineWidth',1)
xlim([0 length(acc_good_days)+0.5])
h=hline(1/7);
set(h,'LineWidth',2)
yticks([0:.2:1])


% linear regression on the median accuracy each day. 
dayss=1:length(acc_med);
x=dayss;y=acc_med;
lm = fitlm(x,y);
bhat = lm.Coefficients.Estimate;
x=[ones(length(acc_med),1) dayss(:)];
y = acc_med(:);
figure;plot(x(:,2),y,'.k','MarkerSize',25)
hold on
yhat = x*bhat;
plot(x(:,2),yhat,'k','LineWidth',1)
ylim([0 1])
set(gcf,'Color','w')
xticks(1:length(acc_med))
yticks([0:.2:1])
ylabel('Median Decoding Accuracy')
xlabel('PnP Days')
xticklabels(days(good_days))
xlim([0.5 length(acc_med)+0.5])
box off

% acc stats
acc_mean = mean(acch);
accb = sort(bootstrp(1000,@mean,acch));
[accb(25) acc_mean accb(975)]


figure;hold on
t2t=[];
t2th=[];
time2target_good_days = time2target_days(good_days);
for i=1:length(time2target_good_days)
    tmp  = time2target_good_days{i};
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    t2t(i) = median(tmp);
    t2th = [t2th;tmp];
end
plot(t2t,'k','LineWidth',2)
ylim([0 3])
xticks(1:length(t2t))
set(gca,'XTickLabel',days(good_days))
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Mean time to Target (s)')
set(gca,'LineWidth',1)
xlim([0 length(time2target_good_days)+0.5])
yticks([0:.5:3])


% time to target  stats
t2t_mean = mean(t2th);
t2tb = sort(bootstrp(1000,@mean,t2th));
[t2tb(25) t2t_mean t2tb(975)]

figure;hist(acch,10)
xlim([0 1])
vline(1/7,'r')
box off
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Decoder Accuracy')
ylabel('Count')
set(gca,'LineWidth',1)
%vline(median(acch),'k')

figure;hist(t2th,10)
xlim([0 2.25])
box off
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Time to Target')
ylabel('Count')
set(gca,'LineWidth',1)


% overall accuracy
overall_trial_accuracy_bkup=overall_trial_accuracy;
for i=1:length(overall_trial_accuracy)
    overall_trial_accuracy(i,:) = overall_trial_accuracy(i,:)./sum(overall_trial_accuracy(i,:));
end

