


clc;clear
addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20230623'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        %folderpath,D(j).name
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
time_to_target=zeros(2,9);
for i=1:length(files)
    disp(i/length(files)*100)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    features = cell2mat(features);
    features = features(769:end,:); %hG
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
    %     if length(state1)<8
    %         data  =[data(:,1) data];
    %     end

    % store the time to target data
    time_to_target(2,TrialData.TargetID) = time_to_target(2,TrialData.TargetID)+1;
    if trial_dur<=3
        time_to_target(1,TrialData.TargetID) = time_to_target(1,TrialData.TargetID)+1;
    end

    % now get the ERPs
    % if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=3
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
    end
    %  end
end


time_to_target(1,:)./time_to_target(2,:)

%%
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
    erps =  squeeze(D9(i,:,:)); % change this to the action to generate ERPs

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
    idx=10:20;
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
    ylim([-2 4])
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
        xticks([tim])
        xticklabels({'S1','S2','S3','S4'})
        yticks([-2:2:4])
        yticklabels({'-2','0','2','4'})
    end

    if suc==1
        box on
        set(gca,'LineWidth',2)
        set(gca,'XColor','g')
        set(gca,'YColor','g')
    end
    d = d+1;
end
sgtitle('Rot Lt. Wrist hG')



%% SESSION DATA

clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
%addpath('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\clicker\7DoF_Classifiers')
cd(root_path)

%day1
session_data(1).Day = '20230609';
session_data(1).folders = {'110911','111556','112441','112857','113803','114227',...
    '114800','115215','120125','120532','121159','121629'};
session_data(1).folder_type={'I','I','I','I','I','I','I','I','I','I','I','I'};
session_data(1).AM_PM = {'am','am','am','am','am','am','am','am','am',...
    'am','am','am'};

%day2
session_data(2).Day = '20230616';
session_data(2).folders={'133831','135341','135808','140347','140759','141444',...
    '141905','142641','143054','144052','144510','145121','145744'};
session_data(2).folder_type={'I','I','I','I','I','I','I','I','I','I','I','I','I'};
session_data(2).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am','am'};


% day 3
session_data(3).Day = '20230621';
session_data(3).folders={'111028','111753','112205'...
    '113313','113754','114535','115004',...
    '120703'};
session_data(3).folder_type={'I','I','I',...
    'O','O','O','O',...
    'B'};
session_data(3).AM_PM = {'am','am','am','am','am','am','am','am'};



% day 4
session_data(4).Day = '20230623';
session_data(4).folders={'133536','134231','134714','135223','135537',...
    '140753','141534',...
    '142510','143323','144816','145204','145633',...
    '150511'};
session_data(4).folder_type={'I','I','I','I','I',...
    'O','O',...
    'B','B','B','B','B',...
    'B'};
session_data(4).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am','am'};



% day 5
session_data(5).Day = '20230628';
session_data(5).folders={'111112','112514','112936','113748',...
    '114746','115223','115900','120259',...
    '121032'};
session_data(5).folder_type={'I','I','I','I',...
    'O','O','O','O',...
    'B'};
session_data(5).AM_PM = {'am','am','am','am','am','am','am','am','am'};


% day 6
session_data(6).Day = '20230630';
session_data(6).folders={'104854','105526','110241','110728',...
    '111956','112514','112942',...
    '113902','114324','114747'};
session_data(6).folder_type={'I','I','I','I',...
    'O','O','O',...
    'B','B','B'};
session_data(6).AM_PM = {'am','am','am','am','am','am','am','am','am','am'};

% day 7
session_data(7).Day = '20230705';
session_data(7).folders={'105644','110316','114813','115543',...
    '120404','120812'};
session_data(7).folder_type={'I','I','I','I',...
    'O','O'};
session_data(7).AM_PM = {'am','am','am','am','am','am'};


% day 8
session_data(8).Day = '20230707';
session_data(8).folders={'104449','105154','110055','110513','111531','111942',...
    '112509','112922','113447','113843','114849'};
session_data(8).folder_type={'I','I','I','I',...
    'O','O','O','O','O','O',...
    'B'};
session_data(8).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am'};

% day 9
session_data(9).Day = '20230712';
session_data(9).folders={'104813','105438','110143','110604',...
    '111816','112156','112829','113160','113800','114125',...
    '115225','115658'};
session_data(9).folder_type={'I','I','I','I',...
    'O','O','O','O','O','O',...
    'B','B'};
session_data(9).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am'};

% day 10
session_data(10).Day = '20230714';
session_data(10).folders={'104450','105120','105739','110205',...
    '111105','111516','112121','112440','113114','113432',...
    '114241','114608'};
session_data(10).folder_type={'I','I','I','I',...
    'O','O','O','O','O','O',...
    'B','B'};
session_data(10).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am'};


save session_data_9DoF session_data -v7.3


%% (MAIN) looking at decoding performance from imagined -> online -> batch
% across days

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_9DoF
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=5;
plot_true=false;
for i=3:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_online = folders_online + folders_batch; % just doing all closed-loop folders at once

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    %batch_idx = find(folders_batch==1);
    %disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% cross_val classification accuracy for imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files);
    % save the data
    %     filename = ['condn_data_ImaginedTrials_Day' num2str(i)];
    %     save(filename, 'condn_data', '-v7.3')

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = accuracy_imagined_data_9DOF(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_imagined_days(:,i) = diag(acc_imagined);


    %%%%%% get classification accuracy for online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_online = accuracy_online_data_9DOF(files);
    if plot_true
        figure;imagesc(acc_online)
        colormap bone
        clim([0 0.8])
        set(gcf,'color','w')
        xticklabels({'Rt. Thumb','Left Leg','Left Thumb','Rt. Bicep',...
            'Lips','Tongue','Both middle','Rot. Rt Wrist','Rot Lt. Wrist'})
        yticklabels({'Rt. Thumb','Left Leg','Left Thumb','Rt. Bicep',...
            'Lips','Tongue','Both middle','Rot. Rt Wrist','Rot Lt. Wrist'})
    end
    acc_online_days(:,i) = diag(acc_online);


    %     %%%%%% cross_val classification accuracy for batch data
    %     folders = session_data(i).folders(batch_idx);
    %     day_date = session_data(i).Day;
    %     files=[];
    %     for ii=1:length(folders)
    %         folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
    %         %cd(folderpath)
    %         files = [files;findfiles('',folderpath)'];
    %     end
    %
    %     % get the classification accuracy
    %     acc_batch = accuracy_online_data(files);
    %     if plot_true
    %         figure;imagesc(acc_batch)
    %         colormap bone
    %         clim([0 1])
    %         set(gcf,'color','w')
    %     end
    %     acc_batch_days(:,i) = diag(acc_batch);
end

acc_imagined_days=acc_imagined_days(:,3:end);
acc_online_days=acc_online_days(:,3:end);



%acc_online_days = (acc_online_days + acc_batch_days)/2;
figure;
ylim([0.0 1])
xlim([0.5 8.5])
hold on
plot(mean(acc_imagined_days,1))
plot(mean(acc_online_days,1))


% linear model for time to see if improvement in decoding accuracy
days=1:10;
y=mean(acc_imagined_days,1)';
figure;hold on
plot(days,y,'.k','MarkerSize',20)
x = [ones(length(days),1) days'];
[B,BINT,R,RINT,STATS] = regress(y,x);
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% [bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp',1);
% yhat1 = x*bhat;
% plot(days,yhat1,'k','LineWidth',1)
xlim([.5 10.5])
xticks([1:10])
set(gcf,'Color','w')
yticks(0:.2:1)
ylim([0 1])
STATS(3)
lm = fitlm(x(:,2),y)


% as regression lines
figure;plot(mean(acc_imagined_days,1),'.','MarkerSize',20)

% stats
tmp = [median(acc_imagined_days,1)' median(acc_online_days,1)' ...
    median(acc_batch_days,1)'];

figure;boxplot(acc_imagined_days)
ylim([0.2 1])
xlim([0.5 10.5])
hold on
boxplot(acc_batch_days,'Colors','k')
a = get(get(gca,'children'),'children');

figure;
boxplot([acc_imagined_days(:) acc_online_days(:) acc_batch_days(:)])

m1 = (acc_imagined_days(:));
m1b = sort(bootstrp(1000,@mean,m1));
m11 = mean(acc_imagined_days,1);
m2 = (acc_online_days(:));
m2b = sort(bootstrp(1000,@mean,m2));
m22 = mean(acc_online_days,1);
m3 = (acc_batch_days(:));
m3b = sort(bootstrp(1000,@mean,m3));
m33 = mean(acc_batch_days,1);
x=1:3;
y=[mean(m1) mean(m2) mean(m3)];
neg = [y(1)-m1b(25) y(2)-m2b(25) y(3)-m3b(25)];
pos = [m1b(975)-y(1) m2b(975)-y(2) m3b(975)-y(3)];
figure;
hold on
cmap = brewermap(10,'Blues');
%cmap = (turbo(7));
for i=1:size(acc_imagined_days,2)
    plot(1+0.1*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
    plot(2+0.1*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
    plot(3+0.1*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
end
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',20,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([.5 3.5])
ylim([0.0 1])
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
yticks(0:.1:1)
set(gca,'FontSize',12)

tmp = [ m11' m22' m33'];
figure;boxplot(tmp)

% fit a general linear regression model across days
acc=[];
days_acc=[];
experiment=[];
tmp = mean(acc_imagined_days,1);
acc = [acc;tmp'];
experiment =[experiment;ones(length(tmp),1)];
tmp1 = mean(acc_online_days,1);
acc = [acc;tmp1'];
experiment =[experiment;2*ones(length(tmp),1)];
tmp2 = mean(acc_batch_days,1);
acc = [acc;tmp2'];
experiment =[experiment;3*ones(length(tmp),1)];

data = table(experiment,acc);
glm = fitglm(data,'acc ~ 1 + experiment');

% stats
[h p tb st] = ttest(tmp,tmp1)
[h p tb st] = ttest(tmp,tmp2)
[h p tb st] = ttest(tmp1,tmp2)

% using a GLM for each action across days
acc=[];
days_acc=[];
experiment=[];
for i=1:size(acc_imagined_days,2)
    a = acc_imagined_days(:,i);
    acc = [acc;a];
    days_acc = [days_acc;i*ones(size(a))];
    experiment = [experiment;1*ones(size(a))];
end
for i=1:size(acc_online_days,2)
    a = acc_online_days(:,i);
    acc = [acc;a];
    days_acc = [days_acc;i*ones(size(a))];
    experiment = [experiment;2*ones(size(a))];
end
for i=1:size(acc_batch_days,2)
    a = acc_batch_days(:,i);
    acc = [acc;a];
    days_acc = [days_acc;i*ones(size(a))];
    experiment = [experiment;3*ones(size(a))];
end

data = table(days_acc,experiment,acc);
glme = fitglme(data,'acc ~ 1 + experiment +(1|days_acc)')

%test of medians between cl1 and cl2
a0=data.acc(data.experiment==1);
a1=data.acc(data.experiment==2);
a2=data.acc(data.experiment==3);

a1 = (median(acc_online_days,1))';
a2 = (median(acc_batch_days,1))';
stat = median(a2)-median(a1);
boot=[];
a=[a1;a2];
for i=1:1000
    idx = randperm(length(a));
    a11 = a(idx(1:10));
    a22 = a(idx(11:end));
    boot(i) = median(a11) - median(a22);
end
figure;hist((boot))
vline(stat)
sum((boot) > stat)/length(boot)

% X = [ones(10,1) (1:10)'];
% Y =  mean(acc_batch_days,1)';
% [B,BINT,R,RINT,STATS] = regress(Y,X)

figure;
boxplot([a0 a1 a2])

% get the accuracies relative to imagined movement within that day
a0 = mean(acc_imagined_days,1);
a1 = mean(acc_online_days,1);
a2 = mean(acc_batch_days,1);
figure;
plot(a0);
hold on
plot(a1);
plot(a2)
ylim([0 1])

a1 = (a1-a0)./a0;
a2 = (a2-a0)./a0;
figure;boxplot([a1' a2'])
hline(0)

b1_acc_rel_imagined_prop = [a1' a2'];
save b1_acc_rel_imagined_prop b1_acc_rel_imagined_prop


%% LOOKING AT THE EFFECT OF POOLING ON ACCURACY ON IMAGINED/closed-loop DATA


clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_9DoF
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=5;
plot_true=false;
files=[];
for i=3:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_online = folders_online + folders_batch; % just doing all closed-loop folders at once

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    %batch_idx = find(folders_batch==1);
    %disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% cross_val classification accuracy for imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;

    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the closed-loop data as well
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;    
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
end

%load the data
condn_data = load_data_for_MLP_TrialLevel(files);

% get cross-val classification accuracy
iterations=10;
[acc_imagined,train_permutations] = accuracy_imagined_data_9DOF(condn_data, iterations);
acc_imagined_mean=squeeze(nanmean(acc_imagined,1));
% plot
figure;imagesc(acc_imagined_mean)
colormap bone
clim([0 0.8])
set(gcf,'color','w')
xticklabels({'Rt. Thumb','Left Leg','Left Thumb','Rt. Bicep',...
    'Lips','Tongue','Both middle','Rot. Rt Wrist','Rot Lt. Wrist'})
yticklabels({'Rt. Thumb','Left Leg','Left Thumb','Rt. Bicep',...
    'Lips','Tongue','Both middle','Rot. Rt Wrist','Rot Lt. Wrist'})

a1=[];a2=[];
for i=1:size(acc_imagined_pooling)
    a1 = [a1 mean(diag(squeeze(acc_imagined_pooling(i,:,:))))];
    a2 = [a2 mean(diag(squeeze(acc_imagined_no_pooling(i,:,:))))];
end
figure;boxplot([a1(:) a2(:)])
ylim([0.5 0.8])
xticks(1:2)
xticklabels({'Pooling','No pooling'})
set(gca,'FontSize',14)
set(gcf,'Color','w')
box off
ylabel('Acc')

%%%%%% get classification accuracy for online data
folders = session_data(i).folders(online_idx);
day_date = session_data(i).Day;
files=[];
for ii=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
    %cd(folderpath)
    files = [files;findfiles('',folderpath)'];
end

% get the classification accuracy
acc_online = accuracy_online_data_9DOF(files);
if plot_true
    figure;imagesc(acc_online)
    colormap bone
    clim([0 0.8])
    set(gcf,'color','w')
    xticklabels({'Rt. Thumb','Left Leg','Left Thumb','Rt. Bicep',...
        'Lips','Tongue','Both middle','Rot. Rt Wrist','Rot Lt. Wrist'})
    yticklabels({'Rt. Thumb','Left Leg','Left Thumb','Rt. Bicep',...
        'Lips','Tongue','Both middle','Rot. Rt Wrist','Rot Lt. Wrist'})
end
acc_online_days(:,i) = diag(acc_online);


%     %%%%%% cross_val classification accuracy for batch data
%     folders = session_data(i).folders(batch_idx);
%     day_date = session_data(i).Day;
%     files=[];
%     for ii=1:length(folders)
%         folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
%         %cd(folderpath)
%         files = [files;findfiles('',folderpath)'];
%     end
%
%     % get the classification accuracy
%     acc_batch = accuracy_online_data(files);
%     if plot_true
%         figure;imagesc(acc_batch)
%         colormap bone
%         clim([0 1])
%         set(gcf,'color','w')
%     end
%     acc_batch_days(:,i) = diag(acc_batch);





