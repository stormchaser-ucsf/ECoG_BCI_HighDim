


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

% day 11
session_data(10).Day = '20230714';
session_data(10).folders={'104450','105120','105739','110205',...
    '111105','111516','112121','112440','113114','113432',...
    '114241','114608'};
session_data(10).folder_type={'I','I','I','I',...
    'O','O','O','O','O','O',...
    'B','B'};
session_data(10).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am'};

% day 12
session_data(10).Day = '20230714';
session_data(10).folders={'104450','105120','105739','110205',...
    '111105','111516','112121','112440','113114','113432',...
    '114241','114608'};
session_data(10).folder_type={'I','I','I','I',...
    'O','O','O','O','O','O',...
    'B','B'};
session_data(10).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am'};


% day 13
session_data(10).Day = '20230714';
session_data(10).folders={'104450','105120','105739','110205',...
    '111105','111516','112121','112440','113114','113432',...
    '114241','114608'};
session_data(10).folder_type={'I','I','I','I',...
    'O','O','O','O','O','O',...
    'B','B'};
session_data(10).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am'};


% day 14
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

%% GET SIG CHANNELS FOR EACH MOVEMENT ACROSS DAYS

%% SIG CH VIA ERP FOR EACH DAY AND OVER ALL MOVEMENTS FOR OL,CL1,CL2


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data_9DoF

% init variables
D1_imag_days=[];
D2_imag_days=[];
D3_imag_days=[];
D4_imag_days=[];
D5_imag_days=[];
D6_imag_days=[];
D7_imag_days=[];
D8_imag_days=[];
D9_imag_days=[];
D1_CL1_days=[];
D2_CL1_days=[];
D3_CL1_days=[];
D4_CL1_days=[];
D5_CL1_days=[];
D6_CL1_days=[];
D7_CL1_days=[];
D8_CL1_days=[];
D9_CL1_days=[];
D1_CL2_days=[];
D2_CL2_days=[];
D3_CL2_days=[];
D4_CL2_days=[];
D5_CL2_days=[];
D6_CL2_days=[];
D7_CL2_days=[];

% loop over days
for i=1:length(session_data)


    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_online = logical((strcmp(session_data(i).folder_type,'B')) + (strcmp(session_data(i).folder_type,'O')));

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    %%%%%%imagined data ERPs
    disp(['Processing Day ' num2str(i) ' Imagined Files '])
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    [D1,D2,D3,D4,D5,D6,D7,D8,D9,tim] = load_erp_data_7DoF(files);

    % run the ERPs and get the significant channels
    load(files{1})
    pmask1 = sig_ch_erps(D1,TrialData,tim);
    pmask2 = sig_ch_erps(D2,TrialData,tim);
    pmask3 = sig_ch_erps(D3,TrialData,tim);
    pmask4 = sig_ch_erps(D4,TrialData,tim);
    pmask5 = sig_ch_erps(D5,TrialData,tim);
    pmask6 = sig_ch_erps(D6,TrialData,tim);
    pmask7 = sig_ch_erps(D7,TrialData,tim);
    pmask8 = sig_ch_erps(D8,TrialData,tim);
    pmask9 = sig_ch_erps(D9,TrialData,tim);
    D1_imag_days=cat(3,D1_imag_days,pmask1);
    D2_imag_days=cat(3,D2_imag_days,pmask2);
    D3_imag_days=cat(3,D3_imag_days,pmask3);
    D4_imag_days=cat(3,D4_imag_days,pmask4);
    D5_imag_days=cat(3,D5_imag_days,pmask5);
    D6_imag_days=cat(3,D6_imag_days,pmask6);
    D7_imag_days=cat(3,D7_imag_days,pmask7);
    D8_imag_days=cat(3,D8_imag_days,pmask8);
    D9_imag_days=cat(3,D9_imag_days,pmask9);


    %%%%%% CL1 data ERPs
    disp(['Processing Day ' num2str(i) ' CL1 Files '])
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    if length(files)>0
        [D1,D2,D3,D4,D5,D6,D7,D8,D9,tim] = load_erp_data_7DoF(files);


        % run the ERPs and get the significant channels
        pmask1 = sig_ch_erps(D1,TrialData,tim);
        pmask2 = sig_ch_erps(D2,TrialData,tim);
        pmask3 = sig_ch_erps(D3,TrialData,tim);
        pmask4 = sig_ch_erps(D4,TrialData,tim);
        pmask5 = sig_ch_erps(D5,TrialData,tim);
        pmask6 = sig_ch_erps(D6,TrialData,tim);
        pmask7 = sig_ch_erps(D7,TrialData,tim);
        pmask8 = sig_ch_erps(D8,TrialData,tim);
        pmask9 = sig_ch_erps(D9,TrialData,tim);
        D1_CL1_days=cat(3,D1_CL1_days,pmask1);
        D2_CL1_days=cat(3,D2_CL1_days,pmask2);
        D3_CL1_days=cat(3,D3_CL1_days,pmask3);
        D4_CL1_days=cat(3,D4_CL1_days,pmask4);
        D5_CL1_days=cat(3,D5_CL1_days,pmask5);
        D6_CL1_days=cat(3,D6_CL1_days,pmask6);
        D7_CL1_days=cat(3,D7_CL1_days,pmask7);
        D8_CL1_days=cat(3,D8_CL1_days,pmask8);
        D9_CL1_days=cat(3,D9_CL1_days,pmask9);

    end

    %
    %     %%%%%% CL2 data ERPs
    %     disp(['Processing Day ' num2str(i) ' CL2 Files '])
    %     folders = session_data(i).folders(batch_idx);
    %     day_date = session_data(i).Day;
    %     files=[];
    %     for ii=1:length(folders)
    %         folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
    %         files = [files;findfiles('',folderpath)'];
    %     end
    %
    %     %load the data
    %     [D1,D2,D3,D4,D5,D6,D7,D8,D9,tim] = load_erp_data_7DoF(files);
    %
    %     % run the ERPs and get the significant channels
    %     pmask1 = sig_ch_erps(D1,TrialData,tim);
    %     pmask2 = sig_ch_erps(D2,TrialData,tim);
    %     pmask3 = sig_ch_erps(D3,TrialData,tim);
    %     pmask4 = sig_ch_erps(D4,TrialData,tim);
    %     pmask5 = sig_ch_erps(D5,TrialData,tim);
    %     pmask6 = sig_ch_erps(D6,TrialData,tim);
    %     pmask7 = sig_ch_erps(D7,TrialData,tim);
    %     D1_CL2_days=cat(3,D1_CL2_days,pmask1);
    %     D2_CL2_days=cat(3,D2_CL2_days,pmask2);
    %     D3_CL2_days=cat(3,D3_CL2_days,pmask3);
    %     D4_CL2_days=cat(3,D4_CL2_days,pmask4);
    %     D5_CL2_days=cat(3,D5_CL2_days,pmask5);
    %     D6_CL2_days=cat(3,D6_CL2_days,pmask6);
    %     D7_CL2_days=cat(3,D7_CL2_days,pmask7);
end

save sig_ch_ERPs_B1_9DoF_July2023_hG -v7.3


% plotting the stats across days
corr_val=[];
mvmt={'Rt Thumb','Lt Leg','Lt Thumb','Rt Bicep','Tong','Lips','Both middle',...
    'Rot Rt Wrist','Rot Lt Wrist'};
for i=1:9 % plot the imag maps along with the correlation across days
    varname = genvarname(['D' num2str(i) '_imag_days']);
    a1 = squeeze(sum(eval(varname),3));
    varname = genvarname(['D' num2str(i) '_CL1_days']);
    a2 = squeeze(sum(eval(varname),3));
    %     varname = genvarname(['D' num2str(i) '_CL2_days']);
    %     a3 = squeeze(sum(eval(varname),3));
    figure;
    %     subplot(1,2,1)
    %     imagesc(a1)
    %colormap bone
    %axis off
    %box on
    %clim([0 6])
    %subplot(1,2,2)
    imagesc(a2)
    %colormap bone
    axis off
    box on
    clim([0 5])
    sgtitle(mvmt{i})
    set(gcf,'Color','w')

    % correlation
    %     corr_val(i,:) = [corr(a1(:),a2(:),'Type','Pearson') ...
    %         corr(a1(:),a3(:),'Type','Pearson'),...
    %         corr(a2(:),a3(:),'Type','Pearson')];

    % distance
    %     D1 = pdist([a1(:)';a2(:)'],'cosine');
    %     D2 = pdist([a1(:)';a3(:)'],'cosine');
    %     D3 = pdist([a2(:)';a3(:)'],'cosine');
    %     corr_val(i,:) = [D1 D2 D3];
end
mean(corr_val(:))


%% LOOKING AT THE PERFORMANCE OF BUILDING A LSTM ACROSS DAYS for 9DoF

clc;clear
close all

% get all the data , build LSTM and see how well it does on computing
% accuracy on held out days during fixed online control

clc;clear

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd(root_path)
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
load session_data_9DoF

% for only 6 DoF original:
%foldernames = {'20210526','20210528','20210602','20210609_pm','20210611'};

foldernames = {'20230609','20230616','20230621','20230623','20230628','20230630',...
    '20230705','20230707','20230712','20230714','20230719','20230721',...
    '20230726','20230728','20230809'};%


datafiles_days={};
k=1;jj=1;
for i=1:length(foldernames)
    disp([i/length(foldernames)]);
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    imag_files_temp=[];
    online_files_temp=[];
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        if exist(filepath)
            imag_files_temp = [imag_files_temp;findfiles('mat',filepath)'];
        end
        filepath1=fullfile(folderpath,D(j).name,'BCI_Fixed');
        if exist(filepath1)
            online_files_temp = [online_files_temp;findfiles('mat',filepath1)'];
        end
    end
    %     if ~isempty(imag_files_temp)
    %         imag_files{k} = imag_files_temp;k=k+1;
    %     end
    %     if ~isempty(online_files_temp)
    %         online_files{jj} = online_files_temp;jj=jj+1;
    %     end
    datafiles_days(i).date =  foldernames{i};
    datafiles_days(i).imagined =  imag_files_temp;
    datafiles_days(i).online =  online_files_temp;

    %     imag_files{i} = imag_files_temp;
    %     online_files{i} = online_files_temp;
end


% PARAMS
% filter bank hg
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [70,77];   % high gamma1
Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
Params.FilterBank(end+1).fpass = [30,36]; % lg1
Params.FilterBank(end+1).fpass = [36,42]; % lg2
Params.FilterBank(end+1).fpass = [42,50]; % lg3
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end
% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

% have to cycle through days 3 to 14 in terms of testing out model across
% days
testing_days=[3:15];
idx = 1:length(datafiles_days);
acc_sample_days_lstm=[];
acc_trial_days_lstm=[];
acc_sample_days_mlp=[];
acc_trial_days_mlp=[];
for i=1:length(testing_days)
    test_days = testing_days(i);
    train_days = ones(size(idx));
    train_days(test_days)=0;
    train_days = find(train_days>0);

    % now get all the neural features to build the LSTM
    files=[];
    for j=1:length(train_days)

        % get all the imagined and online files
        files=[files;datafiles_days(train_days(j)).imagined];
        files=[files;datafiles_days(train_days(j)).online];
    end       

    % get training and testing samples
    clear XTrain XTest YTrain YTest
    %[XTrain,XTest,YTrain,YTest] = get_lstm_features_9DoF(files,Params,lpFilt);
    [XTrain,XTest,YTrain,YTest] = get_lstm_features_9DoF_reduced(files,Params,lpFilt);

    % optional -> reduce the channel dimensions
    load ch_per_bands
    ch_band = ch_band{5};% Delta, Theta, Beta, LG, and HG
    sig_ch_delta = ch_band{1};
    sig_ch_hg = ch_band{5};

    for ii=1:length(XTrain)
        tmp=XTrain{ii};
        tmp_hg = tmp(1:128,:);
        tmp_hg = tmp_hg(sig_ch_hg,:);
        tmp_lfo = tmp(129:256,:);
        tmp_lfo = tmp_lfo(sig_ch_delta,:);
        tmp =[tmp_hg;tmp_lfo];
        XTrain{ii}=tmp;
    end

    for ii=1:length(XTest)
        tmp=XTest{ii};
        tmp_hg = tmp(1:128,:);
        tmp_hg = tmp_hg(sig_ch_hg,:);
        tmp_lfo = tmp(129:256,:);
        tmp_lfo = tmp_lfo(sig_ch_delta,:);
        tmp =[tmp_hg;tmp_lfo];
        XTest{ii}=tmp;
    end
    
    
    % train the LSTM
    net_bilstm = train_lstm(XTrain,XTest,YTrain,YTest,64,0.3,5);

    % test out on the held-out day
    files_test = datafiles_days(test_days).online;
    [acc_lstm_sample,acc_mlp_sample,acc_lstm_trial,acc_mlp_trial]...
        = get_lstm_performance_9DoF_reduced(files_test,net_bilstm,Params,lpFilt,9);
    acc_sample_days_lstm(i,:,:) = acc_lstm_sample;
    acc_sample_days_mlp(i,:,:) = acc_mlp_sample;
    acc_trial_days_lstm(i,:,:) = acc_lstm_trial;
    acc_trial_days_mlp(i,:,:) = acc_mlp_trial;
end

save 9DoF_LSTM_vs_MLP ...
     acc_sample_days_lstm acc_sample_days_mlp acc_trial_days_lstm ...
     acc_trial_days_mlp  -v7.3



acc_lstm_sample = squeeze(mean(acc_trial_days_lstm(10:12,:,:),1));
acc_mlp_sample = squeeze(mean(acc_trial_days_mlp(10:12,:,:),1));


% plotting the success of individual actions
tmp = [diag(acc_lstm_sample) diag(acc_mlp_sample)];
figure;
hold on
for i=1:9
    idx = i:9:size(tmp,1);
    decodes = tmp(idx,:);
    disp(decodes)
    h=bar(2*i-0.25,mean(decodes(:,1)));
    h1=bar(2*i+0.25,mean(decodes(:,2)));
    h.BarWidth=0.4;
    h.FaceColor=[0.2 0.2 0.7];
    h1.BarWidth=0.4;
    h1.FaceColor=[0.7 0.2 0.2];
    h.FaceAlpha=0.85;
    h1.FaceAlpha=0.85;

    %     s=scatter(ones(3,1)*2*i-0.25+0.05*randn(3,1),decodes(:,1),'LineWidth',2);
    %     s.CData = [0.2 0.2 0.7];
    %     s.SizeData=50;
    %
    %     s=scatter(ones(3,1)*2*i+0.25+0.05*randn(3,1),decodes(:,2),'LineWidth',2);
    %     s.CData = [0.7 0.2 0.2];
    %     s.SizeData=50;
end
xticks([2:2:18])
xticklabels({'Right Thumb','Left Leg','Left Thumb','Rt. Bicep','Lips','Tongue','Both Middle',...
    'Rot. Rt Wrist','Rot. Lt Wrist'})
ylabel('Decoding Accuracy')
legend('LSTM','MLP')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
title(datafiles_days(test_days).date)

figure;imagesc(acc_lstm_sample)
colormap bone
title(['PnP LSTM simulated acc - ' num2str(mean(diag(acc_lstm_sample))*100)])
xticks(1:9)
xticklabels({'Right Thumb','Left Leg','Left Thumb','Rt. Bicep','Lips','Tongue','Both Middle',...
    'Rot. Rt Wrist','Rot. Lt Wrist'})
yticks(1:9)
yticklabels({'Right Thumb','Left Leg','Left Thumb','Rt. Bicep','Lips','Tongue','Both Middle',...
    'Rot. Rt Wrist','Rot. Lt Wrist'})
set(gcf,'Color','w')
set(gca,'FontSize',12)


figure;imagesc(acc_mlp_sample)
colormap bone
title(['MLP acc - ' num2str(mean(diag(acc_mlp_sample))*100)])
xticks(1:9)
xticklabels({'Right Thumb','Left Leg','Left Thumb','Rt. Bicep','Lips','Tongue','Both Middle',...
    'Rot. Rt Wrist','Rot. Lt Wrist'})
yticks(1:9)
yticklabels({'Right Thumb','Left Leg','Left Thumb','Rt. Bicep','Lips','Tongue','Both Middle',...
    'Rot. Rt Wrist','Rot. Lt Wrist'})
set(gcf,'Color','w')
set(gca,'FontSize',12)











