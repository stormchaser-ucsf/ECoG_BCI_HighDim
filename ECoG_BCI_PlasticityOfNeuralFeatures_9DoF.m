%% SESSION DATA 9DOF

clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\clicker\7DoF_Classifiers')
cd(root_path)

%day1
session_data(1).Day = '20220202';
session_data(1).folders = {'102552','103014','103454','103943','105243','133904',...
    '134327','135301','135655','140143'};
session_data(1).folder_type={'I','I','I','I','O','I','I','B','B','B'};
session_data(1).AM_PM = {'am','am','am','am','am','pm','pm','pm','pm','pm'};

session_data(2).Day = '20220204';
session_data(2).folders = {'111057','112018','112544','114048',...
    '133312','134225','134655','135121',...
    '140159','140607','141121','141718',...
    '142808'};
session_data(2).folder_type={'I','I','I','O',...
    'I','I','I','I','O','O','O','O',...
    'B'};
session_data(2).AM_PM = {'am','am','am','am',...
    'pm','pm','pm','pm',...
    'pm','pm','pm','pm',...
    'pm'};

save session_data_9DOF session_data

%% SESSION DATA 9DOF B3

clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
cd(root_path)

%day1
session_data(1).Day = '20240117';
session_data(1).folders = {'142849','143310','143943','144438','144847',...
    '145900','150402','150721','151040',...
    '151633','151947','152259'};
session_data(1).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B'};
session_data(1).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am','am'};

%day2
k=2;
session_data(k).Day = '20240119';
session_data(k).folders = {'140305','140743','141436','141939','142525',...
    '143309','143645','144014','144321',...
    '145136','145454','145811'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am','am'};


save session_data_9DOF_B3 session_data


%% (MAIN) looking at decoding performance from imagined -> online -> batch
% across days

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_9DOF
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=10;
plot_true=true;
acc_batch_days_overall=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    if i==2
        online_idx=online_idx(2:end);
        online_idx=online_idx(2:3);
    end

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
    filename = ['condn_data_9DOF_ImaginedTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = accuracy_imagined_data_9DOF(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_imagined)))])
        xticks(1:9)
        yticks(1:9)
        xticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
            'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
        yticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
            'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
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
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_online)))])
        xticks(1:9)
        yticks(1:9)
        xticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
            'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
        yticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
            'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
    end
    acc_online_days(:,i) = diag(acc_online);


    %%%%%% classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_batch = accuracy_online_data_9DOF(files);
    if plot_true
        figure;imagesc(acc_batch)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_batch)))])
        xticks(1:9)
        yticks(1:9)
        xticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
            'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
        yticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
            'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
    end
    acc_batch_days(:,i) = diag(acc_batch);
    acc_batch_days_overall(:,:,i)=acc_batch;
end

save 9DOF_2days_accuracy_results_New -v7.3
%save hDOF_10days_accuracy_results -v7.3


%acc_online_days = (acc_online_days + acc_batch_days)/2;
figure;
ylim([0.2 1])
xlim([0.5 10.5])
hold on
plot(mean(acc_imagined_days,1))
plot(mean(acc_online_days,1))
plot(mean(acc_batch_days,1),'k')

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
for i=1:2
    plot(1+0.2*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',13,'Color',[cmap(end,:) .5])
    plot(2+0.2*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',13,'Color',[cmap(end,:) .5])
    plot(3+0.2*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',13,'Color',[cmap(end,:) .5])
end
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',20,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([.5 3.5])
ylim([0.3 1])
xticks(1:3)
xticklabels({'OL','CL1','CL2'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
yticks(0:.1:1)
set(gca,'FontSize',12)
ylim([0 1])
hline(1/9,'r')

tmp = [ m11' m22' m33'];
figure;boxplot(tmp)

% t-test
[h p tb st] = ttest(acc_imagined_days(:) ,acc_online_days(:));p
[h p tb st] = ttest(acc_imagined_days(:) ,acc_batch_days(:));p
[h p tb st] = ttest(acc_online_days(:) ,acc_batch_days(:));p

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

%% (MAIN B3) looking at decoding performance from imagined -> online -> batch
% across days

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_9DOF_B3
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=10;
plot_true=true;
acc_batch_days_overall=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);


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
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0,0);

    % TEMP STEP TO SEE IF WE CAN SWAP OUT ACTIONS
    % action 9 -> action 4. Then get all other 7DoF actions
    idx9 = [find([condn_data.targetID]==9)];
    idx8 = find([condn_data.targetID]==8);
    idx4 = find([condn_data.targetID]==4);
    k=1;condn_data_new={};
    for ii=1:length(condn_data)
        if condn_data(ii).targetID <=7 && condn_data(ii).targetID ~=4 ...
                && condn_data(ii).targetID ~=5
            condn_data_new(k).neural = condn_data(ii).neural;
            condn_data_new(k).targetID = condn_data(ii).targetID;
            k=k+1;
        end
    end
    for ii=1:length(idx9)
        condn_data_new(k).neural = condn_data(idx9(ii)).neural;
        condn_data_new(k).targetID = 4;%condn_data(idx9(ii)).targetID;
        k=k+1;
    end
    for ii=1:length(idx8)
        condn_data_new(k).neural = condn_data(idx8(ii)).neural;
        condn_data_new(k).targetID = 5;%condn_data(idx9(ii)).targetID;
        k=k+1;
    end

    [acc_imagined,train_permutations] = accuracy_imagined_data(condn_data_new, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    figure;imagesc(acc_imagined)
    colormap parula
    clim([0 1])
    set(gcf,'color','w')
    title(['Accuracy of ' num2str(100*mean(diag(acc_imagined)))])


    % save the data
    %     filename = ['condn_data_9DOF_ImaginedTrials_Day' num2str(i)];
    %     save(filename, 'condn_data', '-v7.3')

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = accuracy_imagined_data_9DOF(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_imagined)))])
        xticks(1:9)
        yticks(1:9)
        xticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
            'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
        yticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
            'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
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
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_online)))])
        xticks(1:9)
        yticks(1:9)
        xticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
            'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
        yticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
            'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
    end
    acc_online_days(:,i) = diag(acc_online);


    %%%%%% classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_batch = accuracy_online_data_9DOF(files);
    if plot_true
        figure;imagesc(acc_batch)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_batch)))])
        xticks(1:9)
        yticks(1:9)
        xticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Lips','Tongue',...
            'Both Middle','Rt. Wrist Pronation','Rt. Wrist Supination'})
        yticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Lips','Tongue',...
            'Both Middle','Rt. Wrist Pronation','Rt. Wrist Supination'})
    end
    acc_batch_days(:,i) = diag(acc_batch);
    acc_batch_days_overall(:,:,i)=acc_batch;
end

save 9DOF_2days_accuracy_results_New -v7.3
%save hDOF_10days_accuracy_results -v7.3


%acc_online_days = (acc_online_days + acc_batch_days)/2;
figure;
ylim([0.2 1])
xlim([0.5 10.5])
hold on
plot(mean(acc_imagined_days,1))
plot(mean(acc_online_days,1))
plot(mean(acc_batch_days,1),'k')

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
for i=1:2
    plot(1+0.2*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',13,'Color',[cmap(end,:) .5])
    plot(2+0.2*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',13,'Color',[cmap(end,:) .5])
    plot(3+0.2*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',13,'Color',[cmap(end,:) .5])
end
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',20,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([.5 3.5])
ylim([0.3 1])
xticks(1:3)
xticklabels({'OL','CL1','CL2'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
yticks(0:.1:1)
set(gca,'FontSize',12)
ylim([0 1])
hline(1/9,'r')

tmp = [ m11' m22' m33'];
figure;boxplot(tmp)

% t-test
[h p tb st] = ttest(acc_imagined_days(:) ,acc_online_days(:));p
[h p tb st] = ttest(acc_imagined_days(:) ,acc_batch_days(:));p
[h p tb st] = ttest(acc_online_days(:) ,acc_batch_days(:));p

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


%% (MAIN B3) removing two actions to make it 7DoF and evaluating performance

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_9DOF_B3
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=5;
plot_true=true;
acc_batch_days_overall=[];
files_imagined=[];
files_online=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    % combine batch and online
    online_idx = [online_idx];


    %disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% get the files imagined
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files_imagined = [files_imagined;findfiles('',folderpath)'];
    end

    %%%%%% get the files online
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files_online = [files_online;findfiles('',folderpath)'];
    end
end


% load all data
load('ECOG_Grid_8596_000067_B3.mat')
files_imagined = [files_imagined;files_online];

%load the imagined data
load('ECOG_Grid_8596_000067_B3.mat')
condn_data_imagined = [load_data_for_MLP_TrialLevel_B3(files_imagined,ecog_grid,0,0)];
tmp={};k=1;
for i=1:length(condn_data_imagined)
    if ~isempty(condn_data_imagined(i).targetID)
        tmp(k).neural = condn_data_imagined(i).neural;
        tmp(k).targetID = condn_data_imagined(i).targetID;
        k=k+1;
    end
end
condn_data_imagined=tmp;

% retain only the 7 main actions
condn_data_imagined7={};
k=1;
for i=1:length(condn_data_imagined)
    if condn_data_imagined(i).targetID <=7
        condn_data_imagined7(k).neural=condn_data_imagined(i).neural;
        condn_data_imagined7(k).targetID=condn_data_imagined(i).targetID;
        k=k+1;
    end
end

% swap out the actions, head and z-up
condn_data = condn_data_imagined;
idx9 = [find([condn_data.targetID]==9)];
idx8 = find([condn_data.targetID]==8);
idx4 = find([condn_data.targetID]==4);
k=1;condn_data_new={};
for ii=1:length(condn_data)
    if condn_data(ii).targetID <=7 && condn_data(ii).targetID ~=4 ...
           && condn_data(ii).targetID ~=5
        condn_data_new(k).neural = condn_data(ii).neural;
        condn_data_new(k).targetID = condn_data(ii).targetID;
        k=k+1;
    end
end
for ii=1:length(idx8)
    condn_data_new(k).neural = condn_data(idx8(ii)).neural;
    condn_data_new(k).targetID = 4;%condn_data(idx9(ii)).targetID;
    k=k+1;
end
for ii=1:length(idx9)
    condn_data_new(k).neural = condn_data(idx9(ii)).neural;
    condn_data_new(k).targetID = 5;%condn_data(idx9(ii)).targetID;
    k=k+1;
end

% test it on imagined data itself (CV)
[acc_imagined,train_permutations,acc_imag_bin] =...
    accuracy_imagined_data(condn_data_new, iterations);
acc_imagined=squeeze(nanmean(acc_imagined,1));
acc_imag_bin=squeeze(nanmean(acc_imag_bin,1));

figure;imagesc(acc_imagined)
colormap parula
clim([0 1])
set(gcf,'color','w')
title(['Accuracy of ' num2str(100*mean(diag(acc_imagined)))])
disp(diag(acc_imagined))
disp(diag(acc_imag_bin))

% using the old 7DoF
condn_data_new = condn_data_imagined7;
% test it on imagined data itself (CV)
[acc_imagined,train_permutations,acc_imag_bin] =...
    accuracy_imagined_data(condn_data_new, iterations);
acc_imagined=squeeze(nanmean(acc_imagined,1));
acc_imag_bin=squeeze(nanmean(acc_imag_bin,1));

figure;imagesc(acc_imagined)
colormap parula
clim([0 1])
set(gcf,'color','w')
title(['Accuracy of ' num2str(100*mean(diag(acc_imagined)))])
disp(diag(acc_imagined))
disp(diag(acc_imag_bin))


% build a classifier
condn_data_overall = condn_data_new;
test_idx = randperm(length(condn_data_overall),round(0.15*length(condn_data_overall)));
test_idx=test_idx(:);
I = ones(length(condn_data_overall),1);
I(test_idx)=0;
train_val_idx = find(I~=0);
prop = (0.7/0.85);
tmp_idx = randperm(length(train_val_idx),round(prop*length(train_val_idx)));
train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
I = ones(length(condn_data_overall),1);
I([train_idx;test_idx])=0;
val_idx = find(I~=0);val_idx=val_idx(:);

% training options for NN
[options,XTrain,YTrain] = ...
    get_options(condn_data_overall,val_idx,train_idx);

layers = get_layers1(120,759);
net = trainNetwork(XTrain,YTrain,layers,options);

% test network on imagined itself
cv_perf = test_network(net,condn_data_overall,test_idx);

% test network on all the held out online data
condn_data_online = [load_data_for_MLP_TrialLevel_B3(files_online,ecog_grid,0,0)];

% get only the first 7 actions
condn_data = condn_data_online;
idx9 = [find([condn_data.targetID]==9)];
idx8 = find([condn_data.targetID]==8);
idx4 = find([condn_data.targetID]==4);
k=1;condn_data_new={};
for ii=1:length(condn_data)
    if ~isempty(condn_data(ii).targetID) && ...
            condn_data(ii).targetID <=7 && condn_data(ii).targetID ~=4 ...
            && condn_data(ii).targetID ~=5
        condn_data_new(k).neural = condn_data(ii).neural;
        condn_data_new(k).targetID = condn_data(ii).targetID;
        k=k+1;
    end
end
for ii=1:length(idx8)
    condn_data_new(k).neural = condn_data(idx8(ii)).neural;
    condn_data_new(k).targetID = 4;%condn_data(idx9(ii)).targetID;
    k=k+1;
end
for ii=1:length(idx9)
    condn_data_new(k).neural = condn_data(idx9(ii)).neural;
    condn_data_new(k).targetID = 5;%condn_data(idx9(ii)).targetID;
    k=k+1;
end

test_idx =1:length(condn_data_new);
cv_perf = test_network(net,condn_data_new,test_idx);



% save the data
%     filename = ['condn_data_9DOF_ImaginedTrials_Day' num2str(i)];
%     save(filename, 'condn_data', '-v7.3')

% get cross-val classification accuracy
[acc_imagined,train_permutations] = accuracy_imagined_data_9DOF(condn_data, iterations);
acc_imagined=squeeze(nanmean(acc_imagined,1));
if plot_true
    figure;imagesc(acc_imagined)
    colormap bone
    clim([0 1])
    set(gcf,'color','w')
    title(['Accuracy of ' num2str(100*mean(diag(acc_imagined)))])
    xticks(1:9)
    yticks(1:9)
    xticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
        'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
    yticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
        'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
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
    clim([0 1])
    set(gcf,'color','w')
    title(['Accuracy of ' num2str(100*mean(diag(acc_online)))])
    xticks(1:9)
    yticks(1:9)
    xticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
        'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
    yticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Tongue','Lips',...
        'Both Middle','Rot. Rt. Wrist','Rot. Lt. Wrist'})
end
acc_online_days(:,i) = diag(acc_online);


%%%%%% classification accuracy for batch data
folders = session_data(i).folders(batch_idx);
day_date = session_data(i).Day;
files=[];
for ii=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
    %cd(folderpath)
    files = [files;findfiles('',folderpath)'];
end

% get the classification accuracy
acc_batch = accuracy_online_data_9DOF(files);
if plot_true
    figure;imagesc(acc_batch)
    colormap bone
    clim([0 1])
    set(gcf,'color','w')
    title(['Accuracy of ' num2str(100*mean(diag(acc_batch)))])
    xticks(1:9)
    yticks(1:9)
    xticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Lips','Tongue',...
        'Both Middle','Rt. Wrist Pronation','Rt. Wrist Supination'})
    yticklabels({'Rt. Thumb','Left Leg','Lt. Thumb','Head','Lips','Tongue',...
        'Both Middle','Rt. Wrist Pronation','Rt. Wrist Supination'})
end
acc_batch_days(:,i) = diag(acc_batch);
acc_batch_days_overall(:,:,i)=acc_batch;
end



%% PUTTING IT ALL TOGETHER IN TERMS OF BUILDING THE AE
%(MAIN MAIN)

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data_9DOF
dist_online_total=[];
dist_imag_total=[];
dist_batch_total=[];
var_imag_total=[];
mean_imag_total=[];
var_online_total=[];
mean_online_total=[];
var_batch_total=[];
mean_batch_total=[];
res=[];
mahab_full_online=[];
mahab_full_imagined=[];
mahab_full_batch=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    if i==2
        online_idx=online_idx(2:end);
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    %%%%%%imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP_9DOF(files);

    % save the data
    filename = ['condn_data_9DOF_Imagined_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

    %     % build the AE based on MLP and only for hG
    %    [net,Xtrain,Ytrain] = build_mlp_AE(condn_data);
    %     %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);
    %
    % get activations in deepest layer but averaged over a trial
    %     imag=1;
    %     [TrialZ_imag,dist_imagined,mean_imagined,var_imagined,idx_imag,~,condn_data_recon] = ...
    %         get_latent_regression(files,net,imag);
    %     dist_imag_total = [dist_imag_total;dist_imagined];
    %     mean_imag_total=[mean_imag_total;pdist(mean_imagined)];
    %     var_imag_total=[var_imag_total;var_imagined'];

    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    condn_data = load_data_for_MLP_9DOF(files);

    % save the data
    filename = ['condn_data_9DOF_Online_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];

    %     % get activations in deepest layer
    %     imag=0;
    %     [TrialZ_online,dist_online,mean_online,var_online,idx_online,~,condn_data_recon_online]...
    %         = get_latent_regression(files,net,imag);
    %     dist_online_total = [dist_online_total;dist_online];
    %     mean_online_total=[mean_online_total;pdist(mean_online)];
    %     var_online_total=[var_online_total;var_online'];



    %%%%%%batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    condn_data = load_data_for_MLP_9DOF(files);

    % save the data
    filename = ['condn_data_9DOF_Batch_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_batch = [mahab_full_batch Donline];

    % get activations in deepest layer
    %     imag=0;
    %     [TrialZ_batch,dist_batch,mean_batch,var_batch,idx_batch] = get_latent_regression(files,net,imag);
    %     dist_batch_total = [dist_batch_total;dist_batch];
    %     mean_batch_total=[mean_batch_total;pdist(mean_batch)];
    %     var_batch_total=[var_batch_total;var_batch'];

    % plotting imagined and online in latent space
    %     idxa = find(idx_imag==4);
    %     idxb = find(idx_online==4);
    %     idxa = idxa(randperm(length(idxa),length(idxb)));
    %     figure;hold on
    %     plot3(TrialZ_imag(1,idxa),TrialZ_imag(2,idxa),TrialZ_imag(3,idxa),'.','MarkerSize',20)
    %     plot3(TrialZ_online(1,idxb),TrialZ_online(2,idxb),TrialZ_online(3,idxb),'.','MarkerSize',20)
    %     c1 = TrialZ_imag(:,idxa);
    %     c2 = TrialZ_online(:,idxb);
    %     c1=cov(c1');
    %     c2=cov(c2');

    %      plot
    %
    %     figure;boxplot([dist_imagined' dist_online'])
    %     box off
    %     set(gcf,'Color','w')
    %     xticks(1:2)
    %     xticklabels({'Imagined Data','Online Data'})
    %     ylabel('Distance')
    %     title('Inter-class distances')
    %     set(gca,'LineWidth',1)
    %     set(gca,'FontSize',12)

    %     [h p tb st]=ttest(dist_imagined,dist_online);
    %     disp([p mean([dist_imagined' dist_online'])]);
    %     res=[res;[p mean([dist_imagined' dist_online'])]];
end


figure;
hold on
plot((median(mahab_full_imagined(:,1:end))))
plot((median(mahab_full_online(:,1:end))))
plot((median(mahab_full_batch(:,1:end))))


%% for chris, detecting start and stop of a signal


% Define the true signal parameters
jump_prob = 0.01; % Probability of a jump at each time step
jump_var = 0.5; % Variance of the Gaussian jump
x_true = 1; % True constant

% Generate the true signal
t = linspace(0, 10, 101); % Time vector
x_true_vec = x_true*ones(size(t)); % True constant signal
for i = 2:length(t)
    if rand < jump_prob
        x_true = x_true + sqrt(jump_var)*randn;
    end
    x_true_vec(i) = x_true;
end

% Add noise to the true signal
noise_var = 0.05; % Variance of the Gaussian noise
y_meas = x_true_vec + sqrt(noise_var)*randn(size(x_true_vec)); % Measurement signal

% Initialize Kalman filter parameters
x_est = 1; % State estimate (initial guess)
P_est = 1; % Error covariance matrix (initial guess)

% Define the system matrices for Kalman filter
dt = t(2) - t(1); % Sampling interval
F = 1; % State transition matrix
Q = 0.01; % Process noise covariance matrix (tuning parameter)
H = 1; % Measurement matrix
R = noise_var; % Measurement noise variance

% Run the Kalman filter
x_kf = zeros(size(t));
for i = 1:length(t)
    % Predict the state and error covariance
    x_pred = F*x_est;
    P_pred = F*P_est*F' + Q;

    % Update the state and error covariance based on the measurement
    K = P_pred*H'/(H*P_pred*H' + R);
    x_est = x_pred + K*(y_meas(i) - H*x_pred);
    P_est = (1 - K*H)*P_pred;

    % Save the estimated signal
    x_kf(i) = x_est;
end

% Plot the true signal, measurement signal, and estimated signal
figure;
plot(t, x_true_vec, '-b', t, y_meas, '.r', t, x_kf, '-g');
legend('True signal', 'Measurement signal', 'Estimated signal');
xlabel('Time');
ylabel('Constant');




