%% (MAIN) RUNNING LMM ON MAHAB DISTANCES FOR B1 AND B3
% to show that there is or is not a systematic trend across days


clc;clear

% load B1 data
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
a=load('mahab_dist_B1_latent');

% load B3 data
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
b=load('mahab_dist_b3_latent');

b1 = a.tmp;
b3 = b.tmp;


% % run LMM non parametric test, within subject but combining CL contexts
% day_name = [(1:size(b1,1))';(1:size(b1,1))'];
% mahab_dist = [b1(:,2);b1(:,3)];
% subj = ones(size(mahab_dist));
% mvmt_type = [ones(size(b1,1),1);2*ones(size(b1,1),1)];
% data = table(subj,day_name,mahab_dist,mvmt_type);
%
% % fit
% glm = fitlme(data,'mahab_dist ~  day_name + (1|mvmt_type) ')
% stat = glm.Coefficients.Estimate
%
% % permutation testing
% pval=[];stat_boot=[];
% for i=1:500
%     disp(i)
%     a = day_name(1:length(b1));
%     b = day_name(length(b1)+1:end);
%     a=a(randperm(numel(a)));
%     b=b(randperm(numel(b)));
%     day_name_tmp = [a;b];
%     data_tmp = table(subj,day_name_tmp,mahab_dist,mvmt_type);
%     glm_tmp = fitlme(data_tmp,'mahab_dist ~ 1 + day_name_tmp + (1|mvmt_type) ');
%     stat_boot = [stat_boot glm_tmp.Coefficients.tStat];
% end
% figure;
% hist(stat_boot(2,:));
% vline(stat(2))
% sum(stat(2)<=stat_boot(2,:))/500


%%%%% plotting , B1

day_name = [(1:size(b1,1))';(1:size(b1,1))'];
mahab_dist = [b1(:,2);b1(:,3)];
subj = ones(size(mahab_dist));
mvmt_type = [ones(size(b1,1),1);2*ones(size(b1,1),1)];
data = table(subj,day_name,mahab_dist,mvmt_type);

% fit
glm = fitlme(data,'mahab_dist ~  day_name + (1|mvmt_type) ')
stat = glm.Coefficients.Estimate

% regression lines
tmp=b1;
figure;
num_days=size(tmp,1);
xlim([0 num_days+1])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:num_days,tmp(:,1),'.b','MarkerSize',20)
y = tmp(:,1);
[B,BINT,R,RINT,STATS1] = regress(y,x);
lm = fitlm(x(:,2),y)
yhat = x*B;
plot(1:num_days,yhat,'b','LineWidth',1)
% online
plot(1:num_days,tmp(:,2),'.k','MarkerSize',20)
plot(1:num_days,tmp(:,2),'.','Color','k','MarkerSize',20)
y = tmp(:,2);
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat = x*B;
%plot(1:num_days,yhat,'Color',[.65 .65 .65 .35],'LineWidth',1)
% batch
plot(1:num_days,tmp(:,3),'.r','MarkerSize',20)
y = tmp(:,3);
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat = x*B;
%plot(1:num_days,yhat,'Color',[1 0 0 .35],'LineWidth',1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:num_days])

bhat = glm.Coefficients(:,2).Estimate;
yhat = x*bhat;
plot(1:num_days,yhat,'m','LineWidth',2)

bhat(1) = bhat(1)+ 7.2399;
yhat = x*bhat;
plot(1:num_days,yhat,'Color',[1 0 0 .35],'LineWidth',1)

bhat(1) = bhat(1)- 7.2399 - 7.2399;
yhat = x*bhat;
plot(1:num_days,yhat,'Color',[.65 .65 .65 .35],'LineWidth',1)

xlabel('Days')
ylabel('Mean Mahalanobis Distance')
ylim([0 75])
yticks([0:15:75])

%%%%% plotting B3:

day_name = [(1:size(b3,1))';(1:size(b3,1))'];
mahab_dist = [b3(:,2);b3(:,3)];
subj = ones(size(mahab_dist));
mvmt_type = [ones(size(b3,1),1);2*ones(size(b3,1),1)];
data = table(subj,day_name,mahab_dist,mvmt_type);

% fit
glm = fitlme(data,'mahab_dist ~  day_name + (1|mvmt_type) ')
stat = glm.Coefficients.Estimate

tmp=b3;
figure;
num_days=size(tmp,1);
xlim([0 num_days+1])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:num_days,tmp(:,1),'.b','MarkerSize',20)
y = tmp(:,1);
[B,BINT,R,RINT,STATS1] = regress(y,x);
lm = fitlm(x(:,2),y)
yhat = x*B;
plot(1:num_days,yhat,'b','LineWidth',1)
% online
plot(1:num_days,tmp(:,2),'.k','MarkerSize',20)
plot(1:num_days,tmp(:,2),'.','Color','k','MarkerSize',20)
y = tmp(:,2);
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat = x*B;
%plot(1:num_days,yhat,'Color',[.65 .65 .65 .35],'LineWidth',1)
% batch
plot(1:num_days,tmp(:,3),'.r','MarkerSize',20)
y = tmp(:,3);
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat = x*B;
%plot(1:num_days,yhat,'Color',[1 0 0 .35],'LineWidth',1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:num_days])


bhat = glm.Coefficients(:,2).Estimate;
yhat = x*bhat;
plot(1:num_days,yhat,'m','LineWidth',2)

bhat(1) = bhat(1)+ 7.349;
yhat = x*bhat;
plot(1:num_days,yhat,'Color',[1 0 0 .35],'LineWidth',1)

bhat(1) = bhat(1)- 7.349 - 7.349;
yhat = x*bhat;
plot(1:num_days,yhat,'Color',[.65 .65 .65 .35],'LineWidth',1)
ylim([0 80])
yticks([0:20:80])

%%%%%% run LMM non parametric test, one context at a time
pval=[];stat_overall=[];
for context = 1:size(b1,2)
    day_name=[];
    mahab_dist=[];
    subj=[];

    % get B1 data
    day_name = [day_name;(1:size(b1,1))';11];
    mahab_dist = [mahab_dist;b1(:,context);NaN];
    subj = [subj;ones(size(b1,1),1);1];

    % get B3 data
    day_name = [day_name;(1:size(b3,1))'];
    mahab_dist = [mahab_dist;b3(:,context)];
    subj = [subj;2*ones(size(b3,1),1)];

    % collate
    data = table(day_name,mahab_dist,subj);

    % fit
    glm = fitlme(data,'mahab_dist ~ 1+(day_name) + (1|subj)');

    % run boot statistics
    stat = glm.Coefficients.tStat(2);
    stat_overall(context)=stat;
    pval(context) = glm.Coefficients.pValue(2);
    stat_boot=[];
    parfor i=1:1000
        disp(i)
        aa = day_name(1:11);
        aa=aa(randperm(numel(aa)));
        bb = day_name(12:end);
        bb=bb(randperm(numel(bb)));
        day_name_tmp = [aa;bb];
        data_tmp = table(day_name_tmp,mahab_dist,subj);
        glm_tmp = fitglme(data_tmp,'mahab_dist ~ 1 + (day_name_tmp) + (1|subj)');
        stat_boot(i) = glm_tmp.Coefficients.tStat(2);
    end

    figure;
    hist(abs(stat_boot),20);
    vline(abs(stat),'r')
    pval(context)= sum(abs(stat_boot)>stat)/length(stat_boot);
end
pval

%%%%%% Comparing OL, CL1 and CL2
% boxplots
figure;hold on
boxplot([b1;b3])
a = get(get(gca,'children'),'children');
for i=1:length(a)
    box1 = a(i);
    set(box1, 'Color', 'k');
end
x1= [1+ 0.1*randn(length(b1),1) 2+ 0.1*randn(length(b1),1) 3+ 0.1*randn(length(b1),1)];
h=scatter(x1,b1,'filled');
for i=1:3
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.5;
end
x1= [1+ 0.1*randn(length(b3),1) 2+ 0.1*randn(length(b3),1) 3+ 0.1*randn(length(b3),1)];
h=scatter(x1,b3,'filled');
for i=1:3
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.5;
end
xlim([0.5 3.5])
xticks(1:3)
xticklabels({'OL','CL1','CL2'})
yticks([0:10:80])
ylim([0 80])
set(gcf,'Color','w')
box off

% stats on the OL, CL1 and CL2 using signed rank test
tmp = [b3];
median(tmp)
[P,H,STATS] = signrank(tmp(:,3),tmp(:,2),'method','approximate') % batch to imagined
[P,H,STATS] = signrank(tmp(:,3),tmp(:,2),'method','approximate') % batch to online
[P,H,STATS] = signrank(tmp(:,2),tmp(:,1),'method','approximate') % online to imagined

% stats using non parametric mixed effect models
tmp = [b3;b1];
subj=[];
mahab_dist=[];
mvmt_type=[];
subj=[ones(11,1);2*ones(10,1)];
subj=[subj;subj];
mahab_dist = [tmp(:,3);tmp(:,2)]; % change this to whatever appropriate
mvmt_type=[ones(21,1);2*ones(21,1)];
data = table(subj,mvmt_type,mahab_dist);

glm = fitlme(data,'mahab_dist ~ mvmt_type + (1|subj)')
stat = glm.Coefficients.tStat(2);
stat_boot=[];

% first is permute labels of the movement between the two subjects
% then permute the labels of the subject themselves

idx1 = find(subj==1);
idx2 = find(subj==2);
mvmt1 = mvmt_type(idx1);
mvmt2 = mvmt_type(idx2);
parfor i=1:2000
    disp(i)
    mvmt_type_tmp = mvmt_type;
    mvmt_type1_tmp = mvmt1(randperm(numel(mvmt1)));
    mvmt_type2_tmp = mvmt2(randperm(numel(mvmt2)));

    mvmt_type_tmp(idx1) = mvmt_type1_tmp;
    mvmt_type_tmp(idx2) = mvmt_type2_tmp;

    %mvmt_type_tmp = mvmt_type(randperm(numel(mvmt_type)));
    data_tmp = table(subj,mvmt_type_tmp,mahab_dist);
    glm_tmp = fitlme(data_tmp,'mahab_dist ~ mvmt_type_tmp + (1|subj)');
    stat_boot(i) = glm_tmp.Coefficients.tStat(2);
end
figure;
hist((stat_boot))
vline((stat))
sum(abs(stat_boot)>abs(stat))/length(stat_boot)
glm = fitlme(data,'mahab_dist ~ mvmt_type + (1|subj)')

%% PREDICTING DECODING ACCURACIES FROM MAHAB DISTANCES B1, B2 AND B3

clc;clear

%%%% B1
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load('mahab_dist_B1_latent');
mahab_dist = tmp;
clear tmp
a=load('hDOF_10days_accuracy_results_New_New_v3.mat');

a1=[];
for i=1:size(a.acc_imagined_days,3)
    x = squeeze(a.acc_imagined_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_imagined_days = a1;

a1=[];
for i=1:size(a.acc_online_days,3)
    x = squeeze(a.acc_online_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_online_days = a1;

a1=[];
for i=1:size(a.acc_batch_days,3)
    x = squeeze(a.acc_batch_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_batch_days = a1;

% getting data in order
tmp = [mean(acc_imagined_days,1)' mean(acc_online_days,1)' ...
    mean(acc_batch_days,1)'];

decoding_acc = tmp(:);
mahab_dist=mahab_dist(:);

% linear regression
figure;plot((mahab_dist),(decoding_acc),'.','MarkerSize',20)
y=decoding_acc;
x= [ones(length(mahab_dist),1) mahab_dist];
[B,BINT,R,RINT,STATS] = regress(y,x)
lm = fitlm(x(:,2),y)
%[b,p,b1]=logistic_reg(x(:,2),y);[b p']

%%% MAIN PLOTTING AND STATS

%2D fit
figure;
hold on
%col={'b','k','r'};
col=[.0 0 1 .5;
    .35 .35 .35 .5;
    1 0 0 .5];
k=1;
data={};
for i=1:10:30
    plot((mahab_dist(i:i+9)),decoding_acc(i:i+9),...
        '.','MarkerSize',20,'color',col(k,:));
    tmp = [mahab_dist(i:i+9) decoding_acc(i:i+9)];
    data{k}=tmp;
    k=k+1;
end

% logistic fit
x= [ones(length(mahab_dist),1) mahab_dist];
y=decoding_acc;
[b,p,b1]=logistic_reg(x(:,2),y);[b p']
xx = linspace(min(x(:,2)),max(x(:,2)),100);
xx = [ones(length(xx),1) xx'];
yhat = 1./(1+exp(-xx*b));
plot(xx(:,2),yhat,'Color','m','LineWidth',1);
xlim([0 70])
yticks([0:.1:1])
xlabel('Mahalanobis Distance')
ylabel('Decoder Accuracy')
set(gcf,'Color','w')
xticks([0:20:60])

% doing LOOCV on the logistic regression fit
cv_loss=[];cv_loss_r2=[];
I = ones(length(decoding_acc),1);
for i=1:length(decoding_acc)
    disp(i)
    test_idx = i;
    train_idx = I;
    train_idx(test_idx)=0;
    train_idx = find(train_idx>0);

    % fit the model on training data
    x=mahab_dist(train_idx);
    x= [ones(length(x),1) x];
    y=decoding_acc(train_idx);
    [b,p,b1]=logistic_reg(x(:,2),y);

    % prediction on held out data point
    xtest = mahab_dist(test_idx);
    xtest= [ones(length(xtest),1) xtest];
    yhat =  1./(1+exp(-xtest*b));
    ytest = decoding_acc(test_idx);
    cv_loss_r2(i) = abs((yhat-ytest));
    cv_loss(i) = -((ytest*log(yhat) + (1-ytest)*log(1-yhat)));
end
cv_loss_stat = cv_loss;
cv_loss_r2_stat = cv_loss_r2;

% doing it against a null distribution, 500 times
cv_loss_boot=[];cv_loss_r2_boot=[];
parfor iter =1:500
    disp(iter)
    cv_loss=[];cv_loss_r2=[];
    I = ones(length(decoding_acc),1);
    decoding_acc_tmp = decoding_acc(randperm(numel(decoding_acc)));
    for i=1:length(decoding_acc)

        test_idx = i;
        train_idx = I;
        train_idx(test_idx)=0;
        train_idx = find(train_idx>0);

        % fit the model on training data
        x=mahab_dist(train_idx);
        x= [ones(length(x),1) x];
        y=decoding_acc_tmp(train_idx);
        [b,p,b1]=logistic_reg(x(:,2),y);

        % prediction on held out data point
        xtest = mahab_dist(test_idx);
        xtest= [ones(length(xtest),1) xtest];
        yhat =  1./(1+exp(-xtest*b));
        ytest = decoding_acc_tmp(test_idx);
        cv_loss_r2(i) = abs((yhat-ytest));
        cv_loss(i) = -(ytest*log(yhat) + (1-ytest)*log(1-yhat));
    end
    cv_loss_boot(iter,:)=cv_loss;
    cv_loss_r2_boot(iter,:) = cv_loss_r2;
end
figure;
hist(mean(cv_loss_boot,2))
vline(mean(cv_loss_stat),'r')
sum(mean(cv_loss_boot,2) < mean(cv_loss_stat))/length(mean(cv_loss_boot,2))

figure;
hist(mean(cv_loss_r2_boot,2))
vline(mean(cv_loss_r2_stat),'r')
sum(mean(cv_loss_r2_boot,2) < mean(cv_loss_r2_stat))/length(mean(cv_loss_r2_boot,2))


%%%%% FOR B3 
clear

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
load('mahab_dist_B3_latent');
mahab_dist = tmp;
clear tmp
a=load('hDOF_11days_accuracy_results_B3_V6.mat');

a1=[];
for i=1:size(a.acc_imagined_days,3)
    x = squeeze(a.acc_imagined_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_imagined_days = a1;

a1=[];
for i=1:size(a.acc_online_days,3)
    x = squeeze(a.acc_online_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_online_days = a1;

a1=[];
for i=1:size(a.acc_batch_days,3)
    x = squeeze(a.acc_batch_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_batch_days = a1;

% getting data in order
tmp = [mean(acc_imagined_days,1)' mean(acc_online_days,1)' ...
    mean(acc_batch_days,1)'];

decoding_acc = tmp(:);
mahab_dist=mahab_dist(:);

% linear regression
figure;plot((mahab_dist),(decoding_acc),'.','MarkerSize',20)
y=decoding_acc;
x= [ones(length(mahab_dist),1) mahab_dist];
[B,BINT,R,RINT,STATS] = regress(y,x)
lm = fitlm(x(:,2),y)
%[b,p,b1]=logistic_reg(x(:,2),y);[b p']

%%% MAIN PLOTTING AND STATS

%2D fit
figure;
hold on
%col={'b','k','r'};
col=[.0 0 1 .5;
    .35 .35 .35 .5;
    1 0 0 .5];
k=1;
data={};
for i=1:11:33
    plot((mahab_dist(i:i+10)),decoding_acc(i:i+10),...
        '.','MarkerSize',20,'color',col(k,:));
    tmp = [mahab_dist(i:i+10) decoding_acc(i:i+10)];
    data{k}=tmp;
    k=k+1;
end

% logistic fit
x= [ones(length(mahab_dist),1) mahab_dist];
y=decoding_acc;
[b,p,b1]=logistic_reg(x(:,2),y);[b p']
xx = linspace(min(x(:,2)),max(x(:,2)),100);
xx = [ones(length(xx),1) xx'];
yhat = 1./(1+exp(-xx*b));
plot(xx(:,2),yhat,'Color','m','LineWidth',1);
xlim([10 80])
yticks([0:.1:1])
xlabel('Mahalanobis Distance')
ylabel('Decoder Accuracy')
set(gcf,'Color','w')
xticks([0:20:80])

% doing LOOCV on the logistic regression fit
cv_loss=[];cv_loss_r2=[];
I = ones(length(decoding_acc),1);
for i=1:length(decoding_acc)
    disp(i)
    test_idx = i;
    train_idx = I;
    train_idx(test_idx)=0;
    train_idx = find(train_idx>0);

    % fit the model on training data
    x=mahab_dist(train_idx);
    x= [ones(length(x),1) x];
    y=decoding_acc(train_idx);
    [b,p,b1]=logistic_reg(x(:,2),y);

    % prediction on held out data point
    xtest = mahab_dist(test_idx);
    xtest= [ones(length(xtest),1) xtest];
    yhat =  1./(1+exp(-xtest*b));
    ytest = decoding_acc(test_idx);
    cv_loss_r2(i) = abs((yhat-ytest));
    cv_loss(i) = -((ytest*log(yhat) + (1-ytest)*log(1-yhat)));
end
cv_loss_stat = cv_loss;
cv_loss_r2_stat = cv_loss_r2;

% doing it against a null distribution, 500 times
cv_loss_boot=[];cv_loss_r2_boot=[];
parfor iter =1:500
    disp(iter)
    cv_loss=[];cv_loss_r2=[];
    I = ones(length(decoding_acc),1);
    decoding_acc_tmp = decoding_acc(randperm(numel(decoding_acc)));
    for i=1:length(decoding_acc)

        test_idx = i;
        train_idx = I;
        train_idx(test_idx)=0;
        train_idx = find(train_idx>0);

        % fit the model on training data
        x=mahab_dist(train_idx);
        x= [ones(length(x),1) x];
        y=decoding_acc_tmp(train_idx);
        [b,p,b1]=logistic_reg(x(:,2),y);

        % prediction on held out data point
        xtest = mahab_dist(test_idx);
        xtest= [ones(length(xtest),1) xtest];
        yhat =  1./(1+exp(-xtest*b));
        ytest = decoding_acc_tmp(test_idx);
        cv_loss_r2(i) = abs((yhat-ytest));
        cv_loss(i) = -(ytest*log(yhat) + (1-ytest)*log(1-yhat));
    end
    cv_loss_boot(iter,:)=cv_loss;
    cv_loss_r2_boot(iter,:) = cv_loss_r2;
end
figure;
hist(mean(cv_loss_boot,2))
vline(mean(cv_loss_stat),'r')
sum(mean(cv_loss_boot,2) < mean(cv_loss_stat))/length(mean(cv_loss_boot,2))

figure;
hist(mean(cv_loss_r2_boot,2))
vline(mean(cv_loss_r2_stat),'r')
sum(mean(cv_loss_r2_boot,2) < mean(cv_loss_r2_stat))/length(mean(cv_loss_r2_boot,2))

%%%%% B2 %%%%
clear
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2')
tmp_overall =[0.0197233	0.210888	NaN
    0.115767	0.350175	0.429508
    0.0779838	0.142056	0.392535
    0.0594268	0.212396	0.557981
    0.193691	0.554491	0.834836
    0.0760399	0.347305	NaN
    ];
mahab_dist = tmp_overall;
a=load('hDOF_6days_accuracy_results_New_B3_v2.mat');



a1=[];
for i=1:size(a.acc_imagined_days,3)
    x = squeeze(a.acc_imagined_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_imagined_days = a1;

a1=[];
for i=1:size(a.acc_online_days,3)
    x = squeeze(a.acc_online_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_online_days = a1;

a1=[];
for i=1:size(a.acc_batch_days,3)
    x = squeeze(a.acc_batch_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_batch_days = a1;

% getting data in order
tmp = [mean(acc_imagined_days,1)' mean(acc_online_days,1)' ...
    mean(acc_batch_days,1)'];

decoding_acc = tmp(:);
mahab_dist=mahab_dist(:);

%decoding_acc = decoding_acc(~isnan(decoding_acc));
%mahab_dist = mahab_dist(~isnan(decoding_acc));

% linear regression
figure;plot((mahab_dist),(decoding_acc),'.','MarkerSize',20)
y=decoding_acc;
x= [ones(length(mahab_dist),1) mahab_dist];
[B,BINT,R,RINT,STATS] = regress(y,x)
lm = fitlm(x(:,2),y)
%[b,p,b1]=logistic_reg(x(:,2),y);[b p']

%%% MAIN PLOTTING AND STATS

%2D fit
figure;
hold on
%col={'b','k','r'};
col=[.0 0 1 .5;
    .35 .35 .35 .5;
    1 0 0 .5];
k=1;
data={};
for i=1:6:18
    plot((mahab_dist(i:i+5)),decoding_acc(i:i+5),...
        '.','MarkerSize',20,'color',col(k,:));
    tmp = [mahab_dist(i:i+5) decoding_acc(i:i+5)];
    data{k}=tmp;
    k=k+1;
end

% logistic fit
mahab_dist = mahab_dist(~isnan(mahab_dist));
decoding_acc = decoding_acc(~isnan(decoding_acc));
x= [ones(length(mahab_dist),1) mahab_dist];
y=decoding_acc;
%[b,p,b1]=logistic_reg(x(:,2),y);[b p']
lr = fitglm(x(:,2),y,'Distribution','Binomial');
b=lr.Coefficients.Estimate;
xx = linspace(min(x(:,2)),max(x(:,2)),100);
xx = [ones(length(xx),1) xx'];
yhat = 1./(1+exp(-xx*b));
plot(xx(:,2),yhat,'Color','m','LineWidth',1);
%xlim([10 80])
%yticks([0:.1:1])
xlabel('Mahalanobis Distance')
ylabel('Decoder Accuracy')
set(gcf,'Color','w')
%xticks([0:20:80])

% doing LOOCV on the logistic regression fit
cv_loss=[];cv_loss_r2=[];
I = ones(length(decoding_acc),1);
for i=1:length(decoding_acc)
    disp(i)
    test_idx = i;
    train_idx = I;
    train_idx(test_idx)=0;
    train_idx = find(train_idx>0);

    % fit the model on training data
    x=mahab_dist(train_idx);
    x= [ones(length(x),1) x];
    y=decoding_acc(train_idx);
    %[b,p,b1]=logistic_reg(x(:,2),y);
    lr = fitglm(x(:,2),y,'Distribution','Binomial');
    b=lr.Coefficients.Estimate;

    % prediction on held out data point
    xtest = mahab_dist(test_idx);
    xtest= [ones(length(xtest),1) xtest];
    yhat =  1./(1+exp(-xtest*b));
    ytest = decoding_acc(test_idx);
    cv_loss_r2(i) = abs((yhat-ytest));
    cv_loss(i) = -((ytest*log(yhat) + (1-ytest)*log(1-yhat)));
end
cv_loss_stat = cv_loss;
cv_loss_r2_stat = cv_loss_r2;

% doing it against a null distribution, 500 times
cv_loss_boot=[];cv_loss_r2_boot=[];
parfor iter =1:1000
    disp(iter)
    cv_loss=[];cv_loss_r2=[];
    I = ones(length(decoding_acc),1);
    decoding_acc_tmp = decoding_acc(randperm(numel(decoding_acc)));
    for i=1:length(decoding_acc)

        test_idx = i;
        train_idx = I;
        train_idx(test_idx)=0;
        train_idx = find(train_idx>0);

        % fit the model on training data
        x=mahab_dist(train_idx);
        x= [ones(length(x),1) x];
        y=decoding_acc_tmp(train_idx);
        %[b,p,b1]=logistic_reg(x(:,2),y);
        lr = fitglm(x(:,2),y,'Distribution','Binomial');
        b=lr.Coefficients.Estimate;

        % prediction on held out data point
        xtest = mahab_dist(test_idx);
        xtest= [ones(length(xtest),1) xtest];
        yhat =  1./(1+exp(-xtest*b));
        ytest = decoding_acc_tmp(test_idx);
        cv_loss_r2(i) = abs((yhat-ytest));
        cv_loss(i) = -(ytest*log(yhat) + (1-ytest)*log(1-yhat));
    end
    cv_loss_boot(iter,:)=cv_loss;
    cv_loss_r2_boot(iter,:) = cv_loss_r2;
end
figure;
hist(mean(cv_loss_boot,2))
vline(mean(cv_loss_stat),'r')
sum(mean(cv_loss_boot,2) < mean(cv_loss_stat))/length(mean(cv_loss_boot,2))

figure;
hist(mean(cv_loss_r2_boot,2))
vline(mean(cv_loss_r2_stat),'r')
sum(mean(cv_loss_r2_boot,2) < mean(cv_loss_r2_stat))/length(mean(cv_loss_r2_boot,2))



%% LOOKING AT THE NEURAL VARIANCES VIA MIXED EFFECT MODEL, B1 and B3 and B2

clc;clear
% B1:
b1=[30.857	6.23092	4.2398
    24.417	6.82874	2.97929
    9.04406	3.71701	0.861332
    27.5913	4.89941	4.2908
    56.641	8.15226	7.53884
    45.6299	8.2134	6.6176
    47.2558	11.765	8.78324
    49.3863	9.94427	8.23726
    47.2904	5.89819	4.20248
    46.8797	6.5413	4.60668
    ];
b1=log(b1);

% B3
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\neural_var_b3_latent.mat')
b3=tmp;
b3=log(b3);

%%%%%% Comparing OL, CL1 and CL2
% boxplots
figure;hold on
boxplot(([b1;b3]))
a = get(get(gca,'children'),'children');
for i=1:length(a)
    box1 = a(i);
    set(box1, 'Color', 'k');
end
x1= [1+ 0.1*randn(length(b1),1) 2+ 0.1*randn(length(b1),1) 3+ 0.1*randn(length(b1),1)];
h=scatter(x1,b1,'filled');
for i=1:3
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.5;
end
x1= [1+ 0.1*randn(length(b3),1) 2+ 0.1*randn(length(b3),1) 3+ 0.1*randn(length(b3),1)];
h=scatter(x1,b3,'filled');
for i=1:3
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.5;
end
xlim([0.5 3.5])
xticks(1:3)
xticklabels({'OL','CL1','CL2'})
%yticks([0:10:80])
%ylim([0 80])
set(gcf,'Color','w')
box off
yticks([0:2:8])

% stats on the OL, CL1 and CL2 using signed rank test
tmp = [b3];
median(tmp)
[P,H,STATS] = signrank(tmp(:,2),tmp(:,1),'method','approximate') % batch to imagined
[P,H,STATS] = signrank(tmp(:,3),tmp(:,2),'method','approximate') % batch to online
[P,H,STATS] = signrank(tmp(:,2),tmp(:,1),'method','approximate') % online to imagined

% stats using non parametric mixed effect models
tmp = [b3;b1];
subj=[];
mahab_dist=[];
mvmt_type=[];
subj=[ones(11,1);2*ones(10,1)];
subj=[subj;subj];
mahab_dist = [tmp(:,3);tmp(:,2)]; % change this to whatever appropriate
mvmt_type=[ones(21,1);2*ones(21,1)];
data = table(subj,mvmt_type,mahab_dist);

glm = fitlme(data,'mahab_dist ~ mvmt_type + (1|subj)')
stat = glm.Coefficients.tStat(2);
stat_boot=[];

% first is permute labels of the movement between the two subjects
% then permute the labels of the subject themselves

idx1 = find(subj==1);
idx2 = find(subj==2);
mvmt1 = mvmt_type(idx1);
mvmt2 = mvmt_type(idx2);
parfor i=1:2000
    disp(i)
    mvmt_type_tmp = mvmt_type;
    mvmt_type1_tmp = mvmt1(randperm(numel(mvmt1)));
    mvmt_type2_tmp = mvmt2(randperm(numel(mvmt2)));

    mvmt_type_tmp(idx1) = mvmt_type1_tmp;
    mvmt_type_tmp(idx2) = mvmt_type2_tmp;

    %mvmt_type_tmp = mvmt_type(randperm(numel(mvmt_type)));
    data_tmp = table(subj,mvmt_type_tmp,mahab_dist);
    glm_tmp = fitlme(data_tmp,'mahab_dist ~ mvmt_type_tmp + (1|subj)');
    stat_boot(i) = glm_tmp.Coefficients.tStat(2);
end
glm = fitlme(data,'mahab_dist ~ mvmt_type + (1|subj)')
figure;
hist((stat_boot))
vline((stat))
sum(abs(stat_boot)>abs(stat))/length(stat_boot)

%%%% neural variance in B2:
tmp_overall=[0.0023975	8.94036e-05	NaN
0.027231	0.0187138	0.0163215
0.0178991	0.0146173	0.00858447
0.0258339	0.0102953	0.00755429
0.0204377	0.0173461	0.010348
0.0456793	0.02214	NaN
];

%save tmp_overall_var_B2 tmp_overall
%load tmp_overall_var_B2

% plotting boxplots
figure;hold on
boxplot(tmp_overall)
a = get(get(gca,'children'),'children');
for i=1:length(a)
    box1 = a(i);
    set(box1, 'Color', 'k');
end

x1= [1+ 0.1*randn(length(tmp_overall),1) 2+ 0.1*randn(length(tmp_overall),1)...
    3+ 0.1*randn(length(tmp_overall),1)];
h=scatter(x1,tmp_overall,'filled');
for i=1:3
    h(i).MarkerFaceColor = 'm';
    h(i).MarkerFaceAlpha = 0.5;
end
set(gcf,'Color','w')
xlim([.5 3.5])
box off
%ylim([0 0.03])
ylim([-5. -3])
%yticks([0:.01:0.03])
xticklabels('')

% stats on them
[p,h,stats]=signrank(tmp_overall(:,1),nanmean(tmp_overall(:,2:3),2),'method','approximate')







%% LOOKING AT THE MAHAB DISTANCES FOR B2


%%% PLOTTING THE MAHAB DISTANCES AS BOXPLOTS AND DOING STATS

tmp_overall =[0.0197233	0.210888	NaN
    0.115767	0.350175	0.429508
    0.0779838	0.142056	0.392535
    0.0594268	0.212396	0.557981
    0.193691	0.554491	0.834836
    0.0760399	0.347305	NaN
    ];


% plotting boxplots
figure;hold on
boxplot(tmp_overall)
a = get(get(gca,'children'),'children');
for i=1:length(a)
    box1 = a(i);
    set(box1, 'Color', 'k');
end

x1= [1+ 0.1*randn(length(tmp_overall),1) 2+ 0.1*randn(length(tmp_overall),1)...
    3+ 0.1*randn(length(tmp_overall),1)];
h=scatter(x1,tmp_overall,'filled');
for i=1:3
    h(i).MarkerFaceColor = 'm';
    h(i).MarkerFaceAlpha = 0.5;
end
set(gcf,'Color','w')
xlim([.5 3.5])
box off
ylim([0 1])
yticks([0:.2:1])
xticklabels('')

% stats on them
[p,h,stats]=signrank(tmp_overall(:,1),nanmean(tmp_overall(:,2:3),2),'method','approximate')


%%%% JUST USING THE 4 DAYS DATA FOR PLOTTING THE REGRESSION LINES

tmp=[0.115767	0.350175	0.429508
    0.0779838	0.142056	0.392535
    0.0594268	0.212396	0.557981
    0.0760399	0.347305	0.834836
    ];



% PLOT REGRESSION LINES equal size in tmp
imag = tmp(:,1);
online = tmp(:,2);
batch = tmp(:,3);
figure;
hold on
days=1:size(tmp,1);
x=[ones(length(days),1) days'];
%imag
plot(days,imag,'.','MarkerSize',20)
y=imag;
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat=x*B;
plot(days,yhat,'b','LineWidth',1)
%online
plot(days,online,'.k','MarkerSize',20)
y=online;
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat=x*B;
plot(days,yhat,'k','LineWidth',1)
%batch
idx=~isnan(batch);
plot(days(idx),batch(idx),'.r','MarkerSize',20)
y=batch(idx);
[B,BINT,R,RINT,STATS3] = regress(y,x(idx,:));
yhat=x(idx,:)*B;
plot(days(idx),yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlim([days(1) - .5 days(end) + .5])
xticks(days)
xticklabels(days)
xlabel('Days')
ylabel('Mean Mahalanobis dist.')
yticks([0:.25:1])
ylim([0 1])


