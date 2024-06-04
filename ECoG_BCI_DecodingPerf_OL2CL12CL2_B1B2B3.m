

%% (MAIN) B1 looking at decoding performance from imagined -> online -> batch
% across days

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data
addpath 'C:\Users\nikic\Documents\MATLAB'
session_data = session_data([1:9 11]); % removing bad days
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=30;
plot_true=false;
num_trials_imag=[];
num_trials_online=[];
num_trials_batch=[];
binomial_res={};
binomial_res_chance={};
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    if i==3 || i==6 || i==8
        folders_pm = strcmp(session_data(i).AM_PM,'pm');
        folders_batch(folders_pm==0)=0;
        if i==8
            idx = find(folders_batch==1);
            folders_batch(idx(3:end))=0;
        end
    else
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_batch(folders_am==0) = 0;
    end

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

    % save the number of trials
    num_trials_imag = [num_trials_imag length(files)];

    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files,0,1);
    % save the data
    %filename = ['condn_data_ImaginedTrials_Day' num2str(i)];
    %save(filename, 'condn_data', '-v7.3')

    % get cross-val classification accuracy
    [acc_imagined,train_permutations,~,bino_pdf,bino_pdf_chance] =...
        accuracy_imagined_data(condn_data, iterations);%accuracy_imagined_data_crossVal
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined*100)
        colormap(brewermap(128,'Blues'))
        clim([0 100])
        set(gcf,'color','w')
        % add text
        for j=1:size(acc_imagined,1)
            for k=1:size(acc_imagined,2)
                if j==k
                    text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','w')
                else
                    text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','k')
                end
            end
        end
        box on
        xticks(1:7)
        yticks(1:7)
        xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
        yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
        title(['OL Acc of ' num2str(100*mean(diag(acc_imagined)))])
    end
    acc_imagined_days(:,:,i) = (acc_imagined);

    % store binomial results
    n = round(median([bino_pdf(1:end).n]));
    succ = round(median([bino_pdf(1:end).succ]));
    pval = binopdf(succ,n,(1/7));
    binomial_res(i).Imagined = [pval];

    % store binomial chance results
    n = round(median([bino_pdf_chance(1:end).n]));
    succ = round(median([bino_pdf_chance(1:end).succ]));
    p=succ/n;
    xx= 0:n;
    bp = binopdf(xx,n,p);
    ch = ceil((1/7)*n);
    [aa,bb] = find(xx==ch);
    pval = sum(bp(1:bb));
    binomial_res_chance(i).Imagined = [pval];

    %%%%%% get classification accuracy for online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    num_trials_online = [num_trials_online length(files)];

    % get the classification accuracy
    %[acc_online,~,bino_pdf] = accuracy_online_data(files);
    [acc_online,~,bino_pdf] = accuracy_online_data_5bins(files);
    
    if plot_true
        figure;imagesc(acc_online*100)
        colormap(brewermap(128,'Blues'))
        clim([0 100])
        set(gcf,'color','w')
        % add text
        for j=1:size(acc_online,1)
            for k=1:size(acc_online,2)
                if j==k
                    text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','w')
                else
                    text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','k')
                end
            end
        end
        box on
        xticklabels ''
        yticklabels ''
    end
    acc_online_days(:,:,i) = (acc_online);
    %store binomial results
    binomial_res(i).online = [bino_pdf.pval];

    n = bino_pdf.n;
    succ = bino_pdf.succ;
    p=succ/n;
    xx= 0:n;
    bp = binopdf(xx,n,p);
    ch = ceil((1/7)*n);
    [aa,bb] = find(xx==ch);
    pval = sum(bp(1:bb));
    binomial_res_chance(i).online = [pval];


    %%%%%% cross_val classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    num_trials_batch = [num_trials_batch length(files)];

    % get the classification accuracy
    %[acc_batch,~,bino_pdf] = accuracy_online_data(files);
    [acc_batch,~,bino_pdf] = accuracy_online_data_5bins(files);
    if plot_true
        figure;imagesc(acc_batch*100)
        colormap(brewermap(128,'Blues'))
        clim([0 100])
        set(gcf,'color','w')
        % add text
        for j=1:size(acc_batch,1)
            for k=1:size(acc_batch,2)
                if j==k
                    text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','w')
                else
                    text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','k')
                end
            end
        end
        box on
        xticklabels ''
        yticklabels ''
    end
    %acc_batch_days(:,i) = diag(acc_batch);
    acc_batch_days(:,:,i) = (acc_batch);
    binomial_res(i).batch = [bino_pdf.pval];

    n = bino_pdf.n;
    succ = bino_pdf.succ;
    p=succ/n;
    xx= 0:n;
    bp = binopdf(xx,n,p);
    ch = ceil((1/7)*n);
    [aa,bb] = find(xx==ch);
    pval = sum(bp(1:bb));
    binomial_res_chance(i).batch = [pval];

end

tmp = squeeze(nanmean(acc_imagined_days,3));
mean(diag(tmp))
tmp = squeeze(nanmean(acc_online_days,3));
mean(diag(tmp))
tmp = squeeze(nanmean(acc_batch_days,3));
mean(diag(tmp))

%save hDOF_10days_accuracy_results_New -v7.3
%save hDOF_10days_accuracy_results_New_New -v7.3 % made some corrections on how accuracy is computed
%save hDOF_10days_accuracy_results -v7.3
%save hDOF_10days_accuracy_results_New_New_v2 -v7.3 %corrections and using all confusion matrices
save hDOF_10days_accuracy_results_New_New_v3 -v7.3 %corrections and using all confusion matrices and with binomial


a = load('hDOF_10days_accuracy_results_New');
b = load('hDOF_10days_accuracy_results_New_New');
[mean(a.acc_imagined_days,1)' mean(b.acc_imagined_days,1)' ]
[mean(a.acc_online_days,1)' mean(b.acc_online_days,1)' ]
[mean(a.acc_batch_days,1)' mean(b.acc_batch_days,1)' ]

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
tmp=[];
for i=1:length(days)
    a=squeeze(acc_imagined_days(:,:,i));
    tmp = [tmp diag(a)];
end
y=mean(tmp,1)';
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
for i=1:10
    plot(1+0.1*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
    plot(2+0.1*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
    plot(3+0.1*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
end
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',20,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([.5 3.5])
ylim([0.5 1])
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
glm = fitglm(data,'acc ~ 1 + experiment','Distribution','binomial');

% stats
[h p tb st] = ttest(tmp,tmp1)
[h p tb st] = ttest(tmp,tmp2)
[h p tb st] = ttest(tmp1,tmp2)

[p,h,stats]=signrank(tmp,tmp1);
[p,h,stats]=signrank(tmp,tmp2);
[p,h,stats]=signrank(tmp1,tmp2);

p = bootstrp_ttest(tmp,tmp1,1)
p = bootstrp_ttest(tmp,tmp2,1)
p = bootstrp_ttest(tmp1,tmp2,1)


%%%%% IMPORTANT %%%%
% correlating perfomance to neural variance and mahab
% mahab_dist=[7.44937	25.0954	46.8808
%     8.98259	23.0811	43.5335
%     8.53827	27.0567	34.1529
%     5.63418	30.2603	36.0458
%     7.84022	40.0723	50.1397
%     10.5456	44.6843	50.8638
%     8.78371	36.7279	49.353
%     6.72437	32.8186	39.9
%     9.00313	38.3491	53.8316
%     11.4657	44.171	61.1524
%     ];
load mahab_dist_B1_latent
mahab_dist=tmp;


%[p,h]=ranksum(mahab_dist(:,2),mahab_dist(:,3))
[p,h,stats]=signrank(mahab_dist(:,3),mahab_dist(:,2),'method','exact')

neural_var=[30.857	6.23092	4.2398
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

median(neural_var)

[p,h,stats]=signrank(neural_var(:,1),neural_var(:,2));p
[p,h,stats]=signrank(neural_var(:,1),neural_var(:,3));p
[p,h,stats]=signrank(neural_var(:,3),neural_var(:,2));p

% permutation test
stat_val = abs(mean(neural_var(:,1)) - mean(neural_var(:,2)));
stat_boot=[];
stat_vec = [neural_var(:,1);neural_var(:,2)];
stat_vec= stat_vec - mean(stat_vec);
for i=1:5000
    idx = randperm(numel(stat_vec));
    a1 = stat_vec(idx(1:70));
    a2 = stat_vec(idx(71:140));
    stat_boot(i) =  abs(mean(a1)-mean(a2));
end
figure;hist(stat_boot)
vline(stat_val,'r')
title(['pval ' num2str(1-sum(stat_val>stat_boot)/length(stat_boot))])
xlabel('OL minus CL1, neural variance')
title('Permutation test');
ylabel('Frequency')
box off

neural_var=neural_var(:);
mahab_dist=mahab_dist(:);


tmp = [mean(acc_imagined_days,1)' mean(acc_online_days,1)' ...
    mean(acc_batch_days,1)'];

decoding_acc = tmp(:);

figure;plot((neural_var),(decoding_acc),'.','MarkerSize',20)
y=decoding_acc;
x= [ones(length(neural_var),1) neural_var];
[B,BINT,R,RINT,STATS] = regress(y,x)
[b,p,b1]=logistic_reg(x(:,2),y);[b p']

figure;plot((mahab_dist),(decoding_acc),'.','MarkerSize',20)
y=decoding_acc;
x= [ones(length(mahab_dist),1) mahab_dist];
[B,BINT,R,RINT,STATS] = regress(y,x)
[b,p,b1]=logistic_reg(x(:,2),y);[b p']


%2D fit
figure;
hold on
col={'b','k','r'};k=1;
data={};
for i=1:10:30
    plot((mahab_dist(i:i+9)),decoding_acc(i:i+9),'.','MarkerSize',20,'color',col{k});
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
plot(xx(:,2),yhat,'Color','k','LineWidth',1);
xlim([0 70])
yticks([0:.1:1])
xlabel('Mahalanobis Distance')
ylabel('Decoder Accuracy')
set(gcf,'Color','w')

% doing LOOCV on the logistic regression fit
cv_loss=[];
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
    %cv_loss(i) = abs((yhat-ytest));
    cv_loss(i) = -(ytest*log(yhat) + (1-ytest)*log(1-yhat));
end
cv_loss_stat = cv_loss;

% doing it against a null distribution, 500 times
cv_loss_boot=[];
parfor iter =1:500
    disp(iter)
    cv_loss=[];
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
        %cv_loss(i) = abs((yhat-ytest));
        cv_loss(i) = -(ytest*log(yhat) + (1-ytest)*log(1-yhat));
    end
    cv_loss_boot(iter,:)=cv_loss;
end
figure;
hist(mean(cv_loss_boot,2))
vline(mean(cv_loss_stat))
sum(mean(cv_loss_boot,2) < mean(cv_loss_stat))/length(mean(cv_loss_boot,2))

% 3D plot
neural_var = log(neural_var);
figure;
hold on
col={'r','g','b'};k=1;
data={};
for i=1:10:30
    plot3(mahab_dist(i:i+9),neural_var(i:i+9),(decoding_acc(i:i+9)),'.',...
        'MarkerSize',30,'color',col{k});
    tmp = [mahab_dist(i:i+9) neural_var(i:i+9) (decoding_acc(i:i+9)) ];
    %tmp = [mahab_dist(i:i+10) neural_var(i:i+10) ];
    data{k}=tmp;
    k=k+1;
end
xlabel('Mahalanobis Distance')
ylabel('Neural variance')
zlabel('Decoding Accuracy')


% logistic regression
data_overall = cell2mat(data');
x = data_overall(:,1:2);
y = data_overall(:,3);
[b,p,b1]=logistic_reg(x,y);
mdl = fitglm(x,y,'Distribution','Binomial');
mdl = mdl.Coefficients.Estimate;
bhat = mdl;
% plot as surface
xx = linspace(min(x(:,1)),max(x(:,1)),1e2);
yy = linspace(min(x(:,2)),max(x(:,2)),1e2);
[X,Y]=meshgrid(xx,yy);
zhat = [ones(length(X(:)),1) X(:) Y(:)];
zhat = 1./(1 + exp(-zhat*bhat));
zhat= reshape(zhat,size(X));
%figure;hold on
%grid on
%scatter3(x(:,1),x(:,2),y,'filled')
s=surf(X,Y,zhat,'FaceAlpha',.25);
s.EdgeColor = 'none';
s.FaceColor='cyan';
legend({'Open loop','CL1','CL2','Logistic Fit'})
set(gcf,'Color','w')
grid on
title('Neural Variance and Mahab distance predicts Decoding Acc')


boot=[];
parfor iter=1:5000
    x1=x;
    [bb,bint,r]=regress(x(:,2),[ones(length(x),1) x(:,1)]);
    x1(:,2)=r;
    x1(:,2) = x1(randperm(numel(y)),2);
    %y1=y(randperm(numel(y)));
    out = fitglm(x1,y,'Distribution','Binomial');
    boot = [boot out.Coefficients.Estimate];
end
pval=[];
for i=1:size(boot,1)
    figure;
    hist(abs(boot(i,:)),20);
    vline(abs(mdl(i)));
    pval(i) = sum(abs(boot(i,:)) >= abs(mdl(i)))/ length(boot(i,:));
    title(num2str(pval(i)))
end

% plot surface
xhat = [ones(size(x,1),1) x];
[xx,yy]=meshgrid(min(xhat(:,2)):0.1:max(xhat(:,2)), min(xhat(:,3)):1:max(xhat(:,3)));
yhat_1 = 1./(1+ exp(mdl(1) + mdl(2)*xx + mdl(3)*yy));
figure;
%mesh(xhat(:,2),xhat(:,3),yhat)
mesh(yy,xx,yhat_1)


x=randn(20,1);
y=randn(20,1);
z=2*x+3*y+2*randn(20,1);
[bhat]=regress(z,[ones(size(x,1),1) x y]);
zhat = [ones(size(x,1),1) x y]*bhat;
figure;
[X,Y]=meshgrid(-3:.01:3,-3:.01:3);
zhat = [ones(length(X(:)),1) X(:) Y(:)]*bhat;
zhat= reshape(zhat,size(X));
figure;hold on
scatter3(x,y,z,'filled')
mesh(X,Y,zhat,'FaceAlpha',.5)

% mahalanobis distance
D=zeros(length(data));
for i=1:length(data)
    a = data{i};
    for j=i+1:length(data)
        b = data{j};
        D(i,j) = mahal2(a,b,2);
        D(j,i) = D(i,j);
    end
end

% 2-means cluster index pairwise with swapping of labels
a = data{3};
b = data{2};
stat = two_means_ci(a,b);
% swap labels
boot=[];
d=[a;b];
s = size(a,1);
for i=1:5000
    idx = randperm(length(d));
    tmp = d(idx,:);
    atmp = tmp(1:s,:);
    btmp = tmp(s+1:end,:);
    boot(i) = two_means_ci(atmp,btmp);
end
figure;hist(boot)
vline(stat)
sum(stat>=boot)/length(boot)

% 2-means cluster index pairwise and null hypothesis testing for the
% two-means cluster index using gaussian distribution
K=zeros(length(data));
P=zeros(length(data));
D=zeros(length(data));
P_d=zeros(length(data));
for i=1:length(data)
    a = data{i};
    for j=i+1:length(data)
        b = data{j};
        if j==3
            b=b(2:end,:);
        end


        % 2 means ci
        K(i,j) = two_means_ci(a,b);
        K(j,i) = K(i,j);
        stat = K(i,j);

        % null testing for each pairwise distance
        % build a common distribution from the two datasets
        a1=a';b1=b';
        s1 = size(a1,2);
        c1 = [a1 b1];
        m = mean(c1,2);
        X = cov(c1');
        C12 = chol(X);
        dboot=[];
        parfor iter=1:5000
            g = randn(size(c1));
            cnew = m + C12'*g;
            % find two clusters in the data
            idx = kmeans(cnew', 2);
            atmp = cnew(:,find(idx==1));
            btmp = cnew(:,find(idx==2));
            dboot(iter) =  two_means_ci(atmp',btmp');
        end
        P(i,j) = 1-sum(dboot>stat)/length(dboot);

        % mahab dist
        D(i,j) = mahal2(a,b,2);
        D(j,i) = D(i,j);
        stat = D(i,j);

        % null testing for each pairwise distance
        % build a common distribution from the two datasets
        a1=a';b1=b';
        s1 = size(a1,2);
        c1 = [a1 b1];
        m = mean(c1,2);
        X = cov(c1');
        C12 = chol(X);
        dboot=[];
        parfor iter=1:5000
            g = randn(size(c1));
            cnew = m + C12'*g;
            % find two clusters in the data
            idx = kmeans(cnew', 2);
            atmp = cnew(:,find(idx==1));
            btmp = cnew(:,find(idx==2));
            dboot(iter) =  mahal2(atmp',btmp',2);
        end
        P_d(i,j) = sum(dboot>stat)/length(dboot);
    end
end


% using LDA on random split
a =data{1};
b = data{3};
d = [a;b];
idx = [0*ones(size(a,1),1);ones(size(a,1),1)];idx_main=idx;
acc=[];
res_acc=[];pval_acc=[];
for iter=1:25
    % randomly select 18 for training and 4 for testing
    idx = idx_main(randperm(numel(idx_main)));
    idx_train = randperm(size(d,1),16);
    I =  ones(size(d,1),1);
    I(idx_train)=0;
    idx_test = find(I==1);

    % train the LDA
    data_train = d(idx_train,:);
    idx_train = idx(idx_train);
    W = LDA(data_train,idx_train);

    % apply on held out data
    data_test = d(idx_test,:);
    data_test = [ones(size(data_test,1),1) data_test];
    idx_test = idx(idx_test);
    L = data_test * W';
    P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);
    [aa,bb]=max(P');bb=bb-1;
    acc(iter) = sum(idx_test==bb')/length(bb);

    % balanced accuracy
    tp=0;tn=0;fp=0;fn=0;
    p = idx_test;grp_test=bb';
    for ii=1:length(p)
        if p(ii)==1 && grp_test(ii)==1
            tp=tp+1;
        end

        if p(ii)==0 && grp_test(ii)==0
            tn=tn+1;
        end

        if p(ii)==1 && grp_test(ii)==0
            fn=fn+1;
        end

        if p(ii)==0 && grp_test(ii)==1
            fp=fp+1;
        end
    end
    res_acc(iter) = 0.5* ( tp/(tp+fn) + tn/(tn+fp) );

    % stats
    alp1=1+tp;
    bet1=1+fn;
    alp2=1+tn;
    bet2=1+fp;
    res=0.001;
    u=0:res:1;
    a=betapdf(u,alp1,bet1);
    b=betapdf(u,alp2,bet2);
    x=conv(a,b);
    z=2*x(1:2:end);
    z=z/(sum(x*res));
    % figure;plot(u,z);hold on;plot(u,a,'k');plot(u,b,'r')
    % calculate p-value
    querypt= 0.5;
    I=(u>querypt);
    pval(iter)=1-sum(z(I)*res);
end
figure;boxplot(bootstrp(1000,@mean,acc))
acc=mean(acc)



% using LDA on LOOCV
a =data{2};
b = data{3};
d = [a;b];
idx = [0*ones(size(a,1),1);ones(size(a,1),1)];idx_main=idx;
acc=[];
res_acc=[];pval_acc=[];
for i=1:length(d)
    idx_test=i;
    I =  ones(size(d,1),1);
    I(idx_test)=0;
    idx_train = find(I==1);

    % train the LDA
    data_train = d(idx_train,:);
    idx_train = idx(idx_train);
    W = LDA(data_train,idx_train);

    % apply on held out data
    data_test = d(idx_test,:);
    data_test = [ones(size(data_test,1),1) data_test];
    idx_test = idx(idx_test);
    L = data_test * W';
    P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);
    [aa,bb]=max(P');bb=bb-1;
    acc(i) = sum(idx_test==bb);
end
acc=mean(acc)










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

a1 = (mean(acc_online_days,1))';
a2 = (mean(acc_batch_days,1))';
stat = mean(a2)-mean(a1);
boot=[];
a=[a1;a2];
for i=1:1000
    idx = randperm(length(a));
    a11 = a(idx(1:10));
    a22 = a(idx(11:end));
    boot(i) = mean(a11) - mean(a22);
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


%% B2: looking at decoding performance from imagined -> online -> batch (MAIN)
% across days

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B2
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=10;
plot_true=false;
binomial_res={};
binomial_res_chance={};
num_trials_imag=[];
num_trials_cv_imag=[];
num_trials_online=[];
num_trials_batch=[];
num_succ_imag=[];
num_succ_online=[];
num_succ_batch=[];
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
        folderpath = fullfile(root_path, day_date,'CenterOut',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    num_trials_imag = [num_trials_imag length(files)];

    %load the data
    load('ECOG_Grid_8596-002131.mat')
    condn_data = load_data_for_MLP_TrialLevel_B2(files,ecog_grid);

    % get cross-val classification accuracy
    [acc_imagined,train_permutations,~,bino_pdf,bino_pdf_chance] = ...
        accuracy_imagined_data_B2(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_imagined_days(:,:,i) = (acc_imagined);
    num_succ_imag = [num_succ_imag mean([bino_pdf(1:end).succ])];
    num_trials_cv_imag = [num_trials_cv_imag mean([bino_pdf(1:end).n])];

    % store binomial results
    n = round(median([bino_pdf(1:end).n]));
    succ = round(median([bino_pdf(1:end).succ]));
    pval = binopdf(succ,n,(1/4));
    binomial_res(i).Imagined = [pval];

    % store binomial chance results
    n = round(median([bino_pdf_chance(1:end).n]));
    succ = round(median([bino_pdf_chance(1:end).succ]));
    p=succ/n;
    xx= 0:n;
    bp = binopdf(xx,n,p);
    ch = ceil((1/4)*n);
    [aa,bb] = find(xx==ch);
    pval = sum(bp(1:bb));
    binomial_res_chance(i).Imagined = [pval];


    %%%%%% get classification accuracy for online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'DiscreteArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    num_trials_online = [num_trials_online length(files)];

    % get the classification accuracy
    [acc_online,~,bino_pdf] = accuracy_online_data_B2(files);
    if plot_true
        figure;imagesc(acc_online)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_online_days(:,:,i) = (acc_online);
    num_succ_online = [num_succ_online mean([bino_pdf(1:end).succ])];

    % store binomial results
    binomial_res(i).online = [bino_pdf.pval];

    n = bino_pdf.n;
    succ = bino_pdf.succ;
    p=succ/n;
    xx= 0:n;
    bp = binopdf(xx,n,p);
    ch = ceil((1/4)*n);
    [aa,bb] = find(xx==ch);
    pval = sum(bp(1:bb));
    binomial_res_chance(i).online = [pval];


    %%%%%% cross_val classification accuracy for batch data
    if ~isempty(batch_idx)
        folders = session_data(i).folders(batch_idx);
        day_date = session_data(i).Day;
        files=[];
        for ii=1:length(folders)
            folderpath = fullfile(root_path, day_date,'DiscreteArrow',folders{ii},'BCI_Fixed');
            %cd(folderpath)
            files = [files;findfiles('',folderpath)'];
        end
        num_trials_batch = [num_trials_batch length(files)];

        % get the classification accuracy
        [acc_batch,~,bino_pdf] = accuracy_online_data_B2(files);
        if plot_true
            figure;imagesc(acc_batch)
            colormap bone
            clim([0 1])
            set(gcf,'color','w')
        end
        acc_batch_days(:,:,i) = (acc_batch);
        num_succ_batch = [num_succ_batch mean([bino_pdf(1:end).succ])];
        % store binomial results
        binomial_res(i).batch = [bino_pdf.pval];

        n = bino_pdf.n;
        succ = bino_pdf.succ;
        p=succ/n;
        xx= 0:n;
        bp = binopdf(xx,n,p);
        ch = ceil((1/4)*n);
        [aa,bb] = find(xx==ch);
        pval = sum(bp(1:bb));
        binomial_res_chance(i).batch = [pval];
    else
        acc_batch_days(:,:,i) = NaN*ones(4,4);
        binomial_res(i).batch = NaN;
        binomial_res_chance(i).batch=NaN;
    end
end

tmp = squeeze(nanmean(acc_imagined_days,3));
mean(diag(tmp))
tmp = squeeze(nanmean(acc_online_days,3));
mean(diag(tmp))
tmp = squeeze(nanmean(acc_batch_days,3));
mean(diag(tmp))

tmp = sort(bootstrp(1000,@mean,num_trials_imag));
round([tmp(25) mean(tmp) tmp(975)])

tmp = sort(bootstrp(1000,@mean,num_trials_online));
round([tmp(25) mean(tmp) tmp(975)])

tmp = sort(bootstrp(1000,@mean,num_trials_batch));
round([tmp(25) mean(tmp) tmp(975)])


load ('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\b1_acc_rel_imagined_prop.mat')
save hDOF_6days_accuracy_results_New_B3_v2 -v7.3 % correction and confusion matrix and binomial


%binomial accuracy p values overall the data
% imag
total_trials = sum(num_trials_cv_imag);
total_succ = round(sum(num_succ_imag));
p = total_succ/total_trials;
xx = 0:total_trials;
bp = binopdf(xx,total_trials,p);
figure;plot(xx,bp)
ch = ceil((1/4)*total_trials);
vline(ch)
[aa,bb] = find(xx==ch);
pval=sum(bp(1:bb))
pval_orig = binopdf(total_succ,total_trials,0.25)

% online
total_trials = sum(num_trials_online);
total_succ = sum(num_succ_online);
p = total_succ/total_trials;
xx = 0:total_trials;
bp = binopdf(xx,total_trials,p);
figure;plot(xx,bp)
ch = ceil((1/4)*total_trials);
vline(ch)
[aa,bb] = find(xx==ch);
pval=sum(bp(1:bb))
pval_orig = binopdf(total_succ,total_trials,0.25)

% batch
total_trials = sum(num_trials_batch);
total_succ = sum(num_succ_batch);
p = total_succ/total_trials;
xx = 0:total_trials;
bp = binopdf(xx,total_trials,p);
figure;plot(xx,bp)
ch = ceil((1/4)*total_trials);
vline(ch)
[aa,bb] = find(xx==ch);
pval=sum(bp(1:bb))
pval_orig = binopdf(total_succ,total_trials,0.25)

% combined online and batch
total_trials = sum([num_trials_batch num_trials_online(2:5)]);
total_succ = sum([num_succ_batch num_succ_online(2:5)]);
p = total_succ/total_trials;
xx = 0:total_trials;
bp = binopdf(xx,total_trials,p);
figure;plot(xx,bp)
ch = ceil((1/4)*total_trials);
vline(ch)
[aa,bb] = find(xx==ch);
pval=sum(bp(1:bb))


% 
% %acc_online_days = (acc_online_days + acc_batch_days)/2;
% figure;
% ylim([0.0 0.65])
% xlim([0.5 6.5])
% hold on
% plot(nanmean(acc_imagined_days,1))
% plot(nanmean(acc_online_days,1))
% plot(nanmean(acc_batch_days,1),'k')
% 
% % as regression lines
% figure;plot(mean(acc_imagined_days,1),'.','MarkerSize',20)
% 
% % stats
% tmp = [median(acc_imagined_days,1)' median(acc_online_days,1)' ...
%     median(acc_batch_days,1)'];
% 
% figure;boxplot(acc_imagined_days)
% ylim([0.2 1])
% xlim([0.5 10.5])
% hold on
% boxplot(acc_batch_days,'Colors','k')
% a = get(get(gca,'children'),'children');

x=[];
for i=1:size(acc_imagined_days,3)
    a = squeeze(acc_imagined_days(:,:,i));
    x= [x diag(a)];
end
acc_imagined_days=x;

x=[];
for i=1:size(acc_online_days,3)
    a = squeeze(acc_online_days(:,:,i));
    x= [x diag(a)];
end
acc_online_days=x;

x=[];
for i=1:size(acc_batch_days,3)
    a = squeeze(acc_batch_days(:,:,i));
    x= [x diag(a)];
end
acc_batch_days=x;

figure;
boxplot([acc_imagined_days(:) acc_online_days(:) acc_batch_days(:)])

m1 = 100*(acc_imagined_days(:));
m1b = sort(bootstrp(1000,@mean,m1));
m11 = 100*mean(acc_imagined_days,1);
m2 = 100*(acc_online_days(:));
m2b = sort(bootstrp(1000,@mean,m2));
m22 = 100*mean(acc_online_days,1);
m3 = 100*(acc_batch_days(:));
m3b = sort(bootstrp(1000,@nanmean,m3));
m33 = 100*nanmean(acc_batch_days,1);
x=1:3;
y=[mean(m1) mean(m2) nanmean(m3)];
s1=std(m1)/sqrt(length(m1));
s2=std(m2)/sqrt(length(m2));
s3=nanstd(m3)/sqrt(sum(~isnan(m3)));
neg = [s1 s2 s3];
pos= [s1 s2 s3];
%neg = [y(1)-m1b(25) y(2)-m2b(25) y(3)-m3b(25)];
%pos = [m1b(975)-y(1) m2b(975)-y(2) m3b(975)-y(3)];
figure;
hold on
cmap = brewermap(6,'Blues');
%cmap = (turbo(7));
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',10,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
for i=1:size(acc_batch_days,2)
    plot(1+0.1*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
    plot(2+0.1*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
    plot(3+0.1*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
end
xlim([.5 3.5])
ylim([0.10 0.55]*100)
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
yticks([0:.1:1]*100)
set(gca,'FontSize',12)
h=hline(25,'--');
h.LineWidth=1;
xlabel('Decoder Type')
ylabel('Accuracy')
xticklabels({'OL','CL1','CL2'})

tmp = [ m11' m22' m33'];
figure;boxplot(tmp)

tmp = [ m1 m2 m3];
figure;boxplot(tmp)

%rank sum tests to compare CL1 and CL2 vs. OL
ol_acc = m11;
cl_acc = nanmean([m22;m33]);
[p,h,stats]=signrank(ol_acc,cl_acc)




% bootstrapped statistics on whether closed loop control sig. above chance
acc_CL1 = mean(acc_online_days,1);
a = acc_CL1 - mean(acc_CL1) + 0.25;
acc_CL1_boot = sort(bootstrp(1000,@mean,a));
figure;hist(acc_CL1_boot)
vline(mean(acc_CL1))
sum(acc_CL1_boot>mean(acc_CL1))/length(acc_CL1_boot)

acc_CL2 = mean(acc_batch_days(:,2:5),1);
a = acc_CL2 - mean(acc_CL2) + 0.25;
acc_CL2_boot = sort(bootstrp(1000,@mean,a));
figure;hist(acc_CL2_boot)
vline(mean(acc_CL2))
sum(acc_CL2_boot>mean(acc_CL2))/length(acc_CL2_boot)

% regression lines for mahab distances in latent space
imag = [0.732087
    0.471915
    0.526922
    0.187374
    0.611636
    0.501813
    ];
online=[4.09371
    1.46306
    0.93976
    2.13517
    1.05116
    1.34932
    ];
batch=[1.58127
    2.48264
    3.26464
    3.10718
    ]; % days 12 thru 5

% PLOT REGRESSION LINES
imag = tmp(:,1);
online = tmp(:,2);
batch = tmp(1:4,3);
figure;
hold on
days=1:4;
x=[ones(length(days),1) days'];
%imag
plot(days,imag(2:5),'.','MarkerSize',20)
y=imag(2:5);
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat=x*B;
plot(days,yhat,'b','LineWidth',1)
%online
plot(days,online(2:5),'.k','MarkerSize',20)
y=online(2:5);
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat=x*B;
plot(days,yhat,'k','LineWidth',1)
%batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlim([.5 4.5])
xticks(1:4)
xticklabels(1:4)
ylim([0 3.5])
yticks([0:3])


% PLOT REGRESSION LINES equal size in tmp
imag = tmp(:,1);
online = tmp(:,2);
batch = tmp(:,3);
figure;
hold on
days=1:4;
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
plot(days,batch,'.r','MarkerSize',20)
y=batch;
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlim([.5 4.5])
xticks(1:4)
xticklabels(1:4)



% using robust regression in matlab equal size
figure;
hold on
% imag
plot(days,imag,'.b','MarkerSize',20)
y=imag;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'b','LineWidth',1)
% online
plot(days,online,'.k','MarkerSize',20)
y=online;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
% beautify
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:4])



% using robust regression in matlab
figure;
hold on
% imag
plot(days,imag(2:5),'.b','MarkerSize',20)
y=imag(2:5);
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'b','LineWidth',1)
% online
plot(days,online(2:5),'.k','MarkerSize',20)
y=online(2:5);
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
% beautify
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:4])
yticks([5:5:35])
ylim([0 1])





% stats on the latent space variance
tmp=[0.000937406	0.0985546	0.0289443
    0.00154575	0.0315809	0.0753841
    0.00141096	0.0103554	0.20185
    0.00043903	0.0643351	0.150029
    0.00125143	0.00337939	0.112707
    0.00113499	0.00377055	0.112707
    ];
[P,ANOVATAB,STATS] = anova1(tmp);

[h p tb st] = ttest(tmp(:,1),tmp(:,2));p
[h p tb st] = ttest(tmp(:,1),tmp(:,3));p
[h p tb st] = ttest(tmp(:,2),tmp(:,3));p

% how about bootstrapped difference of means?
stat  =  mean(tmp(:,2)) - mean(tmp(:,1));
c= [tmp(:,2);tmp(:,1)];
l=size(tmp,1);
boot=[];
c=c-mean(c);
for i=1:5000
    idx=randperm(numel(c));
    a1 = c(idx(1:l));
    b1 = c(idx(l+1:end));
    boot(i) = mean(a1)-mean(b1);
end
figure;hist((boot))
vline((stat),'r')
sum(boot>stat)/length(boot)


% get the accuracies relative to imagined movement within that day
a0 = mean(acc_imagined_days(:,2:5),1);
a1 = mean(acc_online_days(:,2:5),1);
a2 = mean(acc_batch_days(:,2:5),1);
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

b2_acc_rel_imagined = [a1' a2'];

acc_rel_imagined = [b1_acc_rel_imagined_prop ;b2_acc_rel_imagined];
figure;
boxplot(acc_rel_imagined);
a = get(get(gca,'children'),'children');   % Get the handles of all the objects
t = get(a,'tag');   % List the names of all the objects
box1 = a(5);   % The 7th object is the first box
set(box1, 'Color', 'k');   % Set the color of the first box to green
box1 = a(6);   % The 7th object is the first box
set(box1, 'Color', 'k');   % Set the color of the first box to green
line1 = a(3);   % The 7th object is the first box
set(line1, 'Color', 'm');   % Set the color of the first box to green
line1 = a(4);   % The 7th object is the first box
set(line1, 'Color', 'm');   % Set the color of the first box to green



% plotting scatter plot with mean and SE
m1 = acc_rel_imagined(:,1);
m1b = sort(bootstrp(1000,@mean,m1));
m2 = acc_rel_imagined(:,2);
m2b = sort(bootstrp(1000,@mean,m2));
y=[mean(acc_rel_imagined)];
s1=std(m1)/sqrt(length(m1));
s2=std(m2)/sqrt(length(m2));
neg = [s1 s2 ];
pos= [s1 s2 ];
neg = [y(1)-m1b(25) y(2)-m2b(25) ];
pos = [m1b(975)-y(1) m2b(975)-y(2) ];
figure;
hold on
% now the scatter part
%B1
hold on
scatter(ones(10,1)+0.07*randn(10,1),b1_acc_rel_imagined_prop(:,1),'or')
scatter(2*ones(10,1)+0.07*randn(10,1),b1_acc_rel_imagined_prop(:,2),'r')
%B2
scatter(ones(4,1)+0.07*randn(4,1),b2_acc_rel_imagined(:,1),'ob')
scatter(2*ones(4,1)+0.07*randn(4,1),b2_acc_rel_imagined(:,2),'ob')

cmap = brewermap(6,'Blues');
x=1:2;
for i=1:length(y)
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',10,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([0.5 2.5])
hline(0,'--k')
xticks([1 2])
xticklabels({'CL1','CL2'})
ylim([-0.2 0.6])
yticks([-.2:.2:.6])
set(gcf,'Color','w')


[h p tb st] = ttest(acc_rel_imagined(:,1),acc_rel_imagined(:,2))

% saving all data
save acc_relative_imagined_prop_B1B2 -v7.3

% using a mixed effect model
acc_improv=[];
subject=[];
exp_type=[];
%b1
for i=1:size(b1_acc_rel_imagined_prop,2)
    tmp=b1_acc_rel_imagined_prop(:,i);
    acc_improv =[acc_improv;tmp];
    exp_type = [exp_type;i*ones(size(tmp))];
    subject=[subject;1*ones(size(tmp))];
end
%b2
for i=1:size(b2_acc_rel_imagined,2)
    tmp=b2_acc_rel_imagined(:,i);
    %     if i==1
    %         tmp=tmp([1 3 4]);
    %     end
    acc_improv =[acc_improv;tmp];
    exp_type = [exp_type;i*ones(size(tmp))];
    subject=[subject;2*ones(size(tmp))];
end

data=table(acc_improv,exp_type,subject);
glme = fitglme(data,'acc_improv ~ 1+ exp_type + (1|subject)')


%%%%% PREDICTING DECODING PERFORMANCE FROM MAHAB DISTANCES IN LATENT SPACE
% get all the accuracy data in correct format
a1=[];
for i=1:size(acc_imagined_days,3)
    x = squeeze(acc_imagined_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_imagined_days = a1;

a1=[];
for i=1:size(acc_online_days,3)
    x = squeeze(acc_online_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_online_days = a1;

a1=[];
for i=1:size(acc_batch_days,3)
    x = squeeze(acc_batch_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
acc_batch_days = a1;


b=load('mahab_dist_B2_latent.mat');
mahab_dist = b.tmp;
mahab_dist=mahab_dist(:);

tmp = [mean(acc_imagined_days(:,2:5),1)' mean(acc_online_days(:,2:5),1)' ...
    mean(acc_batch_days(:,2:5),1)'];

decoding_acc = tmp(:);

% linear regresssion prelims
figure;plot((mahab_dist),(decoding_acc),'.','MarkerSize',20)
y=decoding_acc;
x= [ones(length(mahab_dist),1) mahab_dist];
[B,BINT,R,RINT,STATS] = regress(y,x);STATS
[b,p,b1]=logistic_reg(x(:,2),y);[b p']


%2D fit
figure;
hold on
col={'b','k','r'};k=1;
data={};
for i=1:4:length(mahab_dist)
    plot((mahab_dist(i:i+3)),decoding_acc(i:i+3),'.','MarkerSize',20,'color',col{k});
    tmp = [mahab_dist(i:i+3) decoding_acc(i:i+3)];
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
plot(xx(:,2),yhat,'Color','k','LineWidth',1);
xlim([0 0.9])
yticks([0:.05:1])
xlabel('Mahalanobis Distance')
ylabel('Decoder Accuracy')
set(gcf,'Color','w')

fitglm(x(:,2),y,'Distribution','Binomial')
fitlm(x(:,2),y,'Robust','on')

% linear regression
[B,BINT,R,RINT,STATS] = regress(y,x);STATS
lm = fitlm(x(:,2),y)



% doing LOOCV on the logistic regression fit
cv_loss=[];
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
    cv_loss(i) = abs((yhat-ytest));
    %cv_loss(i) = -(ytest*log(yhat) + (1-ytest)*log(1-yhat));
end
cv_loss_stat = cv_loss;

% doing it against a null distribution, 500 times
cv_loss_boot=[];
parfor iter =1:50
    disp(iter)
    cv_loss=[];
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
        cv_loss(i) = abs((yhat-ytest));
        %cv_loss(i) = -(ytest*log(yhat) + (1-ytest)*log(1-yhat));
    end
    cv_loss_boot(iter,:)=cv_loss;
end
figure;
hist(mean(cv_loss_boot,2))
vline(mean(cv_loss_stat))
sum(mean(cv_loss_boot,2) < mean(cv_loss_stat))/length(mean(cv_loss_boot,2))




%% B3: looking at decoding performance from imagined -> online -> batch (MAIN)
% across days

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3
addpath 'C:\Users\nikic\Documents\MATLAB'
addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')

num_trials_imag=[];
num_trials_online=[];
num_trials_batch=[];
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=10;
plot_true=false;
binomial_res={};
binomial_res_chance={};
for i=1:11% length(session_data) % 11 is first set of collected data
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

    num_trials_imag = [num_trials_imag length(files)];

    %load the data
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0);

    % get cross-val classification accuracy
    [acc_imagined,train_permutations,~,bino_pdf,bino_pdf_chance] =...
        accuracy_imagined_data(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined*100)
        colormap(brewermap(128,'Blues'))
        clim([0 100])
        set(gcf,'color','w')
        % add text
        for j=1:size(acc_imagined,1)
            for k=1:size(acc_imagined,2)
                if j==k
                    text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','w')
                else
                    text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','k')
                end
            end
        end
        box on
        xticks(1:7)
        yticks(1:7)
        xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
        yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
        title(['OL Acc of ' num2str(100*mean(diag(acc_imagined)))])
    end
    acc_imagined_days(:,:,i) = (acc_imagined);

    % store binomial results
    n = round(median([bino_pdf(1:end).n]));
    succ = round(median([bino_pdf(1:end).succ]));
    pval = binopdf(succ,n,(1/7));
    binomial_res(i).Imagined = [pval];

    % store binomial chance results
    n = round(median([bino_pdf_chance(1:end).n]));
    succ = round(median([bino_pdf_chance(1:end).succ]));
    p=succ/n;
    xx= 0:n;
    bp = binopdf(xx,n,p);
    ch = ceil((1/7)*n);
    [aa,bb] = find(xx==ch);
    pval = sum(bp(1:bb));
    binomial_res_chance(i).Imagined = [pval];


    %%%%%% get classification accuracy for online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    num_trials_online = [num_trials_online length(files)];


    % get the classification accuracy
    %[acc_online,~,bino_pdf] = accuracy_online_data(files);
    [acc_online,~,bino_pdf] = accuracy_online_data_5bins(files);    
    if plot_true
        figure;imagesc(acc_online*100)
        colormap(brewermap(128,'Blues'))
        clim([0 100])
        set(gcf,'color','w')
        % add text
        for j=1:size(acc_online,1)
            for k=1:size(acc_online,2)
                if j==k
                    text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','w')
                else
                    text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','k')
                end
            end
        end
        box on
        xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
        yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
        title(['CL1 Acc of ' num2str(100*mean(diag(acc_online)))])
    end
    %acc_online_days(:,i) = diag(acc_online_bin);
    acc_online_days(:,:,i) = (acc_online);
    %store binomial results
    binomial_res(i).online = [bino_pdf.pval];

    n = bino_pdf.n;
    succ = bino_pdf.succ;
    p=succ/n;
    xx= 0:n;
    bp = binopdf(xx,n,p);
    ch = ceil((1/7)*n);
    [aa,bb] = find(xx==ch);
    pval = sum(bp(1:bb));
    binomial_res_chance(i).online = [pval];


    %%%%%% cross_val classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    num_trials_batch = [num_trials_batch length(files)];

    % get the classification accuracy
    %[acc_batch,acc_batch_bin,bino_pdf] = accuracy_online_data(files);
    [acc_batch,acc_batch_bin,bino_pdf] = accuracy_online_data_5bins(files);
    if plot_true
        figure;imagesc(acc_batch*100)
        colormap(brewermap(128,'Blues'))
        clim([0 100])
        set(gcf,'color','w')
        % add text
        for j=1:size(acc_batch,1)
            for k=1:size(acc_batch,2)
                if j==k
                    text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','w')
                else
                    text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','k')
                end
            end
        end
        box on
        set(gcf,'color','w')
        xticks(1:7)
        yticks(1:7)
        xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
        yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
        title(['CL2 Acc of ' num2str(100*mean(diag(acc_batch)))])
    end
    %acc_batch_days(:,i) = diag(acc_batch);
    acc_batch_days(:,:,i) = (acc_batch);
    %store binomial results
    binomial_res(i).batch = [bino_pdf.pval];

    n = bino_pdf.n;
    succ = bino_pdf.succ;
    p=succ/n;
    xx= 0:n;
    bp = binopdf(xx,n,p);
    ch = ceil((1/7)*n);
    [aa,bb] = find(xx==ch);
    pval = sum(bp(1:bb));
    binomial_res_chance(i).batch = [pval];
end

tmp = squeeze(nanmean(acc_imagined_days,3));
mean(diag(tmp))
tmp = squeeze(nanmean(acc_online_days,3));
mean(diag(tmp))
tmp = squeeze(nanmean(acc_batch_days,3));
mean(diag(tmp))

%load ('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\b1_acc_rel_imagined_prop.mat')
%save hDOF_6days_accuracy_results_New_B2 -v7.3
%save hDOF_11days_accuracy_results_B3 -v7.3
%save hDOF_11days_accuracy_results_B3_corrected -v7.3 % not good
%save hDOF_11days_accuracy_results_B3_v2 -v7.3 % new after old data got deleted: best of the lot
%save hDOF_11days_accuracy_results_B3_v3 -v7.3 % new after old data got deleted
%save hDOF_11days_accuracy_results_B3_v4 -v7.3 % new and correcting for errors in accuracy computation
%save hDOF_11days_accuracy_results_B3_V5 -v7.3 %IMPT new and getting confusion matrix of each day
save hDOF_11days_accuracy_results_B3_V6 -v7.3 %IMPT V5 plus binomial decoding acc.

% plot the p-values of the decoding accuracies



tmp=[];
for i=1:size(acc_imagined_days,3)
    x=squeeze(acc_imagined_days(:,:,i));
    tmp = [tmp diag(x)];
end

tmp = squeeze(nanmean(acc_imagined_days,3));
mean(diag(tmp))

tmp = squeeze(nanmean(acc_online_days,3));
mean(diag(tmp))

tmp = squeeze(nanmean(acc_batch_days,3));
mean(diag(tmp))


a=load('hDOF_11days_accuracy_results_B3_v4');
b=load('hDOF_11days_accuracy_results_B3_v2'); %imag acc 0.8386
([nanmean(a.acc_batch_days,1)' nanmean(b.acc_batch_days,1)'])


%%%%%%%%% PLOTTING DECODING ACCURACIES FROM OL TO CL1 AND CL2
%acc_online_days = (acc_online_days + acc_batch_days)/2;
figure;
ylim([0.0 1])
xlim([0.5 11.5])
hold on
plot(nanmean(acc_imagined_days,1))
plot(nanmean(acc_online_days,1),'r')
plot(nanmean(acc_batch_days,1),'k')

% as regression lines
figure;plot(mean(acc_imagined_days,1),'.','MarkerSize',20)

% stats
tmp = [median(acc_imagined_days,1)' median(acc_online_days,1)' ...
    median(acc_batch_days,1)'];

figure;boxplot(acc_imagined_days)
ylim([0.2 1])
xlim([0.5 11.5])
hold on
boxplot(acc_batch_days,'Colors','k')
a = get(get(gca,'children'),'children');

figure;
boxplot([acc_imagined_days(:) acc_online_days(:) acc_batch_days(:)])


acc_imagined_days=acc_imagined_days(:,1:end);
acc_online_days=acc_online_days(:,1:end);
acc_batch_days=acc_batch_days(:,1:end);

%m1 = (acc_imagined_days(:));
m1=mean(acc_imagined_days,1);
m1b = sort(bootstrp(1000,@mean,m1));
m11 = mean(acc_imagined_days,1);
%m2 = (acc_online_days(:));
m2=mean(acc_online_days,1);
m2b = sort(bootstrp(1000,@mean,m2));
m22 = mean(acc_online_days,1);
%m3 = (acc_batch_days(:));
m3=mean(acc_batch_days,1);
m3b = sort(bootstrp(1000,@nanmean,m3));
m33 = nanmean(acc_batch_days,1);
x=1:3;
y=[mean(m1) mean(m2) nanmean(m3)];
s1=std(m1)/sqrt(length(m1));
s2=std(m2)/sqrt(length(m2));
s3=nanstd(m3)/sqrt(sum(~isnan(m3)));
%neg = [s1 s2 s3];
%pos= [s1 s2 s3];
neg = [y(1)-m1b(25) y(2)-m2b(25) y(3)-m3b(25)];
pos = [m1b(975)-y(1) m2b(975)-y(2) m3b(975)-y(3)];
figure;
hold on
cmap = brewermap(6,'Blues');
%cmap = (turbo(7));
for i=1:size(acc_batch_days,2)
    plot(1+0.1*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
    plot(2+0.1*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
    plot(3+0.1*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
end
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',15,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([.5 3.5])
ylim([0.7 1])
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
yticks(0:.05:1)
set(gca,'FontSize',12)
h=hline(.25,'--');
h.LineWidth=1;
xlabel('Decoder Type')
ylabel('Accuracy')

disp([nanmean(acc_imagined_days(:)) nanmean(acc_online_days(:))...
    nanmean(acc_batch_days(:))])

% using logistic regression
[b,p]=logistic_reg(mean(acc_imagined_days,1),mean(acc_online_days,1))

% using bootstrapped test of the mean
[p ]=bootstrap_diff_mean(mean(acc_imagined_days,1),mean(acc_online_days,1),1e3)
x= mean(acc_batch_days,1);
y = mean(acc_online_days,1);
[b,p,b1]=logistic_reg(x,y);p

[p, tvalue, bootstrp_tvalues]=bootstrp_ttest(mean(acc_online_days,1),...
    mean(acc_imagined_days,1),1,1000);p


tmp = [ m11' m22' m33'];
figure;boxplot(tmp)

tmp = [ m1 m2 m3];
figure;boxplot(tmp)

% Signed rank test
[P,H,STATS] = signrank(mean(acc_batch_days,1),mean(acc_online_days,1));
[P,H,STATS] = signrank(mean(acc_imagined_days,1),mean(acc_online_days,1));
[P,H,STATS] = signrank(mean(acc_batch_days,1),mean(acc_imagined_days,1));

%%%% PLOTTING REGRESSION LINES FOR MAHAB DISTANCES AS A FUNCTION OF DAY %%%%
% load tmp variable here from python
num_days = size(tmp,1);
figure;
xlim([0 num_days+1])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:num_days,tmp(:,1),'.b','MarkerSize',20)
y = tmp(:,1);
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
plot(1:num_days,yhat,'b','LineWidth',1)
% online
plot(1:num_days,tmp(:,2),'.k','MarkerSize',20)
y = tmp(:,2);
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat = x*B;
plot(1:num_days,yhat,'k','LineWidth',1)
% batch
plot(1:num_days,tmp(:,3),'.r','MarkerSize',20)
y = tmp(:,3);
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat = x*B;
plot(1:num_days,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:num_days])
% yticks([5:5:35])
% ylim([5 35])

% using robust regression in matlab
figure;
xlim([0 num_days+1])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:num_days,tmp(:,1),'.b','MarkerSize',20)
y = tmp(:,1);
lm=fitlm(x(:,2:end),y,'Robust','on')
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(1:num_days,yhat,'b','LineWidth',1)
% online
plot(1:num_days,tmp(:,2),'.k','MarkerSize',20)
y = tmp(:,2);
lm=fitlm(x(:,2:end),y,'Robust','on')
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(1:num_days,yhat,'k','LineWidth',1)
% batch
plot(1:num_days,tmp(:,3),'.r','MarkerSize',20)
y = tmp(:,3);
lm=fitlm(x(:,2:end),y,'Robust','on')
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(1:num_days,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:num_days+1])
% yticks([5:5:35])
% ylim([5 35])



%%%%% IMPORTANT %%%%
% correlating perfomance to neural variance and mahab
% mahab_dist=[15.0768	23.0341	25.5969
%     21.0006	35.7938	36.6155
%     19.838	32.3876	36.9021
%     23.3012	38.4087	36.7424
%     24.9293	39.2602	50.2872
%     17.9506	27.7369	35.1501
%     25.1043	40.5396	40.0646
%     23.7339	39.6258	38.6774
%     26.754	39.9261	58.23
%     26.0079	39.4577	53.3721
%     21.0177	39.0024	47.8239
%     ];

load mahab_dist_b3_latent
mahab_dist = tmp;

%[p,h]=ranksum(mahab_dist(:,2),mahab_dist(:,3))
[p,h,stats]=signrank(mahab_dist(:,3),mahab_dist(:,2),'method','exact')

% neural_var=[987.905	546.457	435.817
%     1032.13	564.041	489.257
%     1337.76	781.075	639.571
%     1637.73	795.645	769.213
%     1371.44	890.384	599.882
%     1189.63	790.712	752.482
%     1181.93	472.442	516.443
%     1304.32	751.878	684.07
%     1269.3	664.604	434.364
%     1489.88	791.045	531.593
%     1158.13	522.424	402.161
%     ];
load neural_var_b3_latent
neural_var=tmp;


[p,h,stats]=signrank(neural_var(:,1),neural_var(:,2))
[p,h,stats]=signrank(neural_var(:,1),neural_var(:,3))
[p,h,stats]=signrank(neural_var(:,3),neural_var(:,2))

% permutation test
stat_val = abs(mean(neural_var(:,1)) - mean(neural_var(:,2)));
stat_boot=[];
stat_vec = [neural_var(:,1);neural_var(:,2)];
stat_vec= stat_vec - mean(stat_vec);
for i=1:5000
    idx = randperm(numel(stat_vec));
    a1 = stat_vec(idx(1:70));
    a2 = stat_vec(idx(71:140));
    stat_boot(i) =  abs(mean(a1)-mean(a2));
end
figure;hist(stat_boot)
vline(stat_val,'r')
title(['pval ' num2str(1-sum(stat_val>stat_boot)/length(stat_boot))])
xlabel('OL minus CL1, neural variance')
title('Permutation test');
ylabel('Frequency')
box off

neural_var=neural_var(:);
mahab_dist=mahab_dist(:);


tmp = [mean(acc_imagined_days,1)' mean(acc_online_days,1)' ...
    mean(acc_batch_days,1)'];

decoding_acc = tmp(:);

figure;plot((neural_var),(decoding_acc),'.','MarkerSize',20)
y=decoding_acc;
x= [ones(length(neural_var),1) neural_var];
[B,BINT,R,RINT,STATS] = regress(y,x);STATS
[b,p,b1]=logistic_reg(x(:,2),y);[b p']

figure;plot((mahab_dist),(decoding_acc),'.','MarkerSize',20)
y=decoding_acc;
x= [ones(length(mahab_dist),1) mahab_dist];
[B,BINT,R,RINT,STATS] = regress(y,x);STATS
[b,p,b1]=logistic_reg(x(:,2),y);[b p']


%2D fit
figure;
hold on
col={'b','k','r'};k=1;
data={};
for i=1:11:33
    plot((mahab_dist(i:i+10)),decoding_acc(i:i+10),'.','MarkerSize',20,'color',col{k});
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
plot(xx(:,2),yhat,'Color','k','LineWidth',1);
xlim([15 85])
yticks([0:.05:1])
xlabel('Mahalanobis Distance')
ylabel('Decoder Accuracy')
set(gcf,'Color','w')

fitglm(x(:,2),y,'Distribution','Binomial')

% linear regression
[B,BINT,R,RINT,STATS] = regress(y,x);STATS
lm = fitlm(x(:,2),y)



% doing LOOCV on the logistic regression fit
cv_loss=[];
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
    cv_loss(i) = abs((yhat-ytest));
    %cv_loss(i) = -(ytest*log(yhat) + (1-ytest)*log(1-yhat));
end
cv_loss_stat = cv_loss;

% doing it against a null distribution, 500 times
cv_loss_boot=[];
parfor iter =1:50
    disp(iter)
    cv_loss=[];
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
        cv_loss(i) = abs((yhat-ytest));
        %cv_loss(i) = -(ytest*log(yhat) + (1-ytest)*log(1-yhat));
    end
    cv_loss_boot(iter,:)=cv_loss;
end
figure;
hist(mean(cv_loss_boot,2))
vline(mean(cv_loss_stat))
sum(mean(cv_loss_boot,2) < mean(cv_loss_stat))/length(mean(cv_loss_boot,2))


% 3D plot
% take the log of neural variance
figure;
hold on
col={'r','g','b'};k=1;
data={};
for i=1:11:33
    plot3(mahab_dist(i:i+10),neural_var(i:i+10),(decoding_acc(i:i+10)),'.',...
        'MarkerSize',30,'color',col{k});
    tmp = [mahab_dist(i:i+10) neural_var(i:i+10) (decoding_acc(i:i+10)) ];
    %tmp = [mahab_dist(i:i+10) neural_var(i:i+10) ];
    data{k}=tmp;
    k=k+1;
end
xlabel('Mahalanobis Distance')
ylabel('Neural variance')
zlabel('Decoding Accuracy')


% logistic regression
data_overall = cell2mat(data');
x = data_overall(:,1:2);
y = data_overall(:,3);
[b,p,b1]=logistic_reg(x,y);
mdl = fitglm(x,y,'Distribution','Binomial');
mdl = mdl.Coefficients.Estimate;
bhat = mdl;
% plot as surface
xx = linspace(min(x(:,1)),max(x(:,1)),1e2);
yy = linspace(min(x(:,2)),max(x(:,2)),1e2);
[X,Y]=meshgrid(xx,yy);
zhat = [ones(length(X(:)),1) X(:) Y(:)];
zhat = 1./(1 + exp(-zhat*bhat));
zhat= reshape(zhat,size(X));
%figure;hold on
%grid on
%scatter3(x(:,1),x(:,2),y,'filled')
s=surf(X,Y,zhat,'FaceAlpha',.25);
s.EdgeColor = 'none';
s.FaceColor='cyan';
legend({'Open loop','CL1','CL2','Logistic Fit'})
set(gcf,'Color','w')
grid on
title('Neural Variance and Mahab distance predicts Decoding Acc')


boot=[];
parfor iter=1:5000
    x1=x;
    [bb,bint,r]=regress(x(:,2),[ones(length(x),1) x(:,1)]);
    x1(:,2)=r;
    x1(:,2) = x1(randperm(numel(y)),2);
    %y1=y(randperm(numel(y)));
    out = fitglm(x1,y,'Distribution','Binomial');
    boot = [boot out.Coefficients.Estimate];
end
pval=[];
for i=1:size(boot,1)
    figure;
    hist(abs(boot(i,:)),20);
    vline(abs(mdl(i)));
    pval(i) = sum(abs(boot(i,:)) >= abs(mdl(i)))/ length(boot(i,:));
    title(num2str(pval(i)))
end

% plot surface
xhat = [ones(size(x,1),1) x];
[xx,yy]=meshgrid(min(xhat(:,2)):0.1:max(xhat(:,2)), min(xhat(:,3)):1:max(xhat(:,3)));
yhat_1 = 1./(1+ exp(mdl(1) + mdl(2)*xx + mdl(3)*yy));
figure;
%mesh(xhat(:,2),xhat(:,3),yhat)
mesh(yy,xx,yhat_1)


x=randn(20,1);
y=randn(20,1);
z=2*x+3*y+2*randn(20,1);
[bhat]=regress(z,[ones(size(x,1),1) x y]);
zhat = [ones(size(x,1),1) x y]*bhat;
figure;
[X,Y]=meshgrid(-3:.01:3,-3:.01:3);
zhat = [ones(length(X(:)),1) X(:) Y(:)]*bhat;
zhat= reshape(zhat,size(X));
figure;hold on
scatter3(x,y,z,'filled')
mesh(X,Y,zhat,'FaceAlpha',.5)

% mahalanobis distance
D=zeros(length(data));
for i=1:length(data)
    a = data{i};
    for j=i+1:length(data)
        b = data{j};
        D(i,j) = mahal2(a,b,2);
        D(j,i) = D(i,j);
    end
end

% 2-means cluster index pairwise with swapping of labels
a = data{3};
b = data{2};
stat = two_means_ci(a,b);
% swap labels
boot=[];
d=[a;b];
s = size(a,1);
for i=1:5000
    idx = randperm(length(d));
    tmp = d(idx,:);
    atmp = tmp(1:s,:);
    btmp = tmp(s+1:end,:);
    boot(i) = two_means_ci(atmp,btmp);
end
figure;hist(boot)
vline(stat)
sum(stat>=boot)/length(boot)

% 2-means cluster index pairwise and null hypothesis testing for the
% two-means cluster index using gaussian distribution
K=zeros(length(data));
P=zeros(length(data));
D=zeros(length(data));
P_d=zeros(length(data));
for i=1:length(data)
    a = data{i};
    for j=i+1:length(data)
        b = data{j};
        if j==3
            b=b(2:end,:);
        end


        % 2 means ci
        K(i,j) = two_means_ci(a,b);
        K(j,i) = K(i,j);
        stat = K(i,j);

        % null testing for each pairwise distance
        % build a common distribution from the two datasets
        a1=a';b1=b';
        s1 = size(a1,2);
        c1 = [a1 b1];
        m = mean(c1,2);
        X = cov(c1');
        C12 = chol(X);
        dboot=[];
        parfor iter=1:5000
            g = randn(size(c1));
            cnew = m + C12'*g;
            % find two clusters in the data
            idx = kmeans(cnew', 2);
            atmp = cnew(:,find(idx==1));
            btmp = cnew(:,find(idx==2));
            dboot(iter) =  two_means_ci(atmp',btmp');
        end
        P(i,j) = 1-sum(dboot>stat)/length(dboot);

        % mahab dist
        D(i,j) = mahal2(a,b,2);
        D(j,i) = D(i,j);
        stat = D(i,j);

        % null testing for each pairwise distance
        % build a common distribution from the two datasets
        a1=a';b1=b';
        s1 = size(a1,2);
        c1 = [a1 b1];
        m = mean(c1,2);
        X = cov(c1');
        C12 = chol(X);
        dboot=[];
        parfor iter=1:5000
            g = randn(size(c1));
            cnew = m + C12'*g;
            % find two clusters in the data
            idx = kmeans(cnew', 2);
            atmp = cnew(:,find(idx==1));
            btmp = cnew(:,find(idx==2));
            dboot(iter) =  mahal2(atmp',btmp',2);
        end
        P_d(i,j) = sum(dboot>stat)/length(dboot);
    end
end


% using LDA on random split
a =data{1};
b = data{3};
d = [a;b];
idx = [0*ones(size(a,1),1);ones(size(a,1),1)];idx_main=idx;
acc=[];
res_acc=[];pval_acc=[];
for iter=1:25
    % randomly select 18 for training and 4 for testing
    idx = idx_main(randperm(numel(idx_main)));
    idx_train = randperm(size(d,1),16);
    I =  ones(size(d,1),1);
    I(idx_train)=0;
    idx_test = find(I==1);

    % train the LDA
    data_train = d(idx_train,:);
    idx_train = idx(idx_train);
    W = LDA(data_train,idx_train);

    % apply on held out data
    data_test = d(idx_test,:);
    data_test = [ones(size(data_test,1),1) data_test];
    idx_test = idx(idx_test);
    L = data_test * W';
    P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);
    [aa,bb]=max(P');bb=bb-1;
    acc(iter) = sum(idx_test==bb')/length(bb);

    % balanced accuracy
    tp=0;tn=0;fp=0;fn=0;
    p = idx_test;grp_test=bb';
    for ii=1:length(p)
        if p(ii)==1 && grp_test(ii)==1
            tp=tp+1;
        end

        if p(ii)==0 && grp_test(ii)==0
            tn=tn+1;
        end

        if p(ii)==1 && grp_test(ii)==0
            fn=fn+1;
        end

        if p(ii)==0 && grp_test(ii)==1
            fp=fp+1;
        end
    end
    res_acc(iter) = 0.5* ( tp/(tp+fn) + tn/(tn+fp) );

    % stats
    alp1=1+tp;
    bet1=1+fn;
    alp2=1+tn;
    bet2=1+fp;
    res=0.001;
    u=0:res:1;
    a=betapdf(u,alp1,bet1);
    b=betapdf(u,alp2,bet2);
    x=conv(a,b);
    z=2*x(1:2:end);
    z=z/(sum(x*res));
    % figure;plot(u,z);hold on;plot(u,a,'k');plot(u,b,'r')
    % calculate p-value
    querypt= 0.5;
    I=(u>querypt);
    pval(iter)=1-sum(z(I)*res);
end
figure;boxplot(bootstrp(1000,@mean,acc))
acc=mean(acc)



% using LDA on LOOCV
a =data{2};
b = data{3};
d = [a;b];
idx = [0*ones(size(a,1),1);ones(size(a,1),1)];idx_main=idx;
acc=[];
res_acc=[];pval_acc=[];
for i=1:length(d)
    idx_test=i;
    I =  ones(size(d,1),1);
    I(idx_test)=0;
    idx_train = find(I==1);

    % train the LDA
    data_train = d(idx_train,:);
    idx_train = idx(idx_train);
    W = LDA(data_train,idx_train);

    % apply on held out data
    data_test = d(idx_test,:);
    data_test = [ones(size(data_test,1),1) data_test];
    idx_test = idx(idx_test);
    L = data_test * W';
    P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);
    [aa,bb]=max(P');bb=bb-1;
    acc(i) = sum(idx_test==bb);
end
acc=mean(acc)




% stats on the latent space variance
tmp=[0.000937406	0.0985546	0.0289443
    0.00154575	0.0315809	0.0753841
    0.00141096	0.0103554	0.20185
    0.00043903	0.0643351	0.150029
    0.00125143	0.00337939	0.112707
    0.00113499	0.00377055	0.112707
    ];
[P,ANOVATAB,STATS] = anova1(tmp);

[h p tb st] = ttest(tmp(:,1),tmp(:,2));p
[h p tb st] = ttest(tmp(:,1),tmp(:,3));p
[h p tb st] = ttest(tmp(:,2),tmp(:,3));p

% how about bootstrapped difference of means?
stat  =  mean(tmp(:,2)) - mean(tmp(:,1));
c= [tmp(:,2);tmp(:,1)];
l=size(tmp,1);
boot=[];
c=c-mean(c);
for i=1:5000
    idx=randperm(numel(c));
    a1 = c(idx(1:l));
    b1 = c(idx(l+1:end));
    boot(i) = mean(a1)-mean(b1);
end
figure;hist((boot))
vline((stat),'r')
sum(boot>stat)/length(boot)


% get the accuracies relative to imagined movement within that day
a0 = mean(acc_imagined_days(:,1:end),1);
a1 = mean(acc_online_days(:,1:end),1);
a2 = mean(acc_batch_days(:,1:end),1);
figure;
plot(a0);
hold on
plot(a1);
plot(a2)
ylim([0 1])

a1 = (a1-a0)%./a0;
a2 = (a2-a0)%./a0;
figure;boxplot([a1' a2'])
hline(0)

b3_acc_rel_imagined = [a1' a2'];

acc_rel_imagined = [b1_acc_rel_imagined_prop ;b3_acc_rel_imagined];
figure;
boxplot(acc_rel_imagined*100);
a = get(get(gca,'children'),'children');   % Get the handles of all the objects
t = get(a,'tag');   % List the names of all the objects
box1 = a(5);   % The 7th object is the first box
set(box1, 'Color', 'k');   % Set the color of the first box to green
box1 = a(6);   % The 7th object is the first box
set(box1, 'Color', 'k');   % Set the color of the first box to green
line1 = a(3);   % The 7th object is the first box
set(line1, 'Color', 'm');   % Set the color of the first box to green
line1 = a(4);   % The 7th object is the first box
set(line1, 'Color', 'm');   % Set the color of the first box to green
hline(0,'r')
ylabel('Accuracy Improvements (%)')
xticks(1:2)
xticklabels({'CL1','CL2'})
set(gca,'FontSize',14)
set(gcf,'Color','w')
box off



% plotting scatter plot with mean and SE
m1 = acc_rel_imagined(:,1);
m1b = sort(bootstrp(1000,@mean,m1));
m2 = acc_rel_imagined(:,2);
m2b = sort(bootstrp(1000,@mean,m2));
y=[mean(acc_rel_imagined)];
s1=std(m1)/sqrt(length(m1));
s2=std(m2)/sqrt(length(m2));
neg = [s1 s2 ];
pos= [s1 s2 ];
neg = [y(1)-m1b(25) y(2)-m2b(25) ];
pos = [m1b(975)-y(1) m2b(975)-y(2) ];
figure;
hold on
% now the scatter part
%B1
hold on
scatter(ones(10,1)+0.07*randn(10,1),b1_acc_rel_imagined_prop(:,1),'or')
scatter(2*ones(10,1)+0.07*randn(10,1),b1_acc_rel_imagined_prop(:,2),'r')
%B3
scatter(ones(size(b3_acc_rel_imagined,1),1)+...
    0.07*randn(size(b3_acc_rel_imagined,1),1),b3_acc_rel_imagined(:,1),'ob')
scatter(2*ones(size(b3_acc_rel_imagined,1),1)+...
    0.07*randn(size(b3_acc_rel_imagined,1),1),b3_acc_rel_imagined(:,2),'ob')


cmap = brewermap(6,'Blues');
x=1:2;
for i=1:length(y)
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',10,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([0.5 2.5])
hline(0,'--k')
xticks([1 2])
xticklabels({'CL1','CL2'})
ylim([-0.2 0.6])
yticks([-.2:.2:.6])
set(gcf,'Color','w')


[h p tb st] = ttest(acc_rel_imagined(:,1),acc_rel_imagined(:,2))
[P,H,STATS] = signrank(acc_rel_imagined(:,2))
[P,H,STATS] = ranksum(acc_rel_imagined(:,1),acc_rel_imagined(:,2))

% saving all data
save acc_relative_imagined_prop_B1B3 -v7.3

% using a mixed effect model
acc_improv=[];
subject=[];
exp_type=[];
%b1
for i=1:size(b1_acc_rel_imagined_prop,2)
    tmp=b1_acc_rel_imagined_prop(:,i);
    acc_improv =[acc_improv;tmp];
    exp_type = [exp_type;i*ones(size(tmp))];
    subject=[subject;1*ones(size(tmp))];
end
%b2
for i=1:size(b3_acc_rel_imagined,2)
    tmp=b3_acc_rel_imagined(:,i);
    %     if i==1
    %         tmp=tmp([1 3 4]);
    %     end
    acc_improv =[acc_improv;tmp];
    exp_type = [exp_type;i*ones(size(tmp))];
    subject=[subject;2*ones(size(tmp))];
end

data=table(acc_improv,exp_type,subject);
glme = fitglme(data,'acc_improv ~ 1+ exp_type + (1|subject)')

%% (MAIN) COMBINING AND PLOTTING DECODING ACC and SIG FOR B1 AND B3 OL -> CL1, CL2

clc;clear;close all
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
addpath('C:\Users\nikic\Documents\MATLAB\limo_v1.4')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools')
addpath('C:\Users\nikic\Documents\GitHub\limo_tools\limo_cluster_functions')

% load B3 data
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
%a=load('hDOF_11days_accuracy_results_B3_v2');
%a=load('hDOF_11days_accuracy_results_B3_v4');
%a=load('hDOF_11days_accuracy_results_B3_V5');
a=load('hDOF_11days_accuracy_results_B3_V6');

% load B1 data
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%b=load('hDOF_10days_accuracy_results_New');
%b=load('hDOF_10days_accuracy_results_New_New');
%b=load('hDOF_10days_accuracy_results_New_New_v2');
b=load('hDOF_10days_accuracy_results_New_New_v3');


%%%%%% plotting the significance for decoding accuracies , B3
tmp = a.binomial_res_chance;
tmp = squeeze(cell2mat(struct2cell(tmp)))';
[pfdr,pval]=fdr(tmp,0.05);pfdr
(sum(tmp(:)<=pfdr))/length(tmp(:))
pfdr = log(pfdr);
tmp = log(tmp);
idx= (1:3) + randn(size(tmp))*0.1;
figure;
hold on
scatter(idx,tmp)
hline(pfdr,'--r')
set(gcf,'Color','w')
ylabel('Log Probability')
xticks(1:3)
xticklabels({'OL','CL1','CL2'})
xlim([.5 3.5])
ylim([.5 3.5])


%%%%%% plotting the significance for decoding accuracies , B1
tmp = b.binomial_res_chance;
tmp = squeeze(cell2mat(struct2cell(tmp)))';
[pfdr,pval]=fdr(tmp,0.05);pfdr
(sum(tmp(:)<=pfdr))/length(tmp(:))
pfdr = log(pfdr);
tmp = log(tmp);
idx= (1:3) + randn(size(tmp))*0.1;
figure;
hold on
scatter(idx,tmp)
hline(pfdr,'--r')
set(gcf,'Color','w')
ylabel('Log Probability')
xticks(1:3)
xticklabels({'OL','CL1','CL2'})
xlim([.5 3.5])
ylim([-85 5])

%%%%%%% plotting the confusion matrices, B1
acc_imagined=b.acc_imagined_days;acc_imagined=squeeze(mean(acc_imagined,3));
acc_online=b.acc_online_days;acc_online=squeeze(mean(acc_online,3));
acc_batch=b.acc_batch_days;acc_batch=squeeze(mean(acc_batch,3));
figure;
subplot(1,3,1)
imagesc(acc_imagined*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_imagined,1)
    for k=1:size(acc_imagined,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
title(['OL Acc of ' num2str(100*mean(diag(acc_imagined)))])

subplot(1,3,2)
imagesc(acc_online*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_online,1)
    for k=1:size(acc_online,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
title(['CL1 Acc of ' num2str(100*mean(diag(acc_online)))])

subplot(1,3,3)
imagesc(acc_batch*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_batch,1)
    for k=1:size(acc_batch,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
title(['CL2 Acc of ' num2str(100*mean(diag(acc_batch)))])


%%%%%5% plotting the confusion matrices, B3
acc_imagined=a.acc_imagined_days;acc_imagined=squeeze(mean(acc_imagined,3));
acc_online=a.acc_online_days;acc_online=squeeze(mean(acc_online,3));
acc_batch=a.acc_batch_days;acc_batch=squeeze(mean(acc_batch,3));
figure;
subplot(1,3,1)
imagesc(acc_imagined*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_imagined,1)
    for k=1:size(acc_imagined,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
title(['OL Acc of ' num2str(100*mean(diag(acc_imagined)))])

subplot(1,3,2)
imagesc(acc_online*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_online,1)
    for k=1:size(acc_online,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
title(['CL1 Acc of ' num2str(100*mean(diag(acc_online)))])

subplot(1,3,3)
imagesc(acc_batch*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_batch,1)
    for k=1:size(acc_batch,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
yticklabels({'Rt Thumb','Leg','Lt. Thumb','Head','Tong','Lips','Both middle'})
title(['CL2 Acc of ' num2str(100*mean(diag(acc_batch)))])


%%%%%%% GETTING THE DECODING ACCURACIES MEAN FOR OTHER ANALYSES

a1=[];
for i=1:size(a.acc_imagined_days,3)
    x = squeeze(a.acc_imagined_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
a.acc_imagined_days = a1;
a1=[];
for i=1:size(b.acc_imagined_days,3)
    x = squeeze(b.acc_imagined_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
b.acc_imagined_days = a1;

a1=[];
for i=1:size(a.acc_online_days,3)
    x = squeeze(a.acc_online_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
a.acc_online_days = a1;
a1=[];
for i=1:size(b.acc_online_days,3)
    x = squeeze(b.acc_online_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
b.acc_online_days = a1;

a1=[];
for i=1:size(a.acc_batch_days,3)
    x = squeeze(a.acc_batch_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
a.acc_batch_days = a1;
a1=[];
for i=1:size(b.acc_batch_days,3)
    x = squeeze(b.acc_batch_days(:,:,i));
    x = diag(x);
    a1 = [a1 x];
end
b.acc_batch_days = a1;

% get the data into variables
acc_imagined_days= [a.acc_imagined_days';b.acc_imagined_days']';
acc_online_days= [a.acc_online_days';b.acc_online_days']';
acc_batch_days= [a.acc_batch_days';b.acc_batch_days']';
idx = [ones(size(a.acc_imagined_days',1),1);2*ones(size(b.acc_imagined_days',1),1)];

% plot the data
%m1 = (acc_imagined_days(:));
m11 = mean(acc_imagined_days,1);
m1b = sort(bootstrp(1000,@mean,m11));
%m2 = (acc_online_days(:));
m22 = mean(acc_online_days,1);
m2b = sort(bootstrp(1000,@mean,m22));
%m3 = (acc_batch_days(:));
m33 = mean(acc_batch_days,1);
m3b = sort(bootstrp(1000,@mean,m33));
x=1:3;
y=[mean(m11) mean(m22) mean(m33)];
% scatter B1 and B3 individually
figure; hold on
%boxplot([m11' m22' m33']);
%box off
%a = get(get(gca,'children'),'children');
%for i=1:length(a)
%    box1 = a(i);
%    set(box1, 'Color', 'k');
%end
h=hline(median(m11),'k');
h.LineWidth=3;
h.XData = [0.75 1.25];
h=hline(median(m22),'k');
h.LineWidth=3;
h.XData = [1.75 2.25];
h=hline(median(m33),'k');
h.LineWidth=3;
h.XData = [2.75 3.25];
aa = find(idx==1);
x=(1:3) + 0.1*randn(length(aa),3);
h=scatter(x,[m11(aa)' m22(aa)' m33(aa)'],'filled');
for i=1:3
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.3;
end
aa = find(idx==2);
x=(1:3) + 0.1*randn(length(aa),3);
h=scatter(x,[m11(aa)' m22(aa)' m33(aa)'],'filled');
for i=1:3
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end
ylim([0 1])
yticks([0:.1:1])
h=hline(1/7);
set(h,'LineWidth',1)
xlim([.5 3.5])
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
ylabel('Decoding Accuracy')
yticks([0:.1:1])
ylim([.5 1])


% Signed rank test on b1
[P,H,STATS] = signrank(mean(acc_batch_days(:,12:end),1),mean(acc_online_days(:,12:end),1))
[P,H,STATS] = signrank(mean(acc_batch_days(:,12:end),1),mean(acc_imagined_days(:,12:end),1))
[P,H,STATS] = signrank(mean(acc_imagined_days(:,12:end),1),mean(acc_online_days(:,12:end),1))

[p bootstrp_tvalues tvalue]=bootstrap_ttest(mean(acc_online_days(:,12:end),1),...
    mean(acc_batch_days(:,12:end),1),1,2e4);p

x1 = mean(acc_online_days(:,12:end),1);
y1 = mean(acc_batch_days(:,12:end),1);

[p,h,stats] = ttest(y1-x1)


% Signed rank test on b3
[P,H,STATS] = signrank(mean(acc_batch_days(:,1:11),1),mean(acc_online_days(:,1:11),1))
[P,H,STATS] = signrank(mean(acc_batch_days(:,1:11),1),mean(acc_imagined_days(:,1:11),1))
[P,H,STATS] = signrank(mean(acc_imagined_days(:,1:11),1),mean(acc_online_days(:,1:11),1))

[p bootstrp_tvalues tvalue]=bootstrap_ttest(mean(acc_online_days(:,1:11),1),...
    mean(acc_imagined_days(:,1:11),1),1,2e4);p





% Signed rank test on all
[P,H,STATS] = signrank(mean(acc_batch_days,1),mean(acc_online_days,1));P
[P,H,STATS] = signrank(mean(acc_batch_days,1),mean(acc_imagined_days,1));P
[P,H,STATS] = signrank(mean(acc_imagined_days,1),mean(acc_online_days,1));P

num_trials_imagined = [b.num_trials_imag a.num_trials_imag];
num_trials_online = [b.num_trials_online a.num_trials_online];
num_trials_batch = [b.num_trials_batch a.num_trials_batch];

tmp = sort(bootstrp(1000,@mean,num_trials_imagined));
round([tmp(25) mean(tmp) tmp(975)])

tmp = sort(bootstrp(1000,@mean,num_trials_batch));
round([tmp(25) mean(tmp) tmp(975)])

%%%% improvement in decoding accuracy relative to OL
OL = mean(acc_imagined_days,1);
CL1  = mean(acc_online_days,1);
CL2  = mean(acc_batch_days,1);

CL1 = 100*((CL1-OL)./OL);
CL2 = 100*((CL2-OL)./OL);

[median(CL1(1:11)) median(CL2(1:11))]
[median(CL1(12:end)) median(CL2(12:end))]

figure;
hold on
hh=hline(mean(CL1),'k');
hh.LineWidth=3;
hh.XData = [0.75 1.25];
hh=hline(mean(CL2),'k');
hh.LineWidth=3;
hh.XData = [1.75 2.25];
aa = find(idx==1); % this is B3
x=(1:2) + 0.1*randn(length(aa),2);
h=scatter(x,[CL1(aa)' CL2(aa)'],'filled');
for i=1:2
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end
aa = find(idx==2); % this is B1
x=(1:2) + 0.1*randn(length(aa),2);
hh=scatter(x,[CL1(aa)' CL2(aa)'],'filled');
for i=1:3
    hh(i).MarkerFaceColor = 'b';
    hh(i).MarkerFaceAlpha = 0.3;
end
ylim([-10 60])
yticks([-10:10:60])
xlim([.5 2.5])
xticks(1:2)
xticklabels({'CL1','CL2'})
h=hline(0,'--k');
set(gcf,'Color','w')
set(gca,'LineWidth',1)
ylabel('Improvements in Decoding Acc.')

%[P,H,STATS] = signrank(CL1(1:11),CL2(1:11))
%[P,H,STATS] = signrank(CL1(12:end),CL2(12:end))
[P,H,STATS] = signrank(CL1,CL2)


%%%% mixed effect models for the improvement in CL1 and CL2 vs OL
subj=[];
decoding_impr=[];
mvmt_type=[];
subj=[ones(11,1);2*ones(10,1)];
subj=[subj;subj];
decoding_impr = [CL1';CL2'];
mvmt_type=[ones(21,1);2*ones(21,1)];
data = table(subj,mvmt_type,decoding_impr);


%glm = fitglme(data,'decoding_impr ~ mvmt_type + (1|subj)','Distribution','Binomial')


glm = fitlme(data,'decoding_impr ~ mvmt_type + (1|subj)')
stat = glm.Coefficients.tStat(2);

stat_boot=[];

idx1 = find(subj==1);
idx2 = find(subj==2);
mvmt1 = mvmt_type(idx1);
mvmt2 = mvmt_type(idx2);

parfor i=1:1000
  

    disp(i)
    mvmt_type_tmp = mvmt_type;
    mvmt_type1_tmp = mvmt1(randperm(numel(mvmt1)));
    mvmt_type2_tmp = mvmt2(randperm(numel(mvmt2)));

    mvmt_type_tmp(idx1) = mvmt_type1_tmp;
    mvmt_type_tmp(idx2) = mvmt_type2_tmp;


    %mvmt_type_tmp = mvmt_type(randperm(numel(mvmt_type)));
    data_tmp = table(subj,mvmt_type_tmp,decoding_impr);
    glm_tmp = fitlme(data_tmp,'decoding_impr ~ mvmt_type_tmp + (1|subj)');
    stat_boot(i) = glm_tmp.Coefficients.tStat(2);
end
figure;
hist((stat_boot))
vline((stat))
sum(abs(stat_boot)>abs(stat))/length(stat_boot)
sum((stat_boot)>(stat))/length(stat_boot)

%%%%% mixed effect model to just see improvement in CL1 and CL2 individuall
%%%%% relative to OL

subj=[];
decoding_impr=[];
mvmt_type=[];
subj=[ones(11,1);2*ones(10,1)];
subj=[subj;subj];
decoding_impr = [CL1';CL2'];
mvmt_type=[ones(21,1);2*ones(21,1)];
data = table(subj,mvmt_type,decoding_impr);
data1 = data(1:21,:);
data2 = data(22:end,:);

% CL1
glm = fitlme(data1,'decoding_impr ~ 1 + (1|subj)')
stat = glm.Coefficients.tStat(1);
stat_boot=[];
xx = table2array(data1);
xx(:,3) = xx(:,3)-mean(xx(:,3));
parfor i=1:1000
    disp(i)
    idx = randi(length(xx),1,length(xx));
    xx1 = xx(idx,:);
    subj_tmp = xx1(:,1);
    mvmt_tmp = xx1(:,2);
    decoding_tmp = xx1(:,3);
    data1_tmp = table(subj_tmp,mvmt_tmp,decoding_tmp);    
    glm_tmp = fitlme(data1_tmp,'decoding_tmp ~ 1 + (1|subj_tmp)');
    stat_boot(i) = glm_tmp.Coefficients.tStat(1);
end
figure;
hist(stat_boot)
vline(stat)
%sum(stat_boot>stat)/length(stat_boot)
sum(abs(stat_boot)>abs(stat))/length(abs(stat_boot))

%CL2
glm = fitlme(data2,'decoding_impr ~ 1 + (1|subj)')
stat = glm.Coefficients.tStat(1);
stat_boot=[];
xx = table2array(data2);
xx(:,3) = xx(:,3)-mean(xx(:,3));
parfor i=1:1000
    disp(i)
    idx = randi(length(xx),1,length(xx));
    xx1 = xx(idx,:);
    subj_tmp = xx1(:,1);
    mvmt_tmp = xx1(:,2);
    decoding_tmp = xx1(:,3);
    data1_tmp = table(subj_tmp,mvmt_tmp,decoding_tmp);    
    glm_tmp = fitlme(data1_tmp,'decoding_tmp ~ 1 + (1|subj_tmp)');
    stat_boot(i) = glm_tmp.Coefficients.tStat(1);
end
figure;
hist(stat_boot)
vline(stat)
%sum(stat_boot>stat)/length(stat_boot)
sum(abs(stat_boot)>abs(stat))/length(abs(stat_boot))



%%%%% IMPORTANT %%%%%
%%%%%%%% USING NON PARAMETRIC LINEAR MIXED EFFECT MODEL ON DECODING
%%%%%%%% ACCURACIES DIRECTLY. 
decoding_acc=[];
subj=[];
mvmt_type=[];
tmp=mean(acc_imagined_days,1);
tmp1=mean(acc_online_days,1);
for i=1:length(tmp)
    if i<=11
        subj =[subj;1];
    else
        subj =[subj;2];
    end
    mvmt_type=[mvmt_type;1];
    decoding_acc= [decoding_acc;tmp(i)];
end
for i=1:length(tmp1)
    if i<=11
        subj =[subj;1];
    else
        subj =[subj;2];
    end
    mvmt_type=[mvmt_type;2];
    decoding_acc= [decoding_acc;tmp1(i)];
end

data = table(subj,mvmt_type,decoding_acc);
glm = fitglme(data,'decoding_acc ~ 1 + mvmt_type + (1|subj)')
%glm = fitlm(data,'mahab_dist ~ 1+ day_name');
stat = glm.Coefficients.tStat(2);

stat_boot=[];
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
    data_tmp = table(subj,mvmt_type_tmp,decoding_acc);
    glm_tmp = fitlme(data_tmp,'decoding_acc ~ mvmt_type_tmp + (1|subj)');
    stat_boot(i) = glm_tmp.Coefficients.tStat(2);
end

figure;hist(stat_boot)
vline(stat)
%sum(stat_boot>stat)/length(stat_boot)
sum(abs(stat_boot)>abs(stat))/length(stat_boot)


%%%%% plotting the decoding accuracies
%B3
y2=mean(acc_batch_days,1);y2=y2(1:11);
y1=mean(acc_online_days,1);y1=y1(1:11);
y=mean(acc_imagined_days,1);y=y(1:11);
x=1:length(y);
x=[ones(length(x),1) x(:)];
figure;
hold on
%[B,BINT,R,RINT,STATS] = regress(y',x);
lm=fitlm(x(:,2:end),y,'Robust','off');
B=lm.Coefficients.Estimate;
plot(x(:,2),y,'.k','MarkerSize',20)
yhat = x*B;
plot(x(:,2),yhat,'k','LineWidth',1);
%[B,BINT,R,RINT,STATS1] = regress(y1',x);
lm1=fitlm(x(:,2:end),y1,'Robust','off');
B=lm1.Coefficients.Estimate;
plot(x(:,2),y1,'.b','MarkerSize',20)
yhat = x*B;
plot(x(:,2),yhat,'b','LineWidth',1);
%[B,BINT,R,RINT,STATS2] = regress(y2',x);
lm2=fitlm(x(:,2:end),y2,'Robust','off');
B=lm2.Coefficients.Estimate;
plot(x(:,2),y2,'.r','MarkerSize',20)
yhat = x*B;
plot(x(:,2),yhat,'r','LineWidth',1);
ylim([0 1])
[lm.Coefficients.pValue lm1.Coefficients.pValue lm2.Coefficients.pValue ]
% [b,p,b1]=logistic_reg(x(:,2),y);p
% [b,p,b1]=logistic_reg(x(:,2),y1);p
% [b,p,b1]=logistic_reg(x(:,2),y2);p
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xlabel('Days')
ylabel('Decoding Accuracy')
title('B3')
xticks(1:11)
xlim([0.5 11.5])

%B1
y2=mean(acc_batch_days,1);y2=y2(12:end);
y1=mean(acc_online_days,1);y1=y1(12:end);
y=mean(acc_imagined_days,1);y=y(12:end);
x=1:length(y);
x=[ones(length(x),1) x(:)];
figure;
hold on
%[B,BINT,R,RINT,STATS] = regress(y',x);
lm=fitlm(x(:,2:end),y,'Robust','off');
B=lm.Coefficients.Estimate;
plot(x(:,2),y,'.k','MarkerSize',20)
yhat = x*B;
plot(x(:,2),yhat,'k','LineWidth',1);
%[B,BINT,R,RINT,STATS1] = regress(y1',x);
lm1=fitlm(x(:,2:end),y1,'Robust','off');
B=lm1.Coefficients.Estimate;
plot(x(:,2),y1,'.b','MarkerSize',20)
yhat = x*B;
plot(x(:,2),yhat,'b','LineWidth',1);
%[B,BINT,R,RINT,STATS2] = regress(y2',x);
lm2=fitlm(x(:,2:end),y2,'Robust','off');
B=lm2.Coefficients.Estimate;
plot(x(:,2),y2,'.r','MarkerSize',20)
yhat = x*B;
plot(x(:,2),yhat,'r','LineWidth',1);
ylim([0 1])
[lm.Coefficients.pValue lm1.Coefficients.pValue lm2.Coefficients.pValue ]
% [b,p,b1]=logistic_reg(x(:,2),y);p
% [b,p,b1]=logistic_reg(x(:,2),y1);p
% [b,p,b1]=logistic_reg(x(:,2),y2);p
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xlabel('Days')
ylabel('Decoding Accuracy')
title('B1')
xticks(1:10)
xlim([0.5 10.5])





