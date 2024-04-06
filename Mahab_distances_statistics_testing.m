
% COME HERE FROM THE REPRESENTATIONAL STRUCURE ANALYSES CODE

%% SIMULATION OF THE TESTING SCHEME
% create two datasets 

a= randn(200,2);
b=randn(200,2)+[-2.25,-2.25];
figure;plot(a(:,1),a(:,2),'.','MarkerSize',20)
hold on
plot(b(:,1),b(:,2),'.r','MarkerSize',20)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xlabel('Feature 1')
ylabel('Feature 2')
legend({'Mvmt 1','Mvmt 2'})
box on
% compute mahal distance
dstat = mahal2(a,b,2);
title(['Mahalanobis distance of ' num2str(dstat)])

a=a';b=b';
s1 = size(a,2);
c = [a b];
m = mean(c,2);
X = cov(c');
C12 = chol(X);

% plot the ellipse
[v,d]=eigs(X); % ellipse length is sqrt(e.value) in direction of e.vector
d=diag(d);
ra=sqrt(5.991*d(1));
rb=sqrt(5.991*d(2));
v1=v(:,1);
ang = atan(v1(2)/v1(1));
c0=mean(c,2);
xlim([min(c(1,:))-1 max(c(1,:))+1])
ylim([min(c(2,:))-1 max(c(2,:))+1])
figure;plot(a(1,:),a(2,:),'.','MarkerSize',20)
hold on
plot(b(1,:),b(2,:),'.r','MarkerSize',20)
ellipse(ra,rb,ang,c0(1),c0(2),'k');
title(['Gaussian fit to overall data'])
xlim([min(c(1,:))-1 max(c(1,:))+1])
ylim([min(c(2,:))-1 max(c(2,:))+1])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xlabel('Feature 1')
ylabel('Feature 2')
legend({'Mvmt 1','Mvmt 2'})

% example plotting
g = randn(size(c));
cnew = m + C12'*g;

% just plot it
figure;hold on
plot(cnew(1,:),cnew(2,:),'.k','MarkerSize',20)
ellipse(ra,rb,ang,c0(1),c0(2),'k');
xlim([min(c(1,:))-1 max(c(1,:))+1])
ylim([min(c(2,:))-1 max(c(2,:))+1])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xlabel('Feature 1')
ylabel('Feature 2')
legend('Sampling from Gaussian Fit')
box on


% find two clusters in the data
idx = kmeans(cnew', 2);
atmp = cnew(:,find(idx==1));
btmp = cnew(:,find(idx==2));
figure;plot(atmp(1,:),atmp(2,:),'.','MarkerSize',20)
hold on
plot(btmp(1,:),btmp(2,:),'.r','MarkerSize',20)
ellipse(ra,rb,ang,c0(1),c0(2),'k');
dd= mahal2(atmp',btmp',2);
title(['K-means parcellation, Mahalanobis dist of ' num2str(dd)])
xlim([min(c(1,:))-1 max(c(1,:))+1])
ylim([min(c(2,:))-1 max(c(2,:))+1])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xlabel('Feature 1')
ylabel('Feature 2')
legend({'Cluster 1','Cluster 2'})
box on

parfor i=1:500
    g = randn(size(c));
    cnew = m + C12'*g;

    % find two clusters in the data
    idx = kmeans(cnew', 2);
    atmp = cnew(:,find(idx==1));
    btmp = cnew(:,find(idx==2));
    dboot(i) = mahal2(atmp',btmp',2);

end
figure;hist(dboot)
vline(dstat)
sum(dboot>dstat)/length(dboot)



%%
%%%% null hypothesis testing -> permutation of each freq feature mvmt 1 and
% mvmt2. Create a cmmon distribution, sample from the distribution and use
% k-means to get two clusters. then find the mahab distance between the two
% clusters
%%% using all features
a = Data{13};
b = Data{14};
d = mahal2(a',b',2);
dboot=[];
s1 = size(a,2);
c = [a b];
m = mean(c,2);
X = cov(c');
C12 = chol(X);
parfor i=1:500
    g = randn(size(c));
    cnew = m + C12'*g;

    % find two clusters in the data
    idx = kmeans(cnew', 2);
    atmp = cnew(:,find(idx==1));
    btmp = cnew(:,find(idx==2));
    dboot(i) = mahal2(atmp',btmp',2);
end
figure;hist(dboot)
vline(d)
sum(dboot>d)/length(dboot)

%% SAME AS ABOVE BUT NOW FOR ALL PAIRWISE MOVEMENTS

% parallel cluster
clus = parcluster;
clus.NumWorkers = 18;
par_clus = clus.parpool(18);
load('/media/reza/ResearchDrive/B3 Data for ERP Analysis/Data_B3_ForMahabStats.mat')
addpath(genpath('/home/reza/Repositories/ECoG_BCI_HighDim'))
tic
options = statset('UseParallel',true);
D_p=zeros(length(Data));
D_boot=[];
for i=1:length(Data)
    disp(i)
    a = Data{i};
    for j=i+1:length(Data)
        b = Data{j};
        d = mahal2(a',b',2);
        dboot=[];
        s1 = size(a,2);
        c = [a b];
        m = mean(c,2);
        X = cov(c');
        C12 = chol(X);
        parfor ii=1:750
            g = randn(size(c));
            cnew = m + C12'*g;

            % find two clusters in the data
            idx = kmeans(cnew', 2);
            atmp = cnew(:,find(idx==1));
            btmp = cnew(:,find(idx==2));

            % get mahab distance
            dboot(ii) = mahal2(atmp',btmp',2);
        end
        D_boot = [D_boot;dboot];
        D_p(i,j) = sum(dboot>d)/length(dboot);        
    end
end


% for B3
ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Rotate Left Wrist','Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Knee','Left Knee',...%
    'Right Ankle','Left Ankle',...
    'Lips','Tongue'};

D=zeros(length(ImaginedMvmt));
for i=1:length(Data)
    %disp(['Processing Mvmt ' num2str(i) ...
    %    ' with bins till '  num2str(bins_size(end))])
    A = Data{i}';
    for j=i+1:length(Data)
        B = Data{j}';
        d = mahal2(A,B,2);
        D(i,j)=d;
        D(j,i)=d;
    end
end

Dtmp=squareform(D);
Db = D_boot(:);
Dtmp(end+1:length(Db))=NaN;
figure;boxplot(log10([Dtmp(:) Db(:)]),'Whisker',3)
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Real Data','Null distribution'})
ylabel('Log Distance')
title('B1')
set(gca,'FontSize',12)
box off
set(gca,'LineWidth',1)

[pfdr,pval]=fdr(squareform(D_p')',0.05);pfdr
sum(squareform(D_p')<=pfdr)/length(squareform(D_p'))

%cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%save pairwise_Mahab_Dist_Stats_B1 -v7.3


clear Data
cd('/media/reza/ResearchDrive/B3 Data for ERP Analysis/')
save pairwise_Mahab_Dist_Stats_B3 -v7.3

%clear Data Data_bkup TrialData
%cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
%save pairwise_Mahab_Dist_Stats_B3 -v7.3
toc

%% Going down the dendrogram , simulation
a = rand(18,2);
z=linkage(a);
figure;dendrogram(z)
z(:,end+1) = ((size(a,1)+1):(max(max(z(:,1:2)))+1))';

root_idx = z(end-5,1:2);
max_datapts = size(a,1);
children_all={};
for i=1:length(root_idx)
    children=[];
    % first step is to get the immediate below neighbors
    if root_idx(i)<=max_datapts
        children = [children root_idx(i)];
    end
    parents = z(find(z(:,end)==root_idx(i)),1:2);
    children = [children parents(parents<=max_datapts)];
    going_down= true;
    while going_down
        parents = get_children(z,parents)  ;
        children = [children parents(parents<=max_datapts)];
        parents = parents(parents>max_datapts);
        if all(parents<=max_datapts) || isempty(parents)
            going_down=false;
        end
    end
    children_all{i} = children(children<=max_datapts);
end

%% going down the dendogram and doing null hypothesis testing each node
% real data

Z(:,end+1) = ((size(D,1)+1):(max(max(Z(:,1:2)))+1))';


root_idx = Z(end,1:2);
link_stat = Z(end,3);
children_all = get_children_node(Z,root_idx); % the two clusters of mvmts

data=[];tmp_data={};num_bins={};
for i=1:length(children_all)
    idx = children_all{i};
    tmp = cell2mat(Data(idx));
    data = cat(2,data,tmp);
    tmp_data{i}=tmp;
    num_bins{i}=size(tmp,2);
end

num_mvmts = length(cell2mat(children_all));
num_bins = cumsum(cell2mat(num_bins));
num_bins = num_bins(end)/num_mvmts
num_bins = 1:735:(num_mvmts*735+1);

% fit the gaussian to the overall data
m = mean(data,2);
X = cov(data');
C12 = chol(X);
link_stat_boot=[];
options = statset('UseParallel',true);
parfor i=1:10
    g = randn(size(data));
    cnew = m + C12'*g;



    % find clusters in the data for simulated movements using k-means
    %idx = kmeans(cnew', num_mvmts,'MaxIter',250,'Options',options);

    % find clusters using agglom clustering
    %Z = linkage(cnew','complete');    
    %idx = cluster(Z,'maxclust',num_mvmts);

    % create the dataset
    %data_tmp={};
    %for j=1:length(unique(idx))
    %    data_tmp{j} = cnew(:,find(idx==j));
    %end


    % random assignment
    data_tmp={};
    idx=randperm(size(data,2));
    cnew = data(:,idx);
    for j=1:length(num_bins)-1
       data_tmp{j}=cnew(:, num_bins(j):num_bins(j+1)-1);
    end

    Dboot=zeros(size(data_tmp));
    for j=1:length(data_tmp)
        A = data_tmp{j};
        for k=j+1:length(data_tmp)
            B = data_tmp{k};
            Dboot(j,k) = mahal2(A',B',2);
            Dboot(k,j)= Dboot(j,k);
        end
    end
    Znull  = linkage(squareform(Dboot),'complete');
    %figure;(dendrogram(Znull))
    link_stat_boot(i) = Znull(end,3);
end
link_stat_boot
link_stat
sum(link_stat_boot>=link_stat)/length(link_stat_boot)

%
% %%% individually feature by feature
% a = Data{1};
% b = Data{3};
% d = mahal2(a',b',2);
% dboot=[];
% s1 = size(a,2);
% parfor i=1:1000
%     a1={};
%     b1={};
%     for j=1:128:size(a,1)
%         atmp = a(j:(j+128)-1,:);
%         btmp = b(j:(j+128)-1,:);
%         c = [atmp btmp];
%
%         % permutation
%         %I = randperm(size(c,2));
%         %atmp = c(:,I(1:s1));
%         %btmp = c(:,I(s1+1:end));
%
%         % simulating gaussian
%         m = mean(c,2);
%         X = cov(c');
%         C12 = chol(X);
%         g = randn(size(c));
%         cnew = m + C12'*g;
%         %atmp = cnew(:,1:s1);
%         %btmp = cnew(:,s1+1:end);
%
%         % now find two clusters from the simulated data
%         idx = kmeans(cnew', 2);
%         atmp = cnew(:,find(idx==1));
%         btmp = cnew(:,find(idx==2));
%
%         % combine
%         %a1 = [a1;atmp];
%         %b1 = [b1;btmp];
%         a1 = cat(2,a1,atmp);
%         b1 = cat(2,b1,btmp);
%     end
%     % stack together
%     s11=[];s22=[];
%     for j=1:length(a1)
%         s11(j) = size(a1{j},2);
%         s22(j) = size(b1{j},2);
%     end
%     s11 = min(s11)
%     s22 = min(s22)
%     a11=[];b11=[];
%     for j=1:length(a1)
%         tmp=a1{j};
%         a11=[a11;tmp(:,1:s11)];
%         tmp=b1{j};
%         b11=[b11;tmp(:,1:s22)];
%     end
%     dboot(i) = mahal2(a11',b11',2);
% end
% figure;hist(dboot)
% vline(d)

% testing with draw from a random distribution using same k-means
pval=[];
parfor iter=1:10
    disp(iter)
    orig=sort(randn(250,1));
    a = orig(1:125);
    b= orig(126:end)+1.75; % adding 1 here makes it a new distribution
    c=[ a; b];
    %[h p ci st]=ttest2(a,b);
    %d = abs(st.tstat);
    % statistic is k means cluster statistic
    ssum = sum((a-mean(a)).^2) + sum((b-mean(b)).^2);
    tss = sum((c-mean(c)).^2);
    ci_stat = ssum/tss;
    d = ci_stat;

    % fit a gaussian using entries of C
    m = mean(c);
    s = std(c);
    dboot=[];
    for i=1:1000
        cnew = s.*randn(size(c)) + m;
        idx = kmeans(cnew, 2);
        a1 = cnew(idx==1);
        b1 = cnew(idx==2);
        %[h p ci st]=ttest2(a1,b1);
        %dboot(i) = abs(st.tstat);
        ssum = sum((a1-mean(a1)).^2) + sum((b1-mean(b1)).^2);
        tss = sum((cnew-mean(cnew)).^2);
        dboot(i) =  ssum/tss;
    end
    %figure;hist(dboot)
    %vline(ci_stat)
    pval(iter) = sum(dboot>d)/length(dboot);
end
figure;hist(pval)

%% MAIN going down the dendrogram one leaf node at a time

%need to send Data, Z and the associated linked functions

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
%cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

Z(:,end+1) = ((size(D,1)+1):(max(max(Z(:,1:2)))+1))'
figure;dendrogram(Z(:,1:3),0)

pval_link=[];
pval_2means=[];
for i=0:length(Z)-1

    % get the movements within the clusters
    root_idx = Z(end-i,1:2);
    link_stat = Z(end-i,3);
    children_all = get_children_node(Z,root_idx); % the two clusters of mvmts

    % get the data together
    data=[];tmp_data={};num_bins={};
    for ii=1:length(children_all)
        idx = children_all{ii};
        tmp = cell2mat(Data(idx));
        data = cat(2,data,tmp);
        tmp_data{ii}=tmp;
        num_bins{ii}=size(tmp,2);
    end

    num_mvmts = length(cell2mat(children_all));    

    % fit the gaussian to the overall data
    m = mean(data,2);
    X = cov(data');
    C12 = chol(X);
    link_stat_boot=[];
    options = statset('UseParallel',true);
    parfor ii=1:12
        disp(ii)
        g = randn(size(data));
        cnew = m + C12'*g;



        % find clusters in the data for simulated movements using k-means
        idx = kmeans(cnew', num_mvmts,'MaxIter',250,'Options',options);

        % find clusters using agglom clustering
        %ZZ = linkage(cnew','complete');
        %idx = cluster(ZZ,'maxclust',num_mvmts);
        data_tmp={};
        for j=1:length(unique(idx))
            data_tmp{j} = cnew(:,find(idx==j));
        end


        % random assignment
        %     data_tmp={};
        %     idx=randperm(size(data,2));
        %     cnew = data(:,idx);
        %     for j=1:length(num_bins)-1
        %         data_tmp{j}=cnew(:, num_bins(j):num_bins(j+1)-1);
        %     end

        Dboot=zeros(size(data_tmp,2));
        for j=1:length(data_tmp)
            A = data_tmp{j};
            for k=j+1:length(data_tmp)
                B = data_tmp{k};
                Dboot(j,k) = mahal2(A',B',2);
                Dboot(k,j)= Dboot(j,k);
            end
        end
        Znull  = linkage(squareform(Dboot),'complete');
        %figure;(dendrogram(Znull,0))
        link_stat_boot(ii) = Znull(end,3);
    end
    pval_link(i+1) = sum(link_stat_boot>=link_stat)/length(link_stat_boot);
end

Z(:,end+1) = flipud(pval_link)
