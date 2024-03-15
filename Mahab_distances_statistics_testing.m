
% COME HERE FROM THE REPRESENTATIONAL STRUCURE ANALYSES CODE
% testong between mvmts 2 and 7

%%
%%%% null hypothesis testing -> permutation of each freq feature mvmt 1 and
% mvmt2. Create a cmmon distribution, sample from the distribution and use
% k-means to get two clusters. then find the mahab distance between the two
% clusters
%%% using all features
a = Data{1};
b = Data{13};
d = mahal2(a',b',2);
dboot=[];
s1 = size(a,2);
c = [a b];
m = mean(c,2);
X = cov(c');
C12 = chol(X);
parfor i=1:50
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

%% Going down the dendrogram , smulation
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


root_idx = Z(end-16,1:2);
link_stat = Z(end-16,3);
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
    idx = kmeans(cnew', num_mvmts,'MaxIter',250,'Options',options);    

    % find clusters using agglom clustering
    %Z = linkage(cnew','complete');
    %idx = cluster(Z,'maxclust',num_mvmts);
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
    figure;(dendrogram(Znull))
    link_stat_boot(i) = Znull(end,3);
end
link_stat_boot
link_stat

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