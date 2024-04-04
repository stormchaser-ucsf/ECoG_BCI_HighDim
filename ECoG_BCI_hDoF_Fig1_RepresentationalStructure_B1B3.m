% Figure 1 Represenational structure

%% for B1
clc;clear

addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')

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
    'Right Leg','Left Leg',...
    'Lips','Tongue'};

bins_size(1,:) = [3 16];
bins_size(2,:) = [2 16];
bins_size(3,:) = [3 17];
bins_size(4,:) = [2 17];


% look at the approprate bin size
for n=1:size(bins_size,1)%10:18
    bins_size=3:17; %3:17 is GOLD , with only hG it is even more gold
    %bins_size1 = [bins_size(n,1):bins_size(n,2)];
    [Data,bins_per_mvmt,TrialData] = ...
        load_B1Data_RepresenatationalStruct_Fig1(bins_size);
    chMap = TrialData.Params.ChMap;
    Data_bkup=Data;

    % artfiact correction
    for i=1:length(Data)
        tmp=Data{i};
        for j=1:size(tmp,1)
            t = zscore(tmp(j,:));
            idx=(abs(t)>6);
            tmp(j,idx) = median(tmp(j,:));
        end
        Data{i}=tmp;
    end


    % plot average ERP for a single movement at a specific channel
    %     tmp = Data{1}; % rt thumb
    %     bins = bins_per_mvmt{1};
    %     bins = [0 cumsum(bins)];
    %     erp=[];
    %     for i=1:length(bins)-1
    %         tmp_data = tmp(259,bins(i)+1:bins(i+1));
    %         erp(i,:) = tmp_data;
    %     end
    %     figure;plot(erp')
    %     figure;plot(mean(erp,1))



    % get the number of trials per movement and bins per movement
    %     trials_num=[];bins=[];
    %     for i=1:length(bins_per_mvmt)
    %         tmp=bins_per_mvmt{i};
    %         trials_num(i) = length(tmp);
    %         bins=[bins tmp];
    %     end
    %     unique(bins)



    % plotting for Figure

    % get the mahalanobis distance between the imagined actions
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

    figure;
    subplot(1,2,1)
    imagesc(D)
    xticks(1:size(D,1))
    yticks(1:size(D,1))
    xticklabels(ImaginedMvmt)
    yticklabels(ImaginedMvmt)
    set(gcf,'Color','w')
    colormap turbo
    caxis([0 140])
    %cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
    % D_B1=D;
    %save representational_similarity_matrix_B1 D_B1 ImaginedMvmt

    subplot(1,2,2)
    Z = linkage(squareform(D),'complete');
    [H,T,outperm] = dendrogram(Z,0);
    x = string(get(gca,'xticklabels'));
    x1=[];
    for i=1:length(x)
        tmp = str2num(x{i});
        x1 = [x1 ImaginedMvmt(tmp)];
    end
    xticklabels(x1)
    set(gcf,'Color','w')
    sgtitle(['Bins till ' num2str(bins_size(end))])
    %sgtitle(['Bins till ' [num2str(bins_size1(1)) ' ' num2str(bins_size1(end))]])

    % MD scaling
    figure;
    [Y,~,disparities] = mdscale(D,2,'Start','random');
    Ymds=Y;
    subplot(2,1,1);hold on
    cmap = turbo(length(ImaginedMvmt));
    idx=[1,10,19,20,28,29,30];
    for i=1:size(Y,1)
        plot(Y(i,1),Y(i,2),'.','Color',cmap(i,:),'MarkerSize',20)
        if sum(i==idx)==1
            text(Y(i,1),Y(i,2),ImaginedMvmt{i},'FontWeight','bold','FontSize',10);
        else
            text(Y(i,1),Y(i,2),ImaginedMvmt{i},'FontSize',10);
        end
    end



    % t-sne on average
    subplot(2,1,2);hold on
    A=[];idx=[];samples=20;
    for i=1:length(ImaginedMvmt)
        tmp = Data{i}';
        bins = bins_per_mvmt{i};
        bins = [0 cumsum(bins)];
        %bins = [0 (bins)];
        A1=[];
        for j=1:length(bins)-1
            trial_bins = tmp(bins(j)+1:bins(j+1),1:end);
            A1=  [A1; median(trial_bins,1)];
            idx=[idx;i];
        end
        A1 = bootstrp(samples,@mean,A1);
        A = [A;A1];
    end
    Y = tsne(A,'Perplexity',40);


    % plotting with ellipses
    cmap = parula(length(ImaginedMvmt));
    idx=[1,10,19,20,28,29,30];
    len = 1:samples:size(Y,1)+1;
    for i=1:length(len)-1
        tmp = Y(len(i):len(i+1)-1,:);
        C=  cov(tmp);
        plot_gaussian_ellipsoid(mean(tmp),C);
        %plot(tmp(:,1),tmp(:,2),'.','Color',cmap(i,:),'MarkerSize',20)
        x=mean(tmp,1);
        if sum(i==idx)==1
            text(x(1),x(2),ImaginedMvmt{i},'FontWeight','bold','FontSize',8);
        else
            text(x(1),x(2),ImaginedMvmt{i},'FontSize',8);
        end
    end
    sgtitle(['Bins till ' num2str(bins_size(end))])

    % color the mvmts in MDS space by mvmt type
    % rt hand, lt hand, both hands, rt proximal, lt proximal, rt leg, left
    % leg, head, lips/tong
    %cmap = turbo(9);
    cmap = brewermap(9,'dark2');
    %cmap(6,:) = [.3 1 .3];
    figure;hold on
    for ii=1:9 % rt hand
        plot(Ymds(ii,1),Ymds(ii,2),'.','Color',cmap(1,:),'MarkerSize',20);
        text(Ymds(ii,1),Ymds(ii,2),ImaginedMvmt{ii},'FontSize',10,'Color',cmap(1,:));
    end
    for ii=10:18 % rt hand
        plot(Ymds(ii,1),Ymds(ii,2),'.','Color',cmap(2,:),'MarkerSize',20);
        text(Ymds(ii,1),Ymds(ii,2),ImaginedMvmt{ii},'FontSize',10,'Color',cmap(2,:));
    end
    for ii=19 % bmf
        plot(Ymds(ii,1),Ymds(ii,2),'.','Color',cmap(3,:),'MarkerSize',20);
        text(Ymds(ii,1),Ymds(ii,2),ImaginedMvmt{ii},'FontSize',10,'Color',cmap(3,:));
    end
    for ii=20 % head
        plot(Ymds(ii,1),Ymds(ii,2),'.','Color',cmap(4,:),'MarkerSize',20);
        text(Ymds(ii,1),Ymds(ii,2),ImaginedMvmt{ii},'FontSize',10,'Color',cmap(4,:));
    end
    idx=[21 23 25];
    for ii=1:length(idx) %rt proximal
        plot(Ymds(idx(ii),1),Ymds(idx(ii),2),'.','Color',cmap(5,:),'MarkerSize',20);
        text(Ymds(idx(ii),1),Ymds(idx(ii),2),ImaginedMvmt{idx(ii)},...
            'FontSize',10,'Color',cmap(5,:));
    end
    idx=[21 23 25]+1;
    for ii=1:length(idx) %lt proximal
        plot(Ymds(idx(ii),1),Ymds(idx(ii),2),'.','Color',cmap(6,:),'MarkerSize',20);
        text(Ymds(idx(ii),1),Ymds(idx(ii),2),ImaginedMvmt{idx(ii)},...
            'FontSize',10,'Color',cmap(6,:));
    end
    for ii=27 %rt leg
        plot(Ymds((ii),1),Ymds((ii),2),'.','Color',cmap(7,:),'MarkerSize',20);
        text(Ymds((ii),1),Ymds((ii),2),ImaginedMvmt{(ii)},...
            'FontSize',10,'Color',cmap(7,:));
    end
    for ii=28 %lt leg
        plot(Ymds((ii),1),Ymds((ii),2),'.','Color',cmap(8,:),'MarkerSize',20);
        text(Ymds((ii),1),Ymds((ii),2),ImaginedMvmt{(ii)},...
            'FontSize',10,'Color',cmap(8,:));
    end
    for ii=29:30 %lips tong
        plot(Ymds((ii),1),Ymds((ii),2),'.','Color',cmap(9,:),'MarkerSize',20);
        text(Ymds((ii),1),Ymds((ii),2),ImaginedMvmt{(ii)},...
            'FontSize',10,'Color',cmap(9,:));
    end
end

dim = size(Data{1},1);
%D_all = squareform(D)./sqrt(dim);
%D_3feat = squareform(D)./sqrt(dim);
%D_hg = squareform(D)./sqrt(dim);
%D_delta = squareform(D)./sqrt(dim);
D_beta = squareform(D)./sqrt(dim);


D_feat_compare = [D_3feat' D_hg' D_delta' D_beta' ];
figure;boxplot(log10(D_feat_compare))


% for B1 -> looking at within and between cluster mahab distance ratios
idx1 = [1:9 19] ;% all rt hands and both hands
idx2 = [10:18 23 24 25]; % all lt hand mvmts, both tricep and rt bicep
idx3 = [20 21 22 26:30]; % head, face and other proximal and distal mvmts
clear idx
idx{1}=idx1;
idx{2}=idx2;
idx{3}=idx3;

%%% ratio with all features
% within
Dratio=[];len=1:length(idx);
for i=1:length(idx)
    tmp = idx{i};
    %for each mvmt, get ratio of average within cluster distanceto average
    %b/w cluster distance

    % within cluster distances for each movement
    Dtmp = zeros(length(tmp));
    for j=1:length(tmp)
        a = Data{tmp(j)};
        a=a(1:end,:);
        for k=j+1:length(tmp)
            b = Data{tmp(k)};
            b=b(1:end,:);
            %c=[a b];s=size(a,2);
            %c = c(:,randperm(size(c,2)));
            %a = c(:,1:s);
            %b = c(:,s+1:end);
            Dtmp(j,k) = mahal2(a',b',2);
            Dtmp(k,j) = Dtmp(j,k);
        end
    end
    % remove the zero
    Dwithin = zeros(length(tmp)-1);
    all_idx = ones(length(tmp),1);
    for j=1:size(Dtmp,1)
        a = Dtmp(j,:);
        tmp_idx = all_idx;
        tmp_idx(j)=0;
        tmp_idx=find(tmp_idx==1);
        Dwithin(j,:) = a(tmp_idx);
    end

    % between cluster distances for each movement
    % out-group mvmts index
    out_idx = find( (1-(i == len)) == 1);
    out_idx = cell2mat(idx(out_idx));
    Dtmp = zeros(length(tmp),length(out_idx));
    for j=1:length(tmp)
        a = Data{tmp(j)};
        a=a(1:end,:);
        for k=1:length(out_idx)
            b = Data{out_idx(k)};
            b=b(1:end,:);
            %c=[a b];s=size(a,2);
            %c = c(:,randperm(size(c,2)));
            %a = c(:,1:s);
            %b = c(:,s+1:end);
            Dtmp(j,k) = mahal2(a',b',2);
        end
    end
    Dbetween=Dtmp;
    Dratio = [Dratio mean(Dwithin')./mean(Dbetween')];
end



figure;boxplot(Dratio)

% now after randomly simulating

Dratio_all = Dratio;
Dratio_hg = Dratio;
Dratio_delta = Dratio;
Dratio_beta = Dratio;
Dnull=Dratio;

% ratio with delta

% ratio with beta

% ration with hG

% stats stuff on the mahab distances
% determining clusters as less than 50% of max between cluster distance
Z = linkage(D,'ward');
%Z(:,3) = Z(:,3)./max(Z(:,3));
figure;
dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')

% distance b/w lips and tongue is 53.1762. Sig different from chance?
data1 = Data(13:14);
s1 = size(data1{1},2);
s2 = size(data1{2},2);
data1 = cell2mat(data1);
Znull=[];
parfor i=1:1000
    idx = randperm(size(data1,2));
    null_data = data1(:,idx);
    a = null_data(:,1:s1);
    b = null_data(:,s1+1:end);
    d = mahal2(a',b',2);
    Znull(i) = d;
end
figure;hist(Znull);vline(32.8867)

% whether three movements that join as a cluster are actually distinct from
% each other based on ratio of split at the different levels
idx  = [2,7,8]; % index, pinch, tripod
stat = 47.6/60.61; % (index and pinch joining with tripod)

data1 = Data(idx);
s=[];
for i=1:length(data1)
    s(i) = size(data1{i},2);
end
data1 = cell2mat(data1);
null_ratio=[];
s=[0 cumsum(s)];
for i=1:1000
    disp(i)
    tmp = data1(:,randperm(size(data1,2)));
    for j=1:length(s)-1
        idx = (s(j)+1):(s(j+1));
        tmp_data{j} = tmp(:,idx);
    end
    Dnull=zeros(length(tmp_data));
    for j=1:length(tmp_data)
        a = tmp_data{j};
        for k=j+1:length(tmp_data)
            b=tmp_data{k};
            Dnull(j,k) = mahal2(a',b',2);
            Dnull(k,j) = Dnull(j,k) ;
        end
    end
    Znull = linkage(Dnull,'ward');
    null_ratio(i) = Znull(1,3)/Znull(2,3);
end
figure;hist(null_ratio)
vline(stat)
sum(null_ratio<=stat)/length(null_ratio)

% complete random split how does dengrogram look
idx  = [1:30];
data1 = Data(idx);
s=[];
for i=1:length(data1)
    s(i) = size(data1{i},2);
end
data1 = cell2mat(data1);
null_ratio=[];
s=[0 cumsum(s)];
for i=1:10
    disp(i)
    tmp = data1(:,randperm(size(data1,2)));
    for j=1:length(s)-1
        idx = (s(j)+1):(s(j+1));
        tmp_data{j} = tmp(:,idx);
    end
    Dnull=zeros(length(tmp_data));
    for j=1:length(tmp_data)
        a = tmp_data{j};
        for k=j+1:length(tmp_data)
            b=tmp_data{k};
            Dnull(j,k) = mahal2(a',b',2);
            Dnull(k,j) = Dnull(j,k) ;
        end
    end
    Znull = linkage(Dnull,'ward');
    null_ratio(i) = Znull(end-1,3)/Znull(end,3);
end

% for random
Data_rnd={};
for i=1:length(Data)
    tmp = Data{i};
    tmp = randn(128,size(tmp,2));
    Data_rnd{i} = tmp;
end
Data=Data_rnd;

D_all=squareform(D);
D_hg = squareform(D);
D_delta =  squareform(D);
D_beta =  squareform(D);
D_rnd = squareform(D);
D_rnd_128 = squareform(D);

tmp = log10([D_all' D_hg' D_delta' D_beta' D_rnd' D_rnd_128']);
figure;boxplot(tmp)

%% AMBITIOUS -> B1 MAHAB DISTANCES ON AVERAGE ERP FROM RAW DATA

clear
clc
load high_res_erp_hgLFO_imagined_data
% build the mahab distnaces on just hG ERP
Data={};
for i=1:length(ERP_Data)
    tmp=ERP_Data{i};
    tmp= squeeze(mean(tmp,3));
    tmp=tmp(3500:6500,:);
    Data{i}=tmp';
end

D=zeros(length(ImaginedMvmt));
for i=1:length(Data)
    disp(['Processing Mvmt ' num2str(i) ])
    A = Data{i}';
    for j=i+1:length(Data)
        B = Data{j}';
        d = mahal2(A,B,2);
        D(i,j)=d;
        D(j,i)=d;
    end
end

figure;
subplot(1,2,1)
imagesc(D)
xticks(1:size(D,1))
yticks(1:size(D,1))
xticklabels(ImaginedMvmt)
yticklabels(ImaginedMvmt)
set(gcf,'Color','w')
%colormap bone
%caxis([0 200])

subplot(1,2,2)

Z = linkage(D,'complete');
dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')


%%  B1 but Mahab on average ERP -> have to do it on the raw data
% ALSO VERY PROMISING
clc;clear

addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))

ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Rotate Left Wrist','Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};

% look at the approprate bin size
for n=10:18
    bins_size=2:15;
    [Data,bins_per_mvmt,TrialData] = ...
        load_B1Data_RepresenatationalStruct_Fig1(bins_size);
    chMap = TrialData.Params.ChMap;
    Data_bkup=Data;


    % plot average ERP for a single movement at a specific channel
    tmp = Data{20}; % rt thumb
    bins = bins_per_mvmt{20};
    bins = [0 cumsum(bins)];
    erp=[];
    for i=1:length(bins)-1
        tmp_data = tmp(303,bins(i)+1:bins(i+1));
        erp(i,:) = tmp_data;
    end
    figure;plot(erp')
    figure;plot(mean(erp,1))

    % get the average neural response as the Data
    Data_new={};
    for ii=1:length(Data)
        tmp = Data{ii}; % rt thumb
        bins = bins_per_mvmt{ii};
        bins = [0 cumsum(bins)];
        erp=[];
        for i=1:length(bins)-1
            tmp_data = tmp(:,bins(i)+1:bins(i+1));
            for j=1:size(tmp_data,1)
                tmp_data(j,:) = smooth(tmp_data(j,:),5);
            end
            erp(i,:,:) = tmp_data;
        end
        %erp=squeeze(mean(erp,1));
        erp=permute(erp,[2 3 1]);
        erp = erp(:,:);
        Data_new{ii} = erp;
    end
    Data = Data_new;

    % get the number of trials per movement and bins per movement
    %     trials_num=[];bins=[];
    %     for i=1:length(bins_per_mvmt)
    %         tmp=bins_per_mvmt{i};
    %         trials_num(i) = length(tmp);
    %         bins=[bins tmp];
    %     end
    %     unique(bins)



    % plotting for Figure

    % get the mahalanobis distance between the imagined actions
    D=zeros(length(ImaginedMvmt));
    for i=1:length(Data)
        disp(['Processing Mvmt ' num2str(i) ...
            ' with bins till '  num2str(bins_size(end))])
        A = Data{i}';
        for j=i+1:length(Data)
            B = Data{j}';
            d = mahal2(A,B,2);
            D(i,j)=d;
            D(j,i)=d;
        end
    end

    figure;
    subplot(1,2,1)
    imagesc(D)
    xticks(1:size(D,1))
    yticks(1:size(D,1))
    xticklabels(ImaginedMvmt)
    yticklabels(ImaginedMvmt)
    set(gcf,'Color','w')
    colormap turbo
    %caxis([0 200])

    subplot(1,2,2)

    Z = linkage(squareform(D),'ward');
    dendrogram(Z,0)
    x = string(get(gca,'xticklabels'));
    x1=[];
    for i=1:length(x)
        tmp = str2num(x{i});
        x1 = [x1 ImaginedMvmt(tmp)];
    end
    xticklabels(x1)
    set(gcf,'Color','w')
    sgtitle(['Bins till ' num2str(bins_size(end))])

    % MD scaling
    figure;
    [Y,~,disparities] = mdscale(D,2,'Start','random');
    subplot(2,1,1);hold on
    cmap = turbo(length(ImaginedMvmt));
    idx=[1,10,19,20,28,29,30];
    for i=1:size(Y,1)
        plot(Y(i,1),Y(i,2),'.','Color',cmap(i,:),'MarkerSize',20)
        if sum(i==idx)==1
            text(Y(i,1),Y(i,2),ImaginedMvmt{i},'FontWeight','bold','FontSize',10);
        else
            text(Y(i,1),Y(i,2),ImaginedMvmt{i},'FontSize',10);
        end
    end



    % t-sne on average
    subplot(2,1,2);hold on
    A=[];idx=[];samples=20;
    for i=1:length(ImaginedMvmt)
        tmp = Data{i}';
        bins = bins_per_mvmt{i};
        bins = [0 cumsum(bins)];
        %bins = [0 (bins)];
        A1=[];
        for j=1:length(bins)-1
            trial_bins = tmp(bins(j)+1:bins(j+1),1:end);
            A1=  [A1; median(trial_bins,1)];
            idx=[idx;i];
        end
        A1 = bootstrp(samples,@mean,A1);
        A = [A;A1];
    end
    Y = tsne(A,'Perplexity',40);


    % plotting with ellipses
    cmap = parula(length(ImaginedMvmt));
    idx=[1,10,19,20,28,29,30];
    len = 1:samples:size(Y,1)+1;
    for i=1:length(len)-1
        tmp = Y(len(i):len(i+1)-1,:);
        C=  cov(tmp);
        plot_gaussian_ellipsoid(mean(tmp),C);
        %plot(tmp(:,1),tmp(:,2),'.','Color',cmap(i,:),'MarkerSize',20)
        x=mean(tmp,1);
        if sum(i==idx)==1
            text(x(1),x(2),ImaginedMvmt{i},'FontWeight','bold','FontSize',8);
        else
            text(x(1),x(2),ImaginedMvmt{i},'FontSize',8);
        end
    end
    sgtitle(['Bins till ' num2str(bins_size(end))])



end

% null stats using gaussian simulation
a= Data{22};
b= Data{27};
d = mahal2(a',b',2);



%% for B3

clc;clear

addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))

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

% ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
%     'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
%     'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
%     'Rotate Left Wrist','Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
%     'Squeeze Both Hands',...
%     'Head Movement',...
%     'Right Shoulder Shrug',...
%     'Left Shoulder Shrug',...
%     'Right Tricep','Left Tricep',...
%     'Right Bicep','Left Bicep',...
%     'Right Knee','Left Knee',...
%     'Lips','Tongue'};


cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')

load('ECOG_Grid_8596_000067_B3.mat')
chMap=ecog_grid;
grid_layout = chMap;

% rename all the channels after accounting for 108,113,118
for i=109:112
    [x, y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-1;
end

for i=114:117
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-2;
end

for i=119:256
    [x ,y]=find(grid_layout ==i);
    grid_layout(x,y)=grid_layout(x,y)-3;
end

neighb=zeros(size(grid_layout));
for i=1:numel(grid_layout)
    [x y]=find(grid_layout == i);
    Lx=[x-1 x+1];
    Ly=[y-1 y+1];
    Lx=Lx(logical((Lx<=size(grid_layout,1)) .* (Lx>0)));
    Ly=Ly(logical((Ly<=size(grid_layout,2)) .* (Ly>0)));
    temp=grid_layout(Lx,Ly);
    ch1=[grid_layout(x,Ly)';grid_layout(Lx,y);];
    neighb(i,ch1)=1;
end
figure;
imagesc(neighb)

% look at the approprate bin size
bins_size(1,:) = [1 14];
bins_size(2,:) = [2 14];
bins_size(3,:) = [3 14];
bins_size(4,:) = [1 15];
bins_size(5,:) = [2 15];
bins_size(6,:) = [3 15];
bins_size(7,:) = [1 16];
bins_size(8,:) = [2 16];
bins_size(9,:) = [3 16];
bins_size(10,:) = [1 17];
bins_size(11,:) = [2 17];
bins_size(12,:) = [3 17];
bins_size = bins_size(7:end,:); % this may be optimal? 

D_overall=[]; % this is the best approach
for n=1:size(bins_size,1)%n=10:18
    bins_size1=3:16; % 3 to 16 is good
    %bins_size1 = [bins_size(n,1):bins_size(n,2)];

    [Data,bins_per_mvmt,TrialData] = ...
        load_B3Data_RepresenatationalStruct_Fig1(bins_size1);
    Data_bkup=Data;

    % exlcuding ankle
    %Data = Data([1:28 31 32]);
    %ImaginedMvmt = ImaginedMvmt([1:28 31 32]);

    % artfiact correction
    for i=1:length(Data)
        tmp=Data{i};
        for j=1:size(tmp,1)
            t = zscore(tmp(j,:));
            idx=(abs(t)>6);
            tmp(j,idx) = median(tmp(j,:));
        end
        Data{i}=tmp;
    end


    % plot average ERP for a single movement at a specific channel
    tmp = Data{1}; % rt thumb
    bins = bins_per_mvmt{1};
    bins = [0 cumsum(bins)];
    erp=[];
    for i=1:length(bins)-1
        tmp_data = tmp(137,bins(i)+1:bins(i+1));
        erp(i,:) = tmp_data;
    end
    figure;plot(erp');close
    figure;plot(mean(erp,1));close



    % get the number of trials per movement and bins per movement
    %     trials_num=[];bins=[];
    %     for i=1:length(bins_per_mvmt)
    %         tmp=bins_per_mvmt{i};
    %         trials_num(i) = length(tmp);
    %         bins=[bins tmp];
    %     end
    %     unique(bins)





    % plotting for Figure

    % get the mahalanobis distance between the imagined actions
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
    D_overall(n,:,:)=D;
    %cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
    %D_B3_3to16=D;
    %save D_B3_3to16 D_B3_3to16


    Z = linkage(squareform(D),'complete');
    figure
    dendrogram(Z,0)
    x = string(get(gca,'xticklabels'));
    x1=[];
    for i=1:length(x)
        tmp = str2num(x{i});
        x1 = [x1 ImaginedMvmt(tmp)];
    end
    xticklabels(x1)
    set(gcf,'Color','w')
    sgtitle(['Bins till ' [num2str(bins_size1(1)) ' ' num2str(bins_size1(end))]])
    close
end


close all
D = squeeze(mean(D_overall,1));
figure;imagesc(D)
colormap turbo
xticks(1:length(ImaginedMvmt))
yticks(1:length(ImaginedMvmt))
xticklabels(ImaginedMvmt)
yticklabels(ImaginedMvmt)
Z = linkage(squareform(D),'complete');
figure
dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
%D_B3_Avg = D;
%save representational_similarity_matrix_B3 D_B3_Avg ImaginedMvmt

%%%% correlation between B1 and B3
% load D_B3_3to16; D = D_B3_3to16;
% load representational_similarity_matrix_B3
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\representational_similarity_matrix_B1.mat','D_B1');
DB1=D_B1;
% extract just the movements that are common to both B1 and B3
bad_idx=[29:30];
I = ones(size(D,1),1);
I(bad_idx)=0;I=logical(I);
DB3=D;
DB3 = DB3(:,I);
DB3 = DB3(I,:);
[stat,pval] = corr(squareform(DB3)',squareform(DB1)')
a=squareform(DB3)';b=squareform(DB1)';
boot_corr=[];
for iter=1:1000
    a1 = a(randperm(length(a)));
    b1 = b(randperm(length(b)));
    boot_corr(iter) = corr(a1,b1);
end
figure;hist(boot_corr)
vline(stat)

m = mean(boot_corr);
s = std(boot_corr);
r = m + s*randn(1000,1);
hold on
ksdensity(r)


% 
% figure;
% subplot(1,2,1)
% imagesc(D)
% xticks(1:size(D,1))
% yticks(1:size(D,1))
% xticklabels(ImaginedMvmt)
% yticklabels(ImaginedMvmt)
% set(gcf,'Color','w')
% colormap turbo
% %caxis([0 200])
% 
% subplot(1,2,2)
% 
% Z = linkage(squareform(D),'complete');
% dendrogram(Z,0)
% x = string(get(gca,'xticklabels'));
% x1=[];
% for i=1:length(x)
%     tmp = str2num(x{i});
%     x1 = [x1 ImaginedMvmt(tmp)];
% end
% xticklabels(x1)
% set(gcf,'Color','w')
% sgtitle(['Bins till ' num2str(bins_size(end))])
% 
% % MD scaling
% figure;
% [Y,~,disparities] = mdscale(D,2,'Start','random');
% subplot(2,1,1);hold on
% cmap = turbo(length(ImaginedMvmt));
% idx=[1,10,19,20,28,29,30];
% for i=1:size(Y,1)
%     plot(Y(i,1),Y(i,2),'.','Color',cmap(i,:),'MarkerSize',20)
%     if sum(i==idx)==1
%         text(Y(i,1),Y(i,2),ImaginedMvmt{i},'FontWeight','bold','FontSize',10);
%     else
%         text(Y(i,1),Y(i,2),ImaginedMvmt{i},'FontSize',10);
%     end
% end
% 
% 
% 
% % t-sne on average
% subplot(2,1,2);hold on
% A=[];idx=[];samples=20;
% for i=1:length(ImaginedMvmt)
%     tmp = Data{i}';
%     bins = bins_per_mvmt{i};
%     bins = [0 cumsum(bins)];
%     %bins = [0 (bins)];
%     A1=[];
%     for j=1:length(bins)-1
%         trial_bins = tmp(bins(j)+1:bins(j+1),1:end);
%         A1=  [A1; median(trial_bins,1)];
%         idx=[idx;i];
%     end
%     A1 = bootstrp(samples,@mean,A1);
%     A = [A;A1];
% end
% Y = tsne(A,'Perplexity',40);
% 
% 
% % plotting with ellipses
% cmap = parula(length(ImaginedMvmt));
% idx=[1,10,19,20,28,29,30];
% len = 1:samples:size(Y,1)+1;
% for i=1:length(len)-1
%     tmp = Y(len(i):len(i+1)-1,:);
%     C=  cov(tmp);
%     plot_gaussian_ellipsoid(mean(tmp),C);
%     %plot(tmp(:,1),tmp(:,2),'.','Color',cmap(i,:),'MarkerSize',20)
%     x=mean(tmp,1);
%     if sum(i==idx)==1
%         text(x(1),x(2),ImaginedMvmt{i},'FontWeight','bold','FontSize',8);
%     else
%         text(x(1),x(2),ImaginedMvmt{i},'FontSize',8);
%     end
% end
% sgtitle(['Bins till ' num2str(bins_size(end))])
% 
% 
% 
% end




