

% plan here is to look at changes in neural feature discriminability from
% early in training to online control, tracked across days

% hypothesis is that B1 gets better at generating those 'spatial pops' that
% discriminate betwween the various actions
clc;clear
close all

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

% for only 6 DoF original:
%foldernames = {'20210526','20210528','20210602','20210609_pm','20210611'};

foldernames = {'20210615','20210616','20210623','20210625','20210630','20210702',...
    '20210707','20210716','20210728','20210804','20210806','20210813','20210818',...
    '20210825','20210827','20210901','20210903','20210910','20210917','20210924','20210929',...
    '20211001''20211006','20211008','20211013','20211015','20211022','20211027','20211029','20211103',...
    '20211105','20211117','20211119','20220126','20220128','20220202','20220204','20220209','20220211',...
    '20220218','20220223','20220225','20220302'};
cd(root_path)


% 20210423 -> 111360, CenterOut -> right hand focus - rt -> rt thumb, top -> rt index finger, bottom -> rt middle finger, left -> left thumb
% do regression to show that there is not much information towards
% regression as there is towards classification ?



%% looking at changes in real time neural features in response to errors


%% looking at changes in cosine distance between decoders across learning from first init.

%20210615 is the first day, look at decoder relationships betwen imagined
%and online neural network weights for that day


% 114537, 20210615 -> seeded from imagined movement data with pooling
% 115420, 20210615 -> batch update to the decoder
% 135913 -> seeded part 2, same day
% 140642 -> batch update 2
clc;clear

addpath('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\clicker\7DoF_Classifiers')

% get the seeding decoders
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\114537\BCI_Fixed\Data0006.mat')
net{1} = TrialData.Params.NeuralNetFunction;
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\135913\BCI_Fixed\Data0006.mat')
net{3} = TrialData.Params.NeuralNetFunction;
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210630\Robot3DArrow\103756\BCI_Fixed\Data0009.mat')
net{5} = TrialData.Params.NeuralNetFunction;


% get the batch update decoder
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\115420\BCI_Fixed\Data0004.mat')
net{2} = TrialData.Params.NeuralNetFunction;
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\140642\BCI_Fixed\Data0004.mat')
net{4} = TrialData.Params.NeuralNetFunction;
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210630\Robot3DArrow\110415\BCI_Fixed\Data0013.mat')
net{6} = TrialData.Params.NeuralNetFunction;
net{7} = 'MLP_PreTrained_7DoF_Days1to11';

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

% get similarity over a range of random entries
D=zeros(length(net));
for  j=1:length(net)
    decA = net{j};
    for k=j+1:length(net)
        decB = net{k};
        d=[];
        for i=1:1000
            X = randn(96,1);
            [Y,Xf,Af,a1,a2,a3,a4] = feval(decA,X);
            x1=[a1;a2;a3;Y];
            [Y,Xf,Af,a1,a2,a3,a4] = feval(decB,X);
            x2=[a1;a2;a3;Y];
            tmp =pdist([x1 x2]','cosine');
            %d(i)=sqrt(sum((x1-x2).^2))';
            d(i) = tmp;
        end
        D(j,k) = median(d);
        D(k,j) = median(d);
    end
end

D
Z=linkage(D,'complete')
figure;dendrogram(Z)
figure;plot(D(7,:))



% in addition, look at the distance between the distribution of neural
% data, per condition over recoring blocks and days to get a sense if there
% is stability in the neural data


%% looking at interclass differences at the decoder last layer for imagined vs. online batch

% have to do it day by day to see how decoder discriminability changes from
% an autoencoder perspective, do it for two to three actions : is it % the
% variance of activity that is getting tigher? or that the neural activity
% forms a new space or goes into a different manifold?

% IMPORTANT POINT: WHEN TRAINING THE DECODER DONT 2-NORM THE DATA, BUT THEN
% 2-NORM THE DATA WHEN FEEDING IT THRU THE DECODER THAT SEEMS TO HELP IN
% GETTING THOSE GAUSSIAN SHAPES WHEN DOING THE PCA ON THE SOFTMAX LAYER

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\clicker\7DoF_Classifiers')

% get the seeding decoders
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\114537\BCI_Fixed\Data0006.mat')
net{1} = TrialData.Params.NeuralNetFunction;

% get the batch update decoder
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\115420\BCI_Fixed\Data0004.mat')
net{2} = TrialData.Params.NeuralNetFunction;

%%%%% analysis on the training data
% get the files that were used in the seeding decoder
folders={'110604','111123','111649'};
day_date = '20210615';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'Imagined');
    files = [files;findfiles('',folderpath)'];
end

% have to hold on train and testing on held out trials
len=length(files);
idx=randperm(len,round(0.8*len));
idx1=ones(len,1);
idx1(idx)=0;
train_trials = files(idx);
test_trials = files(logical(idx1));


% load the training data
condn_data = load_data_for_MLP(train_trials);
condn_data = load_data_for_MLP(files);
% get decoder from training data
decoder = get_decoder_for_MLP(condn_data);
cd('C:\Users\Nikhlesh\Documents\GitHub\ECoG_BCI_HighDim')
genFunction(decoder,'MLP_7DoF_Plasticity_20210615_Imagined')

% load the testing data
condn_data = load_data_for_MLP(test_trials);
% get the softmax values at the last layer for the testing trials
decoder_name='MLP_7DoF_Plasticity_20210615_Imagined';
%decoder_name=net{1};
[Y,labels] = get_softmax(condn_data,decoder_name);

% look at the PC space of the softmax layer, color it by label
[c,s,l]=pca(Y');
cmap = parula(7);
figure;hold on
for i=1:length(labels)
    plot3(s(i,1),s(i,2),s(i,3),'.','MarkerSize',10,'Color',cmap(labels(i),:));
end

% get the files that were used to batch update the decoder i.e., the online
% trials
folders = {'113909','114318','114537'};
%folders = {'115420'};
day_date = '20210615';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end
condn_data = load_data_for_MLP(files);
decoder_name=net{1};
[Y,labels] = get_softmax(condn_data,decoder_name);

% look at the PC space of the softmax layer, color it by label
[c,s,l]=pca(Y');
cmap = parula(7);
figure;hold on
for i=1:length(labels)
    plot3(s(i,1),s(i,2),s(i,3),'.','MarkerSize',10,'Color',cmap(labels(i),:));
end





%% keeping track of all the imagined plus online sessions for arrow experiment

clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\clicker\7DoF_Classifiers')
cd(root_path)

%day1
session_data(1).Day = '20210615';
session_data(1).folders = {'110604','111123','111649','113524','113909','114318',...
    '114537','115420','115703','132843','133545','134131','134735','135913','140642',...
    '140904','144621','144814','145829','150031','150224','150839'};
session_data(1).folder_type={'I','I','I','O','O','O','O','B','B','I','I','I','I','O','B',...
    'B','B','B','B','B','B','B'};
session_data(1).AM_PM = {'am','am','am','am','am','am','am','am','am',...
    'pm','pm','pm','pm','pm','pm','pm','pm','pm','pm','pm','pm','pm'};

%day2
session_data(2).Day = '20210616';
session_data(2).folders={'111251','111821','112750','113117','113759','114449',...
    '134638','135318','135829','140842','141045','141459','143736'};
session_data(2).folder_type={'I','I','O','O','O','B','I','I','I','O','O','B','B'};
session_data(2).AM_PM = {'am','am','am','am','am','am','pm','pm','pm','pm','pm','pm','pm'};


% day 3
session_data(3).Day = '20210623';
session_data(3).folders={'110830','111416','111854','112823','113026',...
    '133244','133928','134357','135435','135630','135830','140530','142530','142723'};
session_data(3).folder_type={'I','I','I','O','O','I','I','I','O','O','O','B','B','B'};
session_data(3).AM_PM = {'am','am','am','am','am','pm','pm','pm','pm','pm','pm','pm',...
    'pm','pm'};



% day 4
session_data(4).Day = '20210625';
session_data(4).folders={'111134','112108','112805','113645','114239','132902',...
    '134133','142139'};
session_data(4).folder_type={'I','I','I','O','B','O','B','B'};
session_data(4).AM_PM = {'am','am','am','am','am','pm','pm','pm'};

% day 5
session_data(5).Day = '20210630';
session_data(5).folders={'101857','102342','102825','103756','110415','133210',...
    '133905','134420','135813','140408'};
session_data(5).folder_type={'I','I','I','O','B','O','I','I','O','B'};
session_data(5).AM_PM = {'am','am','am','am','am','pm','pm','pm','pm','pm'};

% day 6
session_data(6).Day = '20210702';
session_data(6).folders={'135108','135915','140426','141920','142120','142320',...
    '142800','145811'};
session_data(6).folder_type={'I','I','I','O','O','O','B','B'};
session_data(6).AM_PM = {'pm','pm','pm','pm','pm','pm','pm','pm'};

% day 7
session_data(7).Day = '20210707';
session_data(7).folders={'103731','104916','105644','110518','111026','132803',...
    '133525','134019','135008'};
session_data(7).folder_type={'I','I','I','O','B','I','I','I','O'};
session_data(7).AM_PM = {'am','am','am','am','am','pm','pm','pm','pm'};

% need to get data for 20210709
% session_data(8).Day = '20210709';
% session_data(8).folders={'101301','102021','102634'};
% session_data(8).folder_type={'I','I','I'};
% session_data(8).AM_PM = {'am','am','am''am','am'};

% day 8
session_data(8).Day = '20210714';
session_data(8).folders={'101741','102514','103106','104621','132615','133137','133748',...
    '140047','140924','141733','142605','143752','145541','150310'};
session_data(8).folder_type={'I','I','I','O','B','B','B','B','B','I','I','O','O','O'};
session_data(8).AM_PM = {'am','am','am','am','pm','pm','pm','pm','pm','pm','pm',...
    'pm','pm','pm'};

% day 9
session_data(9).Day = '20210716';
session_data(9).folders={'102008','102726','103214','104134','104745','133339',...
    '133908','134306','134936'};
session_data(9).folder_type={'I','I','I','O','B','B','B','I','O'};
session_data(9).AM_PM = {'am','am','am','am','am','pm','pm','pm','pm'};

% day 10
session_data(10).Day = '20210728';
session_data(10).folders={'103034','103745','104244','105354','110143','132842','133727',...
    '134258'};
session_data(10).folder_type={'I','I','I','O','B','B','B','B'};
session_data(10).AM_PM = {'am','am','am','am','am','pm','pm','pm'};

% day 11
session_data(11).Day = '20210804';
session_data(11).folders={'102546','103317','103821','104809','105403','133255','134125',...
    '134652'};
session_data(11).folder_type={'I','I','I','O','B','B','B','B'};
session_data(11).AM_PM = {'am','am','am','am','am','pm','pm','pm'};

%day 12
session_data(11).Day = '20210806';
session_data(11).folders={'103003','103828','104406','105415','105859','110512','134206',...
    '134915','140110','140536','141223'};
session_data(11).folder_type={'I','I','I','O','O','B','B','B','B','B','B'};
session_data(11).AM_PM = {'am','am','am','am','am','am','pm','pm','pm','pm','pm'};




save session_data session_data -v7.3

%% analzying session specific data



%% using an MLP-AE to look at differences between imagined and online control
% projecting to the latent space here works best for 4 of the 7 actions and
% not all simultaneously due to noise

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim')
addpath(genpath(pwd))

% Imagined movement data
%folders={'110604','111123','111649'};%20210615
%folders={'134638','135318','135829'};%20210616
%folders={'133244','133928','134357'};%20210623
%folders={'111134','112108','112805'};%20210625
%folders={'102546','103317','103821'};%20210804
%folders={'103003','103828','104406'};%20210806
folders={'135108','135915','140426'};%20210702
day_date = '20210702';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'Imagined');
    files = [files;findfiles('',folderpath)'];
end


%%%% have to do procustus when mapping data from one session to another

% online data
%folders = {'113524','113909','114318','114537'};%20210615
%folders = {'140842','141045'};
%folders={'135435','135630','135830'};%20210623
%folders={'113645','114239'};
%folders={'112750','113117','113759'};%20210616
%folders={'132615','133137','133748','140047','140924'};%20210714
%folders={'113645','114239','132902','134133','142139'};%20210625
%folders={'133255','134125','134652'};%20210804
%folders={'105415','105859','110512'};%20210806
folders={'141920','142120','142320', '142800','145811'};%20210702

day_date = '20210702';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end

%load the data
condn_data = load_data_for_MLP(files);

% build the AE based on MLP and only for hG
%[net,Xtrain,Ytrain] = build_mlp_AE(condn_data);
[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);


% now build a classifier on the outerlayers

% get activations in deepest layer but averaged over a trial
TrialZ=[];
idx=[];
imag=0;
for i=1:length(files)
    disp(i)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        if imag==0
            counter=TrialData.Params.ClickCounter;
            kinax=kinax(end-counter+1:end);
        end
        temp = cell2mat(features(kinax));
        chmap = TrialData.Params.ChMap;
        X = bci_pooling(temp,chmap);

        %2-norm the data
        for j=1:size(X,2)
            X(:,j)=X(:,j)./norm(X(:,j));
        end

        % feed it through the AE
        X = X(1:96,:);
        Z = activations(net,X','autoencoder');
        if imag==0
            if TrialData.SelectedTargetID == TrialData.TargetID
                %Z = Z(:,end-4:end);
                TrialZ = [TrialZ Z];
                idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
                %Z = mean(Z,2);
                %TrialZ = [TrialZ Z];
                %idx=[idx TrialData.TargetID];
            end
        else
            TrialZ = [TrialZ Z];
            idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
        end
    end
end

% plot the trial averaged activity in the latent space
Z=TrialZ;
%[c,s,l]=pca(Z');
%Z=s';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    %if i==1||i==6||i==7||i==4||i==2
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:),'MarkerSize',20);
    %end
end
xlabel('Latent 1')
ylabel('Latent 2')
zlabel('Latent 3')

if imag==1
    title('Imagined Latent Space')
else
    title('Proj. Online Data through Latent Space')
end
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

% get pairwise mahalanbois distance
len = length(unique(idx));
D = zeros(len);
for i=1:len
    idxx = find(idx==i);
    A=Z(:,idxx);
    for j=i+1:len
        idxx = find(idx==j);
        B=Z(:,idxx);
        D(i,j) = mahal2(A',B',2);
        D(j,i) = D(i,j);
    end
end
dist_online = squareform(D);

figure;boxplot([dist_imagined' dist_online'])
box off
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Distance')
title('Inter-class distances')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

[h p tb st]=ttest(dist_imagined,dist_online)
mean([dist_imagined' dist_online'])



%% using an MLP-AE + decoding layer for class specific latent differences
%to look at differences between imagined and online control
% this works only when updating the decoding layer

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\Nikhlesh\Documents\GitHub\ECoG_BCI_HighDim'))


% Imagined movement data
folders={'111251','111821'};
%folders={'134638','135318','135829'};
%folders={'133244','133928','134357'};%20210623
%folders={'111134','112108','112805'}
day_date = '20210616';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'Imagined');
    files = [files;findfiles('',folderpath)'];
end


%%%% have to do procustus when mapping data from one session to another

% online data
folders = {'112750','113117','113759'};
%folders = {'140842','141045'};
%folders={'135435','135630','135830'};%20210623
%folders={'113645','114239'};
day_date = '20210616';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end

% batch update files
folders = {'114449'};
day_date = '20210616';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end

%load the data
condn_data = load_data_for_MLP(files);

% build the AE based on MLP and only for hG
[net,Xtrain,Ytrain] = build_mlp_AE(condn_data);

% re-build the AE latent space after changing wts thru a softmax layer
[net1] =  add_decoding_AE(net,condn_data);


% perform a batch update: update the softmax weights using new online data
[net2] =  add_decoding_AE_batch(net,net1,condn_data);

% get activations from the latent space
TrialZ=[];
idx=[];
imag=0;
batch=0;
data_correct=[];
for i=1:length(files)
    disp(i)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        if imag==0
            counter=TrialData.Params.ClickCounter;
            kinax=kinax(end-counter+1:end);
        end
        temp = cell2mat(features(kinax));
        chmap = TrialData.Params.ChMap;
        X = bci_pooling(temp,chmap);

        %2-norm the data
        for j=1:size(X,2)
            X(:,j)=X(:,j)./norm(X(:,j));
        end

        %feed it through the AE
        X = X(1:32,:);
        Z = activations(net,X','autoencoder');

        % store


        % pass it next through softmax layer
        %         if batch==0
        %             Z = activations(net1,Z','Classif');
        %         else
        %             Z = activations(net2,Z','Classif');
        %         end

        % straight pass thru softmax layer
        %Z=activations(net1,X','Classif');

        if imag==0
            if TrialData.SelectedTargetID == TrialData.TargetID
                TrialZ = [TrialZ Z];
                idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
                %Z = mean(Z,2);
                %idx=[idx TrialData.TargetID];
            end
        else
            TrialZ = [TrialZ Z];
            idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
        end
    end
end



% plot the trial averaged activity in the latent space
Z=TrialZ;
%[c,s,l]=pca(Z');
%Z=s';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    %if i==1||i==6||i==7||i==4||i==3
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:),'MarkerSize',20);
    %end
end
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')

if imag==1
    title('Imagined Decoder Space')
else
    if batch==0
        title('Proj. Online Data through Imagined Decoder Space')
    else
        title('Proj. Online Data through Imagined Decoder Space- BATCH')
    end
end
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

% get pairwise mahalanbois distance
len = length(unique(idx));
D = zeros(len);
for i=1:len
    idxx = find(idx==i);
    A=Z(:,idxx);
    for j=i+1:len
        idxx = find(idx==j);
        B=Z(:,idxx);
        D(i,j) = mahal2(A',B',2);
        D(j,i) = D(i,j);
    end
end
dist_batch = squareform(D);

figure;boxplot([dist_imagined' dist_online' dist_batch'])
box off
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Distance')
title('Inter-class distances')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

[h p tb st]=ttest(dist_imagined,dist_online)


%% CONTINUING WITH APPROACH ABOVE, LOOKING AT LATENT SPACE ACROSS DAYS
% build imagined space on day 1, project dat 2 data onto it..
% cumulatively build the spaces up to see when it all stabilizes

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim')
addpath(genpath(pwd))
addpath('C:\Users\nikic\Documents\MATLAB')

% get the folder names etc
load session_data


% Imagined movement data
folders={'110604','111123','111649'};%'20210615'
%folders={'134638','135318','135829'};20210616
%folders={'133244','133928','134357'};%20210623
%folders={'111134','112108','112805'}
day_date = '20210615';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'Imagined');
    files = [files;findfiles('',folderpath)'];
end


%%%% have to do procustus when mapping data from one session to another

% online data
%folders = {'113524','113909','114318','114537'};%20210615
%folders = {'140842','141045'};
%folders={'135435','135630','135830'};%20210623
%folders={'112750','113117','113759'};%20210616
%folders={'140842','141045','141459','143736'};%20210616
%folders={'113645','114239'};%'20210625'
%folders={'113645','114239','132902',...
%'134133','142139'};%20210625
folders={'141920','142120','142320','142800','145811'};%'20210702'


day_date = '20210702';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end

%ratio of correct bins to total bins in online data
res = ratio_correct_bins(files);
mean(res.data)

%load the data
condn_data = load_data_for_MLP(files);

% build the AE based on MLP and only for hG
[net,Xtrain,Ytrain] = build_mlp_AE(condn_data);

% now build a classifier on the outerlayers

% get activations in deepest layer but averaged over a trial
TrialZ=[];
idx=[];
imag=0;
for i=1:length(files)
    disp(i)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        if imag==0
            counter=TrialData.Params.ClickCounter;
            kinax=kinax(end-counter+1:end);
        end
        temp = cell2mat(features(kinax));
        chmap = TrialData.Params.ChMap;
        X = bci_pooling(temp,chmap);

        %2-norm the data
        for j=1:size(X,2)
            X(:,j)=X(:,j)./norm(X(:,j));
        end

        % feed it through the AE
        X = X(65:96,:);
        %X = X(1:96,:);
        Z = activations(net,X','autoencoder');
        if imag==0
            if TrialData.SelectedTargetID == TrialData.TargetID
                TrialZ = [TrialZ Z];
                idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
                %Z = mean(Z,2);
                %idx=[idx TrialData.TargetID];
            end
        else
            TrialZ = [TrialZ Z];
            idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
        end
    end
end

% plot the trial averaged activity in the latent space
Z=TrialZ;
%[c,s,l]=pca(Z');
%Z=s';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    %if i==1||i==6||i==7||i==4||i==2
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:),'MarkerSize',20);
    %end
end
xlabel('Latent 1')
ylabel('Latent 2')
zlabel('Latent 3')

if imag==1
    title('Imagined Latent Space')
else
    title('Proj. Online Data through Imagined Latent Space')
end
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

% get silhoutte index
%silh = silhouette(Z',idx');
%figure;hist(silh)



% get pairwise mahalanbois distance
len = length(unique(idx));
D = zeros(len);
for i=1:len
    idxx = find(idx==i);
    A=Z(:,idxx);
    for j=i+1:len
        idxx = find(idx==j);
        B=Z(:,idxx);
        D(i,j) = mahal2(A',B',2);
        D(j,i) = D(i,j);
    end
end
dist_online = squareform(D);

figure;boxplot([dist_imagined' dist_online'])
box off
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Distance')
title('Inter-class distances')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

[h p tb st]=ttest(dist_imagined,dist_online)


%% PUTTING IT ALL TOGETHER IN TERMS OF BUILDING THE AE
%(MAIN)

clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data
dist_online_total=[];
dist_imag_total=[];
var_imag_total=[];
mean_imag_total=[];
var_online_total=[];
mean_online_total=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);

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
    condn_data = load_data_for_MLP(files);

    % build the AE based on MLP and only for hG
    [net,Xtrain,Ytrain] = build_mlp_AE(condn_data);
    %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);

    % get activations in deepest layer but averaged over a trial
    imag=1;
    [TrialZ_imag,dist_imagined,mean_imagined,var_imagined] = get_latent(files,net,imag);
    dist_imag_total = [dist_imag_total;dist_imagined];
    mean_imag_total=[mean_imag_total;pdist(mean_imagined)];
    var_imag_total=[var_imag_total;var_imagined'];

    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for i=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    condn_data = load_data_for_MLP(files);

    % get activations in deepest layer
    imag=0;
    [TrialZ_online,dist_online,mean_online,var_online] = get_latent(files,net,imag);
    dist_online_total = [dist_online_total;dist_online];
    mean_online_total=[mean_online_total;pdist(mean_online)];
    var_online_total=[var_online_total;var_online'];

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

    [h p tb st]=ttest(dist_imagined,dist_online);
    disp([p mean([dist_imagined' dist_online'])]);
end

figure;
plot(mean(dist_online_total'))
set(gcf,'Color','w')
title('Across Day Learning')
ylabel('Mahalanobis Dist.')
xlabel('Day')
xlim([0.5 11.5])
tmp = mean(dist_online_total');
figure;
tmp1 = tmp(1:4);
tmp2 = tmp(5:end);
tmp1(end+1:length(tmp2))=NaN;
boxplot([tmp1' tmp2'])
xticklabels({'Early Days','Late Days'})
ylabel('Mahalanobis Dist')
title('Online Bins proj. thru Imagined Manifold')
set(gcf,'Color','w')


tmp = mean(dist_imag_total');
figure;
tmp1 = tmp(1:4);
tmp2 = tmp(5:end);
tmp1(end+1:length(tmp2))=NaN;
boxplot([tmp1' tmp2'])
xticklabels({'Early Days','Late Days'})
ylabel('Mahalanobis Dist')
title('Imag Bins proj. thru Imagined Manifold')
set(gcf,'Color','w')

figure;
boxplot([dist_imag_total(1,:)' dist_online_total(1,:)' ])


%plotting the difference in manifold angles between mvmt and time
ang = [mean(dist_imag_total,2) mean(dist_online_total,2)];
%ang=fliplr(ang);
figure;hold on
%scatter(ones(length(ang),1)+0.05*randn(length(ang),1),ang(:,1));
%scatter(2*ones(length(ang),1)+0.05*randn(length(ang),1),ang(:,2));
idx = 0.05*randn(length(ang),2);
%idx = zeros(size(idx));
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=winter(length(ang));
for i=1:length(ang)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
end
ylim([5 65])
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Mahalanobis distance')
set(gca,'FontSize',14)

%plotting changes in the variance of the latent distributions
figure;boxplot([var_imag_total(:) var_online_total(:)])
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Variance in latent space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
set(gca,'LineWidth',2)
box off

ang=[mean(var_imag_total,2) mean(var_online_total,2)];
figure;hold on
idx = 0.05*randn(length(ang),2);
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=parula(length(ang));
for i=1:length(ang)
    %plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',col(i,:),'LineWidth',1)
end
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Variance in latent space')
set(gca,'FontSize',14)
[h p tb st]=ttest(ang(:,1),ang(:,2))

figure;
plot(ang(:,2)-ang(:,1),'.k','MarkerSize',20)
tmp=ang(:,2)-ang(:,1);

% plotting changes in the mean distance between distributions over learning
figure;boxplot([mean_imag_total(:) mean_online_total(:)])
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Variance in latent space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
set(gca,'LineWidth',2)
box off

ang=[mean(mean_imag_total,2) mean(mean_online_total,2)];
figure;hold on
idx = 0.05*randn(length(ang),2);
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=parula(length(ang));
for i=1:length(ang)
    %plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',col(i,:),'LineWidth',1)
end
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Distance b/w means in latent space')
set(gca,'FontSize',14)
[h p tb st]=ttest(ang(:,1),ang(:,2))

figure;
plot(ang(:,2)-ang(:,1),'.k','MarkerSize',20)
tmp=ang(:,2)-ang(:,1);
[bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp,1);
hold on
plot([ (1:length(tmp))'],...
    [ ones(size(tmp,1),1) (1:length(tmp))']*bhat,'k','LineWidth',1);
xlim([0.5 12])
xlabel('Days')
ylabel('Delta Online vs. Imagined')
title('Mean Separation in Latent Space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
box off

% boostrapped test
bhat_boot=[];
parfor iter=1:500
    x=1:length(tmp);
    x=x(randperm(length(x)));
    [bhat1 p wh se ci t_stat]=robust_fit(x',tmp,1);
    bhat_boot(iter)=bhat1(2);
end
sum(bhat_boot>bhat(2))/length(bhat_boot)

x= [ ones(size(tmp,1),1) (1:length(tmp))'];
y = tmp;
[B,BINT,R,RINT,STATS] = regress(y,x);
STATS(3)



%save bci_manifold_results_learning -v7.3




%% getting the proportion of correct bins within a session
% goal here is to see if there is learning i.e. the ratio of correct
% bins increases daily

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
load session_data
acc_bins_ratio=[];
acc_lower_bound=[];
acc_upper_bound=[];
for i=1:length(session_data)
    disp(100*i/length(session_data))
    folder_names = session_data(i).folders;
    folder_type = session_data(i).folder_type;
    folder_am = session_data(i).AM_PM;
    idx=[];
    for j=1:length(folder_type)
        if i~=6 && i~=10
            if strcmp(folder_type{j},'O')  && strcmp(folder_am{j},'am')
                idx=[idx j];
            end
        elseif i==10
            if strcmp(folder_type{j},'O') || strcmp(folder_type{j},'B') && strcmp(folder_am{j},'am')
                idx=[idx j];
            end
        else
            if strcmp(folder_type{j},'O') %|| strcmp(folder_type{j},'B')
                idx=[idx j];
            end
        end
    end
    folders = folder_names(idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    %ratio of correct bins to total bins in online data
    res = ratio_correct_bins(files);
    acc_bins_ratio(i) = median(res.data);
    bb=sort(bootstrp(1000,@median,res.data));
    acc_lower_bound(i)=bb(25);
    acc_upper_bound(i)=bb(975);
end

figure;plot(acc_bins_ratio,'k','LineWidth',1')
ylim([0 1.05])
xlim([0 11.5])
set(gcf,'Color','w')
box off
xticks(1:length(session_data))
xticklabels(1:length(session_data))
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
xlabel('Days')
ylabel('Median Accuracy with Mode Filter')
title('Performance of same-day trained decoder')
hold on
hline(1/7,'r')
%plot(acc_lower_bound,'--k')
%plot(acc_upper_bound,'--k')

% even when using the mean accuracy, the later days have sig. more accuracy
% than the earlier days in the online task
figure;boxplot([acc_bins_ratio(1:4)' acc_bins_ratio(end-3:end)'])

% can one visualize this in latent space? I.e., the variability in latent
% space has to be reducing in later days as compared to early days.

% build AE for early days on imagined, feed in online, all bins, color code
% by incorrect bins, get variance and mahab distance. Then do same for
% later days and get variance of clouds and mahab distance between imagined
% actions.

% first four days: build AE on imagined, then feed in data from online

%% HOW DOES MANIFOLD CHANGE ACROSS DAYS...REPRESENTATIONAL DRIFT
% Representational drift across days, how does latent space spread across
% days? Is it crystal clear and spaces are set but we just need more data
% across days to fill it up? The manifold rotates and changes across days,
% some drift due to baseline issues etc. and you need more days to capture
% the complexity while plasticity is happening at the same time? Use
% procrustus analyses here

%one way to do that is to project day two data onto day ones data and see
%whether projection lies within same pace in latent space i.e., how much
%procrustus distance between spaces across days...

clc;clear;


root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
load session_data
addpath 'C:\Users\nikic\Documents\MATLAB'
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
files=[];
files1=[];

for i=2:3


    day_date = session_data(i).Day;
    for ii=1:length(session_data(i).folders)
        if strcmp(session_data(i).folder_type(ii),'I')
            folderpath =  fullfile(root_path,day_date,'Robot3DArrow',...
                session_data(i).folders{ii},'Imagined');
            files = [files;findfiles('',folderpath)'];
        else
            folderpath1 =  fullfile(root_path,day_date,'Robot3DArrow',...
                session_data(i).folders{ii},'BCI_Fixed');
            files1 =[files1;findfiles('',folderpath1)'];
        end
    end

end

% keep apart 20% of files for testing
idx_train = randperm(length(files1),round(0.80*length(files1)));
I = ones(size(files1));
I(idx_train)=0;
idx_test = find(I==1);
files_train = files1(idx_train);
files_test = files1(idx_test);

files_train = [files ;files_train];



% build an AE latent space for this model uaing just the imagined data
condn_data = load_data_for_MLP(files_train);
[net,Xtrain,Ytrain] = build_mlp_AE_supervised_total(condn_data);

% now project data from own day (held out) onto the manifold
imag=0;
[TrialZ_online,dist_online,mean_online,var_online,Zidx,acc] = get_latent(files_test,net,imag);
figure;
hold on
tid=find(Zidx==1);
plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.b','MarkerSize',20)
tid=find(Zidx==5);
plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.r','MarkerSize',20)
tid=find(Zidx==7);
plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.g','MarkerSize',20)
axis tight
xx = get(gca,'xlim');
yy = get(gca,'ylim');
zz = get(gca,'zlim');
view(47,31)
set(gcf,'Color','w')
xlabel('Latent 1')
ylabel('Latent 2')
zlabel('Latent 3')
set(gca,'FontSize',14)
title(['AE through Day ' num2str(i) ' acc of ' num2str(mean(diag(acc)))])
%title(['AE through Day 2-3 acc of ' num2str(nanmean(diag(acc)))])
%set(gcf,'Position',[680,50,934,946])


%load all of the online data from the all the consecutive days
acc_overall=[];
mahab_dist=[];
I = ones(length(session_data),1);
I(2:3)=0;
I=find(I==1);
for k=1:length(I)
    j=I(k);
    files=[];
    files1=[];
    day_date = session_data(j).Day;
    for ii=1:length(session_data(j).folders)
        if strcmp(session_data(j).folder_type(ii),'I')
            folderpath =  fullfile(root_path,day_date,'Robot3DArrow',...
                session_data(j).folders{ii},'Imagined');
            files = [files;findfiles('',folderpath)'];
        else
            folderpath1 =  fullfile(root_path,day_date,'Robot3DArrow',...
                session_data(j).folders{ii},'BCI_Fixed');
            files1 =[files1;findfiles('',folderpath1)'];
        end
    end
    
    % now project data from second day onto the manifold from the first
    % day
    imag=0;
    [TrialZ_online2,dist_online2,mean_online2,var_online2,Zidx2,acc1] = get_latent(files1,net,imag);    
    close
    mahab_dist = [mahab_dist;mean(dist_online2)];

    % plot the mahab distribution 
    figure;
    subplot(2,1,1)
    boxplot([dist_online' dist_online2'])
    xticks(1:2)
    xticklabels({'Day 2-3', ['Day ' num2str(j)]})
    set(gcf,'Color','w')
    set(gca,'FontSize',14)
    set(gca,'LineWidth',1)
    ylabel('Mahalanobis distance')

    %plotting just condition specific data
    subplot(2,1,2)
    hold on
    tid2=find(Zidx2==1);
    plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.b','MarkerSize',20)
    tid2=find(Zidx2==5);
    plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.r','MarkerSize',20)
    tid2=find(Zidx2==7);
    plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.g','MarkerSize',20)
    xlim(xx)
    ylim(yy)
    zlim(zz)
    view(47,31)
    set(gcf,'Color','w')
    xlabel('Latent 1')
    ylabel('Latent 2')
    zlabel('Latent 3')
    set(gca,'FontSize',14)
    tmp=(diag(acc1));
    %title(['Day ' num2str(j) ' thru AE Day ' num2str(i) ' with acc ' num2str(mean(tmp))])
    title(['Day ' num2str(j) ' thru AE Day 2-3 with acc ' num2str(mean(tmp))])
    acc_overall =[acc_overall diag(acc1)];
    set(gcf,'Position',[680,50,934,946])
end

acc_overall_compar = [diag(acc) acc_overall];
figure;
boxplot(acc_overall_compar)

figure;boxplot(mahab_dist)
hline(mean(dist_online),'r')
ylim([0 35])
xlim([0.5 1.5])
xticks(1)
xticklabels('Held out days')
ylabel('Mahab Distance')
set(gca,'FontSize',14)
set(gcf,'Color','w')

%80.56% accuracy of day 1 on its own data on top condition 1,5,7
% 
% % old code when looking at things from a day to day basis
% j=4;
% files=[];
% files1=[];
% day_date = session_data(j).Day;
% for ii=1:length(session_data(j).folders)
%     if strcmp(session_data(j).folder_type(ii),'I')
%         folderpath =  fullfile(root_path,day_date,'Robot3DArrow',...
%             session_data(j).folders{ii},'Imagined');
%         files = [files;findfiles('',folderpath)'];
%     else
%         folderpath1 =  fullfile(root_path,day_date,'Robot3DArrow',...
%             session_data(j).folders{ii},'BCI_Fixed');
%         files1 =[files1;findfiles('',folderpath1)'];
%     end
% end
% % now project data from second day onto the manifold from the first
% % day
% imag=0;
% [TrialZ_online2,dist_online2,mean_online2,var_online2,Zidx2,acc1] = get_latent(files1,net,imag);
% 
% % plotting the data on the manifold
% figure;
% hold on
% plot3(TrialZ_online(1,:),TrialZ_online(2,:),TrialZ_online(3,:),'.','MarkerSize',20)
% plot3(TrialZ_online2(1,:),TrialZ_online2(2,:),TrialZ_online2(3,:),'.','MarkerSize',20)
% figure;boxplot([dist_online' dist_online2'])
% 
% %plotting just condition specific data
% figure;
% hold on
% tid=find(Zidx==1);
% plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.b','MarkerSize',20)
% tid=find(Zidx==5);
% plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.r','MarkerSize',20)
% tid=find(Zidx==7);
% plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.g','MarkerSize',20)
% axis tight
% xx = get(gca,'xlim');
% yy = get(gca,'ylim');
% zz = get(gca,'zlim');
% view(-134,36)
% set(gcf,'Color','w')
% xlabel('Latent 1')
% ylabel('Latent 2')
% zlabel('Latent 3')
% set(gca,'FontSize',14)
% figure;hold on
% tid2=find(Zidx2==1);
% plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.b','MarkerSize',20)
% tid2=find(Zidx2==5);
% plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.r','MarkerSize',20)
% tid2=find(Zidx2==7);
% plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.g','MarkerSize',20)
% xlim(xx)
% ylim(yy)
% zlim(zz)
% view(-134,36)
% set(gcf,'Color','w')
% xlabel('Latent 1')
% ylabel('Latent 2')
% zlabel('Latent 3')
% set(gca,'FontSize',14)
% tmp=(diag(acc1));
% title(['Day ' num2str(j) ' thru AE 78 with acc ' num2str(mean(tmp([1 5 7])))])
% 
% % plotting
% 







%% LOOKING AT WHETHER THE PRIN ANGLES OF MANIFOLDS CHANGE FROM IMAGINED TO ONLINE (7DOF)

% a few options
% FIRST is to take distribution of prin angles between imagined files per
% action to other imagined files as compared to prin angles between
% imagined files and online files
% SECOND is to compare the angles between the 7 actions and see if there is
% greater separation during online vs imagined

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
load session_data

for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);

    %%%%%%imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end


    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files1=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files1 = [files1;findfiles('',folderpath)'];
    end

    % get the principal angles between the various commands
    prin_angles = get_prin_angles(files,files1);

end





%% COLLATE AE ACTIVATION DURING PNP DAYS
% INVOLVES FIRST BUILDING THE AE SPACE FOR ALL DATA FROM FIRST 11 DAYS


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
load session_data

% option of maybe aligning all the data to the first day using procrustus
files=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);


    %%%%%%imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    %%%%%%batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end
end

% build a massive autoencoder by loading all the data from all 11day files
condn_data = load_data_for_MLP(files);
[net,Xtrain,Ytrain] = build_mlp_AE_supervised_total(condn_data);

% save the AE
net_AE_total=net;
save net_AE_total net_AE_total



%% LOOKING AT TEMPORAL DYNAMICS OF AE LATENT SPACE ACTIVATION IN THE PATH TASK

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
foldernames={'20210825','20210827'} %20210818

% load the total AE
load net_AE_total

% get the files
files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},...
        'Robot3DPath');
    files = [files;findfiles('',folderpath,1)'];
end

% remove persistence files
files1=[];
for i=1:length(files)
    if ~isempty(regexp(files{i},'Data'))
        files1=[files1;files(i)];
    end
end
files=files1;

% get all the paths
Params.Paths{1} = [[200, -200, 0]; [0, -200, 0]; [0,0,0];  [-200, 0,0]; [-200, 200, 0]];
Params.Paths{2} = [[-200, 200, 0]; [-200, 0,0]; [0,0,0]; [0, -200, 0]; [200, -200, 0]];
Params.Paths{3} = [[200,0,-100]; [200, 0, 100]; [0, 0, 100];[0, 0, -100]; [-200, 0, -100]];
Params.Paths{4} = [ [-200, 0, -100]; [0, 0, -100]; [0, 0, 100];[200, 0, 100]; [200,0,-100]];
Params.Paths{5} = [[200, -100, 0]; [0, -100, 0]; [-141.4, 41.4, 0]; [0,180, 0]; [200, 180, 0]];
Params.Paths{6} = [[200, 180, 0];[0,180, 0]; [-141.4, 41.4, 0]; [0, -100, 0];[200, -100, 0]];

% load a specific file and look at latent activity
load(files{1})
target_id=TrialData.TargetID;
target_pos = TrialData.TargetPosition;
target_path = Params.Paths{target_id};
%target_path=flipud(target_path);
%target_path = cumsum(target_path);
start_pos = TrialData.CursorState(:,1);
kin = TrialData.CursorState;
kinax = TrialData.TaskState;
kinax = [find(kinax==3)];
kin=kin(:,kinax);

% plot the data
cmap=turbo(size(kin,2));
figure;
for i=1:size(kin,2)
    plot3(kin(1,i),kin(2,i),kin(3,i),'.','MarkerSize',30,'Color',cmap(i,:))
    hold on
end
plot3(kin(1,1),kin(2,1),kin(3,1),'xg','MarkerSize',30)
plot3(target_pos(1),target_pos(2),target_pos(3),'xr','MarkerSize',30)
for i=size(target_path,1):-1:1
    plot3(target_path(i,1),target_path(i,2),target_path(i,3),'ob','MarkerSize',30)
end

%%%%% project data into latent space
clicker_output = TrialData.FilteredClickerState;
features  = TrialData.SmoothedNeuralFeatures;
temp = cell2mat(features(kinax));
chmap = TrialData.Params.ChMap;
X = bci_pooling(temp,chmap);
%2-norm the data
for j=1:size(X,2)
    X(:,j)=X(:,j)./norm(X(:,j));
end
% feed it through the AE
X = X(1:96,:);
Z = activations(net_AE_total,X','autoencoder');
figure;hold on
view(56,35)
%cmap=parula(size(Z,2));
cmap=parula(8);
for i=1:size(Z,2)
    decode = clicker_output(i)+1;
    plot3(Z(1,i),Z(2,i),Z(3,i),'.','MarkerSize',20,'Color',cmap(decode,:))
    pause(0.05)
end

% plot video together
v = VideoWriter('PathTrial2.avi');
v.FrameRate=12;
open(v);
figure;
set(gcf,'Color','w')
subplot(2,1,1)
hold on
xlim([min(kin(1,:)) max(kin(1,:))])
ylim([min(kin(2,:)) max(kin(2,:))])
%zlim([min(kin(3,:)) max(kin(3,:))])
zlim([-1 1])
for ii=size(target_path,1):-1:1
    plot3(target_path(ii,1),target_path(ii,2),target_path(ii,3),'ob','MarkerSize',30)
end
hold on
plot3(target_pos(1),target_pos(2),target_pos(3),'xr','MarkerSize',30)
view(56,35)
axis tight
cmap=parula(7);
for i=1:size(Z,2)

    subplot(2,1,1)
    plot3(kin(1,i),kin(2,i),kin(3,i),'.k','MarkerSize',30)
    hold on

    subplot(2,1,2)
    view(56,35)
    xlim([-1 4])
    ylim([-1 3])
    zlim([-0.5 2])
    decode = clicker_output(i);
    if decode>0
        plot3(Z(1,i),Z(2,i),Z(3,i),'.','MarkerSize',20,'Color',cmap(decode,:))
        hold on
    end
    pause(0.05)

    frame=getframe(gcf);
    writeVideo(v,frame);
end
close(v)

%% LOOKING AT WHETHER THE MANIFOLDS CHANGE FROM IMAGINED TO ONLINE (HAND)

