

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
session_data(12).Day = '20210806';
session_data(12).folders={'103003','103828','104406','105415','105859','110512','134206',...
    '134915','140110','140536','141223'};
session_data(12).folder_type={'I','I','I','O','O','B','B','B','B','B','B'};
session_data(12).AM_PM = {'am','am','am','am','am','am','pm','pm','pm','pm','pm'};




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
%(MAIN MAIN)

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data
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
session_data = session_data([1:9 11]); % removing bad days
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    %folders_online = logical((strcmp(session_data(i).folder_type,'B')) + (strcmp(session_data(i).folder_type,'O')));
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

    %     % save the data
    %     filename = ['condn_data_Imagined_Day' num2str(i)];
    %     save(filename, 'condn_data', '-v7.3')

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
    condn_data = load_data_for_MLP(files);

    %     % save the data
    %     filename = ['condn_data_Online_Day' num2str(i)];
    %     save(filename, 'condn_data', '-v7.3')

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
    condn_data = load_data_for_MLP(files);

    %     % save the data
    %     filename = ['condn_data_Batch_Day' num2str(i)];
    %     save(filename, 'condn_data', '-v7.3')

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

close all

save mahab_distances_Full_B1 -v7.3

% compare and contrast the reconstruction with the original on an average
% basis
% to reconstruct data back
for i=1:7
    pooled_data = condn_data{i};
    pooled_data = pooled_data(:,3:3:end);
    tmp = mean(pooled_data,1);
    pooled_data_orig=reshape(tmp,8,4)';

    pooled_data_ae = condn_data_recon_online{i};
    tmp = mean(pooled_data_ae,1);
    pooled_data_recon=reshape(tmp,8,4)';

    clims_min = min([pooled_data_recon(:); pooled_data_orig(:)]);
    clims_max = max([pooled_data_recon(:); pooled_data_orig(:)]);

    figure;
    subplot(2,1,1)
    imagesc(pooled_data_orig)
    clim([clims_min clims_max])
    subplot(2,1,2)
    imagesc(pooled_data_recon)
    clim([clims_min clims_max])
    sgtitle(num2str(i))
end

% getting data from python, reconstructing back to full channel space and
% then plotting on B1 brain
% TRANSOFORMATION FRROM POOLED TO ORIGINAL
% x- > 2x-1,2x. y -> 2y-1,2y.
imaging_B1
figure;
imagesc(tmp3)
recon_data = zeros(size(chmap));
for i=1:size(tmp3,1) % going down the rows
    for j=1:size(tmp3,2) % going across the cols
        a=tmp3(i,j);
        i_idx = [2*i-1,2*i];
        j_idx = [2*j-1,2*j];
        recon_data(i_idx(1), j_idx(1))=a;
        recon_data(i_idx(1), j_idx(2))=a;
        recon_data(i_idx(2), j_idx(1))=a;
        recon_data(i_idx(2), j_idx(2))=a;
    end
end
figure;imagesc(recon_data)
recon_data = flipud(recon_data)';
recon_data = recon_data(:);
% plotting on brain as heat map
figure;
c_h = ctmr_gauss_plot(cortex,elecmatrix(1:128,:),recon_data(:),'lh',1,1,1);
e_h = el_add(elecmatrix([1:length(ch)],:), 'color', 'w', 'msize',2);
% plotting electrode sizes
val = linspace(min(recon_data),max(recon_data),128);
sz = linspace(1,10,128);
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
for j=1:length(recon_data)
    [aa,bb] = min(abs(val-recon_data(j)));
    ms = sz(bb)+1;
    e_h = el_add(elecmatrix(j,:), 'color', 'b','msize',ms);
end
set(gcf,'Color','w')


close all
figure;
hold on
plot(smooth(median(mahab_full_imagined(:,1:end)),2))
plot(smooth(median(mahab_full_online(:,1:end)),2))
plot(smooth(median(mahab_full_batch(:,1:end)),2))


figure;
hold on
plot((median(mahab_full_imagined(:,1:end))))
plot((median(mahab_full_online(:,1:end))))
plot((median(mahab_full_batch(:,1:end))))

clear tmp
w = [1/2 1/2];
tmp(:,1) = median(mahab_full_imagined(:,1:end));
tmp(:,2) = median(mahab_full_online(:,1:end));
tmp(:,3) = median(mahab_full_batch(:,1:end));


% mixed effect model for batch and Init seed slopes
day_name=[];
mahab_dist=[];
for i=1:size(tmp,1)
    day_name = [day_name;i;i];
    %day_name = [day_name;i];
    mahab_dist = [mahab_dist;tmp(i,2:3)'];
    %mahab_dist = [mahab_dist;tmp(i,1)'];
end
data = table(day_name,mahab_dist);
glm = fitglme(data,'mahab_dist ~ 1+ day_name');
%glm = fitlm(data,'mahab_dist ~ 1+ day_name');
stat = glm.Coefficients.tStat(2);
stat_boot=[];
for i=1:2000
    disp(i)
    day_name_tmp = day_name(randperm(numel(day_name)));
    data_tmp = table(day_name_tmp,mahab_dist);
    glm_tmp = fitglme(data_tmp,'mahab_dist ~ 1 + day_name_tmp');
    stat_boot(i) = glm_tmp.Coefficients.tStat(2);
end
figure;hist(stat_boot)
vline(stat)
sum(stat_boot>stat)/length(stat_boot)

% for i=1:size(tmp,2)
%     %xx = filter(w,1,[tmp(1,i) ;tmp(:,i)]);
%     xx = filter(w,1,[tmp(:,i) ;tmp(end,i)]);
%     tmp(:,i) = xx(2:end);
% end

% plotting with regression lines, mahab full
% plotting the regression for Mahab distance increases as a function of day
figure;
num_days=size(tmp,1);
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

%t-tests
[h p tb st]=ttest(tmp(:,1),tmp(:,2));p
[h p tb st]=ttest(tmp(:,1),tmp(:,3));p
[h p tb st]=ttest(tmp(:,3),tmp(:,2));p

% boxplots of mahab full
figure;
aa=([mahab_full_imagined(:) mahab_full_online(:) mahab_full_batch(:)]);
figure;boxplot(aa)


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


% stats on distances, early days and late days online mahalanobis distances
early_days_online = dist_online_total(1:4,:);
early_days_online=early_days_online(:);
late_days_online = dist_online_total(5:end,:);
late_days_online=late_days_online(:);
early_days_online(end+1:length(late_days_online))=NaN;
figure;boxplot([early_days_online late_days_online ]);


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
ang = [mean(dist_imag_total,2) mean(dist_online_total,2) ];
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
ylim([5 65]) %0 to 40 for regression
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Mahalanobis distance')
set(gca,'FontSize',14)

%plotting changes in the variance of the latent distributions
figure;boxplot(log([var_imag_total(:) var_online_total(:) var_batch_total(:)]))
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Variance in latent space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
set(gca,'LineWidth',2)
box off

ang=log([mean(var_imag_total,2) mean(var_batch_total,2)]);
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
figure;boxplot([mean_imag_total(:) mean_batch_total(:)])
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Mean Diff in latent space')
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
y = tmp(:,2);
[B,BINT,R,RINT,STATS] = regress(y,x);
STATS(3)


x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
y = tmp(:,2);
[B,BINT,R,RINT,STATS] = regress(y,x);
STATS(3)


figure;plot(tmp(:,2),'.','MarkerSize',20)


tmp=randn(10,3);
figure;plot(tmp(:,1),'.','MarkerSize',20)



%%%% REGRESSION LINES AS FUNCITON OF DAY %%%%%
clear tmp
w = [1/2 1/2 ];
tmp(:,1) = mean(mahab_full_imagined(:,1:end));
tmp(:,2) = mean(mahab_full_online(:,1:end));
tmp(:,3) = mean(mahab_full_batch(:,1:end));

for i=1:size(tmp,2)
    xx = filter(w,1,[tmp(1,i);tmp(:,i)]);
    %xx = filter(w,1,[tmp(:,i) ;tmp(end,i)]);
    tmp(:,i) = xx(2:end);
end


%save res_python_2D_B1 -v7.3

% stats on the Mahab distances
%organize it as experiment, day, mahab dist
data=[];
for i=1:length(tmp)
    exp_type = [1 2 3]';
    day_name = [i i i]';
    latent_mahab_dist = tmp(i,:)';
    data = [data ; [exp_type day_name latent_mahab_dist]];
end
data_latent_dist_B1 = data;
save data_latent_dist_B1 data_latent_dist_B1 -v7.3
anova1(tmp)
% getting mean separation
mean(tmp)
std(tmp)/sqrt(10)

% doing paired t-tests
[h p tb st]=ttest(tmp(:,3),tmp(:,2))
[h p tb st]=ttest(tmp(:,3),tmp(:,1))
[h p tb st]=ttest(tmp(:,2),tmp(:,1))

% plotting the regression for Mahab distance increases as a function of day
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




% plotting the robust regression for Mahab distance increases as a function of day
figure;
xlim([0 11])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:10,tmp(:,1),'.b','MarkerSize',20)
[bhat p1 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,1),2);
yhat = x*bhat;
plot(1:10,yhat,'b','LineWidth',1)
% online
plot(1:10,tmp(:,2),'.k','MarkerSize',20)
[bhat p2 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,2),2);
yhat = x*bhat;
plot(1:10,yhat,'k','LineWidth',1)
% batch
plot(1:10,tmp(:,3),'.r','MarkerSize',20)
[bhat p3 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,3),2);
yhat = x*bhat;
plot(1:10,yhat,'r','LineWidth',1)


% bootstrapped test for regression coefficient
y = tmp(:,3);
[B2,BINT,R,RINT,STATS2] = regress(y,x);
bhat_boot=[];
for i=1:2000
    ytmp = y(randperm(numel(y)));
    [B_tmp,~,Rtmp,~,STATS_tmp] = regress(ytmp,x);
    bhat_boot(i) = B_tmp(2);
end
figure;hist(bhat_boot)
vline(B2(2),'r')
sum(bhat_boot>B2(2))/length(bhat_boot)

% mixed effect model for batch and Init seed slopes
day_name=[];
mahab_dist=[];
for i=1:size(tmp,1)
    day_name = [day_name;i;i];
    %day_name = [day_name;i];
    mahab_dist = [mahab_dist;tmp(i,2:3)'];
    %mahab_dist = [mahab_dist;tmp(i,1)'];
end
data = table(day_name,mahab_dist);
glm = fitglme(data,'mahab_dist ~ 1+ day_name');
%glm = fitlm(data,'mahab_dist ~ 1+ day_name');
stat = glm.Coefficients.tStat(2);
stat_boot=[];
for i=1:2000
    disp(i)
    day_name_tmp = day_name(randperm(numel(day_name)));
    data_tmp = table(day_name_tmp,mahab_dist);
    glm_tmp = fitglme(data_tmp,'mahab_dist ~ 1 + day_name_tmp');
    stat_boot(i) = glm_tmp.Coefficients.tStat(2);
end
figure;hist(stat_boot)
vline(stat)
sum(stat_boot>stat)/length(stat_boot)


% plotting the stats for the variance in latent-space for B1. Load the data
% from python where each sampel represents the matrix determinant for each
% movement, and collated across days

% scatter
m1 = tmp(:,1);
m1b = sort(bootstrp(1000,@mean,m1));
m11 = mean(tmp(:,1));
m2 = tmp(:,2);
m2b = sort(bootstrp(1000,@mean,m2));
m22 = mean(tmp(:,2));
m3 = tmp(:,3);
m3b = sort(bootstrp(1000,@mean,m3));
m33 = mean(tmp(:,3));
x=1:3;
y=[mean(m1) mean(m2) mean(m3)];
neg = [y(1)-m1b(25) y(2)-m2b(25) y(3)-m3b(25)];
pos = [m1b(975)-y(1) m2b(975)-y(2) m3b(975)-y(3)];
figure;
hold on
cmap = brewermap(10,'Blues');
%cmap = (turbo(7));
x1=1+0.1*randn(size(m1));
scatter(x1,m1)
x1=2+0.1*randn(size(m2));
scatter(x1,m2)
x1=3+0.1*randn(size(m3));
scatter(x1,m3)
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',5,'Color','k','LineWidth',2,'MarkerFaceColor',[.25 .25 .25])
end
xlim([.5 3.5])
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
% ylim([0.5 1])
% yticks(0:.1:1)
set(gca,'FontSize',12)


% getting the mixed effect model stats for latent variance in  B1
%save latent_variance_B1 -v7.3
% fitting the data into a RM anova format
tmp=[tmp];
Data_RM_B2_latentVar=[];
for i=1:length(tmp)
    exp_type = [1 2 3]';
    subject = [i i i]';
    lat_var = tmp(i,:)';
    Data_RM_B2_latentVar = [Data_RM_B2_latentVar;[exp_type subject lat_dist]];
end
save Data_RM_B2_latentVar Data_RM_B2_latentVar -v7.3

% putting into a mixed effect model format
day_name=[];
latent_var=[];
exp_type=[];
subject=[];
I = 1:7:length(tmp);
for i=1:length(I)
    tmp1 = tmp(I(i):I(i)+6,:);
    for j=1:length(tmp1)
        exp_type = [exp_type;[1 2 3]'];
        latent_var = [latent_var;(tmp1(j,:))'];
        day_name = [day_name;[i i i]'];
        subject = [subject;[1 1 1]'];
    end
end
data=table(exp_type,latent_var,day_name,subject);
glme = fitglme(data,'latent_var ~ 1 + exp_type +(1|day_name)+(1|subject)')

% doing paired t-tests
I = sort([find(data.exp_type==3);find(data.exp_type==2)]);
data1=data(I,:);
glme = fitglme(data1,'latent_var ~ 1 + exp_type +(1|day_name)')




%save bci_manifold_results_learning -v7.3


%% (MAIN) combining stats across B1 and B2 for Mahab distances

clear;clc
%b2=load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2\res_python_B2_latent.mat');
b2=load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\res_python_B2_latent_pt2.mat');
b1=load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\res_python_2D_B1.mat');

tmp1=b1.tmp;
tmp2=b2.tmp;

mean(tmp2)
std(tmp2)/sqrt(length(tmp2))

subject=[];
exp_type=[];
mah_dist=[];
for i=1:length(tmp1)
    subject=[subject;[1 1 1]'];
    exp_type=[exp_type;[1 2 3]'];
    mah_dist=[mah_dist ;tmp1(i,:)'];
end
for i=1:length(tmp2)
    subject=[subject;[2 2 2]'];
    exp_type=[exp_type;[1 2 3]'];
    mah_dist=[mah_dist ;tmp1(i,:)'];
end
data = table(mah_dist,exp_type,subject);
glm = fitglme(data,'mah_dist ~ 1+ exp_type + (1|subject)')

% fitting the data into a RM anova format
tmp=[tmp2];
Data_RM_B1B2=[];
for i=1:length(tmp)
    exp_type = [1 2 3]';
    subject = [i i i]';
    lat_dist = tmp(i,:)';
    Data_RM_B1B2 = [Data_RM_B1B2;[exp_type subject lat_dist]];
end
save Data_RM_B1B2 Data_RM_B1B2 -v7.3


%% (MAIN) loading individual trial data for trainign autoencoders

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
res=[];
mahab_full_online=[];
mahab_full_imagined=[];
session_data = session_data([1:9 11]); % removing bad days
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    %folders_online = logical((strcmp(session_data(i).folder_type,'B')) + (strcmp(session_data(i).folder_type,'O')));
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

    %load the single trial data
    trial_data = load_data_for_MLP_TrialLevel(files);

    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

    %     % build the AE based on MLP and only for hG
    [net,Xtrain,Ytrain] = build_mlp_AE(condn_data);
    %     %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);
    %
    % get activations in deepest layer but averaged over a trial
    imag=1;
    [TrialZ_imag,dist_imagined,mean_imagined,var_imagined,idx_imag] = get_latent_regression(files,net,imag);
    dist_imag_total = [dist_imag_total;dist_imagined];
    mean_imag_total=[mean_imag_total;pdist(mean_imagined)];
    var_imag_total=[var_imag_total;var_imagined'];

    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the single trial data
    trial_data = load_data_for_MLP_TrialLevel(files);


    %load the data
    condn_data = load_data_for_MLP(files);

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];

    % get activations in deepest layer
    imag=0;
    [TrialZ_online,dist_online,mean_online,var_online,idx_online] = get_latent_regression(files,net,imag);
    dist_online_total = [dist_online_total;dist_online];
    mean_online_total=[mean_online_total;pdist(mean_online)];
    var_online_total=[var_online_total;var_online'];

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

    [h p tb st]=ttest(dist_imagined,dist_online);
    disp([p mean([dist_imagined' dist_online'])]);
    res=[res;[p mean([dist_imagined' dist_online'])]];
end


%% PUTTING IT ALL TOGETHER IN TERMS OF BUILDING THE AE WITH PROCRUSTES
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
res=[];
mahab_full_online=[];
mahab_full_imagined=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    %folders_online = strcmp(session_data(i).folder_type,'O');
    folders_online = logical((strcmp(session_data(i).folder_type,'B')) + (strcmp(session_data(i).folder_type,'O')));
    %if i~=6
    folders_am = strcmp(session_data(i).AM_PM,'am');
    folders_pm = strcmp(session_data(i).AM_PM,'pm');
    %folders_imag(folders_am==0)=0;
    %folders_online(folders_am==0)=0;
    %end

    imag_idx_am = find( (folders_imag .* folders_am)==1 );
    imag_idx_pm = find( (folders_imag .* folders_pm)==1 );
    online_idx_am = find( (folders_online .* folders_am)==1 );
    online_idx_pm = find( (folders_online .* folders_pm)==1 );

    %load morning session imagined
    condn_data_am=[];
    folders = session_data(i).folders(imag_idx_am);
    day_date = session_data(i).Day;
    files_am=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files_am = [files_am;findfiles('',folderpath)'];
    end
    if length(files_am)>0
        condn_data_am = load_data_for_MLP(files_am);
    end

    % load afternoon session imagined
    condn_data_pm=[];
    folders = session_data(i).folders(imag_idx_pm);
    day_date = session_data(i).Day;
    files_pm=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files_pm = [files_pm;findfiles('',folderpath)'];
    end
    if length(files_pm)>0
        condn_data_pm = load_data_for_MLP(files_pm);
    end

    % procrutus transformation between am and pm
    %     if length(condn_data_am)>0 &&  length(condn_data_pm)>0
    %         condn_data=[];
    %         for ii=1:length(condn_data_am)
    %             a=condn_data_am{ii};
    %             b=condn_data_pm{ii};
    %             if size(a,1) < size(b,1)
    %                 idx = randperm(size(b,1),size(a,1));
    %                 b1 = b(idx,:);
    %                 idx = randperm(size(a,1),size(a,1));
    %                 a1 = a(idx,:);
    %                 [d,z,transform] = procrustes(a1,b1,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %             elseif size(b,1) < size(a,1)
    %                 idx = randperm(size(a,1),size(b,1));
    %                 a1 = a(idx,:);
    %                 idx = randperm(size(b,1),size(b,1));
    %                 b1 = b(idx,:);
    %                 [d,z,transform] = procrustes(a1,b1,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %             elseif size(b,1) == size(a,1)
    %                 [d,z,transform] = procrustes(a,b,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %             end
    %             condn_data{ii} = [a;bout];
    %         end
    %     elseif length(condn_data_am) > 0 && length(condn_data_pm) == 0
    %         condn_data = condn_data_am;
    %     elseif length(condn_data_pm) > 0 &&  length(condn_data_am) == 0
    %         condn_data = condn_data_pm;
    %     end

    % getting rid of the mean shift between am and pm sessions
    if length(condn_data_am)>0 &&  length(condn_data_pm)>0
        condn_data=[];
        for ii=1:length(condn_data_am)
            a=condn_data_am{ii};
            b=condn_data_pm{ii};
            dif = mean(b) - mean(a);
            b = b-dif;
            condn_data{ii} = [a;b];
        end
    elseif length(condn_data_am) > 0 && length(condn_data_pm) == 0
        condn_data = condn_data_am;
    elseif length(condn_data_pm) > 0 &&  length(condn_data_am) == 0
        condn_data = condn_data_pm;
    end


    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

    % build the AE based on MLP and only for hG
    [net,Xtrain,Ytrain] = build_mlp_AE(condn_data);
    %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);

    % get activations in deepest layer but averaged over a trial
    imag=1;
    files=[files_am;files_pm];
    [TrialZ_imag,dist_imagined,mean_imagined,var_imagined,idx_imag] = ...
        get_latent_regression_procrustes(condn_data,net,imag);
    dist_imag_total = [dist_imag_total;dist_imagined];
    mean_imag_total=[mean_imag_total;pdist(mean_imagined)];
    var_imag_total=[var_imag_total;var_imagined'];

    %load morning session online
    folders = session_data(i).folders(online_idx_am);
    day_date = session_data(i).Day;
    files_am=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files_am = [files_am;findfiles('',folderpath)'];
    end
    if length(files_am)>0
        condn_data_am = load_data_for_MLP(files_am);
    end

    % load afternoon session online
    folders = session_data(i).folders(online_idx_pm);
    day_date = session_data(i).Day;
    files_pm=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files_pm = [files_pm;findfiles('',folderpath)'];
    end
    if length(files_pm)>0
        condn_data_pm = load_data_for_MLP(files_pm);
    end

    %     % procrutus transformation between am and pm
    %      if length(condn_data_am)>0 &&  length(condn_data_pm)>0
    %         condn_data=[];
    %         for ii=1:length(condn_data_am)
    %             a=condn_data_am{ii};
    %             b=condn_data_pm{ii};
    %             if size(a,1) < size(b,1)
    %                 idx = randperm(size(b,1),size(a,1));
    %                 b1 = b(idx,:);
    %                 idx = randperm(size(a,1),size(a,1));
    %                 a1 = a(idx,:);
    %                 [d,z,transform] = procrustes(a1,b1,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %             elseif size(b,1) < size(a,1)
    %                 idx = randperm(size(a,1),size(b,1));
    %                 a1 = a(idx,:);
    %                 idx = randperm(size(b,1),size(b,1));
    %                 b1 = b(idx,:);
    %                 [d,z,transform] = procrustes(a1,b1,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %             elseif size(b,1) == size(a,1)
    %                 [d,z,transform] = procrustes(a,b,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %                 dif = mean(b) - mean(a);
    %                 b = b-dif;
    %             end
    %             condn_data{ii} = [a;bout];
    %         end
    %     elseif length(condn_data_am) > 0 && length(condn_data_pm) == 0
    %         condn_data = condn_data_am;
    %     elseif length(condn_data_pm) > 0 &&  length(condn_data_am) == 0
    %         condn_data = condn_data_pm;
    %     end

    % getting rid of the mean shift between am and pm sessions
    if length(condn_data_am)>0 &&  length(condn_data_pm)>0
        condn_data=[];
        for ii=1:length(condn_data_am)
            a=condn_data_am{ii};
            b=condn_data_pm{ii};
            dif = mean(b) - mean(a);
            b = b-dif;
            condn_data{ii} = [a;b];
        end
    elseif length(condn_data_am) > 0 && length(condn_data_pm) == 0
        condn_data = condn_data_am;
    elseif length(condn_data_pm) > 0 &&  length(condn_data_am) == 0
        condn_data = condn_data_pm;
    end

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];

    % get activations in deepest layer
    imag=0;
    [TrialZ_online,dist_online,mean_online,var_online,idx_online] = ...
        get_latent_regression_procrustes(condn_data,net,imag);
    dist_online_total = [dist_online_total;dist_online];
    mean_online_total=[mean_online_total;pdist(mean_online)];
    var_online_total=[var_online_total;var_online'];

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

    [h p tb st]=ttest(dist_imagined,dist_online);
    disp([p mean([dist_imagined' dist_online'])]);
    res=[res;[p mean([dist_imagined' dist_online'])]];
end


figure;
hold on
plot(smooth(mean(mahab_full_imagined(:,1:11)),3))
plot(smooth(mean(mahab_full_online(:,1:11)),3))
xlim([1 8])

res

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


% stats on distances, early days and late days online mahalanobis distances
early_days_online = dist_online_total(1:4,:);
early_days_online=early_days_online(:);
late_days_online = dist_online_total(5:end,:);
late_days_online=late_days_online(:);
early_days_online(end+1:length(late_days_online))=NaN;
figure;boxplot([early_days_online late_days_online ]);


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
ylim([5 65]) %0 to 40 for regression
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
ylabel('Mean Diff in latent space')
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





%% getting the proportion of correct bins within a session
% goal here is to see if there is learning i.e. the ratio of correct
% bins increases daily (MAIN IMPORTANT)

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
load session_data
acc_bins_ratio=[];
acc_lower_bound=[];
acc_upper_bound=[];
session_data = session_data([1:9 11]); % removing bad days
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
    acc_bins_ratio(i) = mean(res.data);
    bb=sort(bootstrp(1000,@mean,res.data));
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
iterations=50;
plot_true=false;
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

    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files,0,1);
    % save the data
    %filename = ['condn_data_ImaginedTrials_Day' num2str(i)];
    %save(filename, 'condn_data', '-v7.3')

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = accuracy_imagined_data(condn_data, iterations);
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
        xticklabels ''
        yticklabels ''
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
    acc_online = accuracy_online_data(files);
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
    acc_online_days(:,i) = diag(acc_online);


    %%%%%% cross_val classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_batch = accuracy_online_data(files);
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
    acc_batch_days(:,i) = diag(acc_batch);
end

%save hDOF_10days_accuracy_results_New -v7.3
save hDOF_10days_accuracy_results_New_New -v7.3 % made some corrections on how accuracy is computed
%save hDOF_10days_accuracy_results -v7.3


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

%% (MAIN) saving trial level data for later use with represenational drift

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
iterations=50;
plot_true=false;
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

    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files);
    % save the data
    filename = ['condn_data_ImaginedTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    %%%%% online data %%%%%%%
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files);
    % save the data
    filename = ['condn_data_OnlineTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')


    %%%%%% batch data %%%%%%%
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files);
    % save the data
    filename = ['condn_data_BatchTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

end


%% (MAIN PART 1) CONTINUING PNP REPRESENTANTIONAL STUFF
% goal here is to look at generalizability as consecutive days are added to
% a MLP classifier, all combinations of consecutive days

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_days={};
for days = 1:9 % how many consecutive days
    disp(['processing ' num2str(days) ' day'])
    acc_tmp=[];
    for i=1:10-days+1
        train_days = i:i+days-1;
        I =  ones(10,1);
        I(train_days)=0;
        test_days = find(I==1);

        % load the training data
        train_data=[];
        for j=1:length(train_days)
            imag_filename = ['condn_data_ImaginedTrials_Day' num2str(train_days(j))];
            load(imag_filename)
            train_data=[train_data condn_data];
            online_filename = ['condn_data_OnlineTrials_Day' num2str(train_days(j))];
            load(online_filename)
            train_data=[train_data condn_data];
            batch_filename = ['condn_data_BatchTrials_Day' num2str(train_days(j))];
            load(batch_filename)
            train_data=[train_data condn_data];
        end

        % load the testing
        test_data=[];
        for j=1:length(test_days)
            %             imag_filename = ['condn_data_ImaginedTrials_Day' num2str(test_days(j))];
            %             load(imag_filename)
            %             test_data=[test_data condn_data];
            online_filename = ['condn_data_OnlineTrials_Day' num2str(test_days(j))];
            load(online_filename)
            test_data=[test_data condn_data];
            batch_filename = ['condn_data_BatchTrials_Day' num2str(test_days(j))];
            load(batch_filename)
            test_data=[test_data condn_data];
        end

        % build a classifier and get the accuracy
        acc = accuracy_repdrift_days(train_data,test_data,1);
        acc = squeeze(mean(acc,1));
        acc_tmp = [acc_tmp mean(diag(acc))];
    end
    acc_days{days} = acc_tmp;
end

acc=[];
for i=1:length(acc_days)
    tmp = acc_days{i};
    acc(i) = nanmean(tmp);
end
figure;stem(acc)


%% (MAIN  PART 2) CONTINUING PNP REPRESENTANTIONAL STUFF
% goal here is to look at generalizability as consecutive days are added to
% a MLP classifier, increasing order of conseuctive days

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_days={};
for days = 1:9 % how many consecutive days

    acc_tmp=[];
    train_days = 1:days;
    test_days = days+1:10;
    disp(['processing day ' num2str(train_days)])


    % load the training data
    train_data=[];
    for j=1:length(train_days)
        imag_filename = ['condn_data_ImaginedTrials_Day' num2str(train_days(j))];
        load(imag_filename)
        train_data=[train_data condn_data];
        online_filename = ['condn_data_OnlineTrials_Day' num2str(train_days(j))];
        load(online_filename)
        train_data=[train_data condn_data];
        batch_filename = ['condn_data_BatchTrials_Day' num2str(train_days(j))];
        load(batch_filename)
        train_data=[train_data condn_data];
    end

    % load the testing
    test_data=[];
    for j=1:length(test_days)
        %             imag_filename = ['condn_data_ImaginedTrials_Day' num2str(test_days(j))];
        %             load(imag_filename)
        %             test_data=[test_data condn_data];
        online_filename = ['condn_data_OnlineTrials_Day' num2str(test_days(j))];
        load(online_filename)
        test_data=[test_data condn_data];
        batch_filename = ['condn_data_BatchTrials_Day' num2str(test_days(j))];
        load(batch_filename)
        test_data=[test_data condn_data];
    end

    % build a classifier and get the accuracy
    acc = accuracy_repdrift_days(train_data,test_data,1);
    acc = squeeze(mean(acc,1));
    acc_tmp = [acc_tmp mean(diag(acc))];

    % store results
    acc_days{days} = acc_tmp;
end



acc=[];
for i=1:length(acc_days)
    tmp = acc_days{i};
    acc(i) = nanmean(tmp);
end
figure;plot(acc)

%% (MAIN) LOOKING AT SPATIAL POPS B/W IMAGINED, ONLINE AND BATCH

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

    %%%%%% hG spatial map, ERPs for indicated target direction
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get ERPs and sig. channels
    [hg_erps_imag,sig_ch_imag] = get_erps_imag(files); % first 5 bins is state 1



    %%%%%% get erps from online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    [hg_erps_online,sig_ch_online] = get_erps_online(files); % first 5 bins is state 1
end

% plotting a single channel
ch = 114;
imag_data = hg_erps_imag{5};
imag_data = squeeze(imag_data(ch,:,:));
online_data = hg_erps_online{5};
online_data = squeeze(online_data(ch,:,:));
figure;
hold on
%plot(imag_data,'Color',[.8 .2 .2 .5])
plot(mean(imag_data,2),'r')
%plot(online_data,'Color',[.2 .2 .8 .5])
plot(mean(online_data,2),'b')
vline([5 10],'k')
legend({'Imag','Online'})

tmp = (online_data(11:35,:));
tmp1 = (imag_data(11:35,:));
[var(tmp(:)) var(tmp1(:))]

% plotting comaprisons
ch = 97;
figure;
subplot(2,1,1)
imag_data1 = hg_erps_imag{3};
imag_data1 = squeeze(imag_data1(ch,:,:));
imag_data2 = hg_erps_imag{1};
imag_data2 = squeeze(imag_data2(ch,:,:));
hold on
%plot(imag_data,'Color',[.8 .2 .2 .5])
plot(mean(imag_data1,2),'r')
%plot(online_data,'Color',[.2 .2 .8 .5])
plot(mean(imag_data2,2),'b')
vline([5 10],'k')
legend({'LT','RT'})
title('Imagined')
subplot(2,1,2)
imag_data1 = hg_erps_online{3};
imag_data1 = squeeze(imag_data1(ch,:,:));
imag_data2 = hg_erps_online{1};
imag_data2 = squeeze(imag_data2(ch,:,:));
hold on
%plot(imag_data,'Color',[.8 .2 .2 .5])
plot(mean(imag_data1,2),'r')
%plot(online_data,'Color',[.2 .2 .8 .5])
plot(mean(imag_data2,2),'b')
vline([5 10],'k')
legend({'LT','RT'})
title('Online')


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


% regression lines for accuracy over days
days=1:9;
figure;hold on
plot(days,tmp,'.k','MarkerSize',20)
x = [ones(length(days),1) days'];
y = tmp';
[B,BINT,R,RINT,STATS] = regress(y,x);
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% [bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp',1);
% yhat1 = x*bhat;
% plot(days,yhat1,'k','LineWidth',1)
xlim([.5 9.5])
xticks([1:9])
xticklabels({'1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9'})
%ylim([4 10])
set(gcf,'Color','w')
yticks(4:2:12)
ylim([4 12])


% robust regression lines for accuracy over days
days=1:9;
figure;hold on
plot(days,tmp,'.k','MarkerSize',20)
x = [ones(length(days),1) days'];
y = tmp';
[B,BINT,R,RINT,STATS] = regress(y,x);
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
[bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp',1);
yhat1 = x*bhat;
plot(days,yhat1,'b','LineWidth',1)
xlim([.5 9.5])
xticks([1:9])
xticklabels({'1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9'})
%ylim([4 10])
set(gcf,'Color','w')
yticks(4:2:12)
ylim([4 12])



% regression lines for accuracy over days excluding the last day
days=1:8;
tmp=tmp(1:8);
figure;hold on
plot(days,tmp,'.k','MarkerSize',20)
x = [ones(length(days),1) days'];
y = tmp';
[B,BINT,R,RINT,STATS] = regress(y,x);
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% [bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp',1);
% yhat1 = x*bhat;
% plot(days,yhat1,'k','LineWidth',1)
xlim([.5 8.5])
xticks([1:8])
xticklabels({'1','1-2','1-3','1-4','1-5','1-6','1-7','1-8'})
%ylim([4 10])
set(gcf,'Color','w')
yticks(35:5:60)
ylim([35 60])
title('Accuracy in latent space')
xlabel('Days used to build latent space')
ylabel('Accuracy of day 10 data')






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
v = VideoWriter('PathTrial3.avi');
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



%% SESSION DATA FOR B2


clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
cd(root_path)

% IMAGINED DATA FOLDER IS CENTER OUT AND ONLINE DATA IS IN DISCRETE ARROW

%day1
session_data(1).Day = '20210203';
session_data(1).folders = {'145654','150544','154256'};
session_data(1).folder_type={'I','I','O'};
session_data(1).AM_PM = {'pm','pm','pm'};

%day2
session_data(2).Day = '20210210';
session_data(2).folders = {'143225','151341','151744','154405','155813'};%,'155813'154405
session_data(2).folder_type={'I','O','O','B','B'};%,'B'
session_data(2).AM_PM = {'pm','pm','pm','pm'};

session_data(3).Day = '20210217';
session_data(3).folders = {'142905','144942','145755','150256','150450','150848','151056'};
session_data(3).folder_type={'I','O','O','O','O','B','B'};
session_data(3).AM_PM = {'pm','pm','pm','pm','pm','pm','pm'};

session_data(4).Day = '20210224';
session_data(4).folders = {'143622','161431','163333','163522','163826','164805','164958'};%'165313'
session_data(4).folder_type={'I','I','O','O','O','B','B'};%'B'
session_data(4).AM_PM = {'pm','pm'};

session_data(5).Day = '20210324';
session_data(5).folders = {'140739','142834','143514','144108'};
session_data(5).folder_type={'I','O','O','B'};%'B'
session_data(5).AM_PM = {'pm','pm','pm','pm'};

% session_data(6).Day = '20210324';
% session_data(6).folders = {'144837','152426','153046'};
% session_data(6).folder_type={'I','O','O'};%'B'
% session_data(6).AM_PM = {'pm','pm','pm'};

session_data(6).Day = '20210407';
session_data(6).folders = {'153243','155425'};
session_data(6).folder_type={'I','O'};%'B'
session_data(6).AM_PM = {'pm','pm'};


save session_data_B2 session_data -v7.3

%% SESSION DATA FOR B3


clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
cd(root_path)

% IMAGINED DATA FOLDER IS CENTER OUT AND ONLINE DATA IS IN DISCRETE ARROW

% day 1 %%%% GET THE DATA AGAIN FROM THE PC AS IT HAS NOT BEEN SAVED
k=1;
session_data(k).Day = '20230223';
session_data(k).folders = {'125028','125649','130309','130627','131112',...
    '133039','133358','133843','134055','134525',...
    '135550','135845','140223','140438'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O','O',...
    'B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am','am',...
    'am','am','am','am'};


%day2
k=2;
session_data(k).Day = '20230301';
session_data(k).folders = {'113743','114316','114639','114958','120038','120246',...
    '120505','120825','121458','122238','122443','122641','122858'};
session_data(k).folder_type={'I','I','I','I','O','O','O','O','B','B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am','am'};

%day3
k=3;
session_data(k).Day = '20230302';
session_data(k).folders = {'122334','122931','123406','124341','125002',...
    '125915','130405','130751','131139','131614',...
    '132424','132824','133236','133742'};
session_data(k).folder_type={'I','I','I','I','I','O','O','O','O','O',...
    'B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am','am','am'};

%day4
k=4;
session_data(k).Day = '20230308';
session_data(k).folders = {'114109','114632','114940','115300','115621',...
    '120914','121201','121443','121702','121926',...
    '122749','123008','123237','123447','123846'};
session_data(k).folder_type={'I','I','I','I','I','O','O','O','O','O',...
    'B','B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am','am','am','am'};

%day5
k=5;
session_data(k).Day = '20230309';
session_data(k).folders = {'135628','140326','140904','141504','142051',...
    '143001','143435','143906','144336','144737',...
    '145814','150749'};
session_data(k).folder_type={'I','I','I','I','I','O','O','O','O','O',...
    'B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am'};


%day6
k=6;
session_data(k).Day = '20230315';
session_data(k).folders = {'114014','114328','114637','114946',...
    '115850','120057','120307','120613',...
    '121138','121335','121544'};
session_data(k).folder_type={'I','I','I','I',...
    'O','O','O','O'...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am',...
    'am','am','am','am',...
    'am','am','am'};

%day7
k=7;
session_data(k).Day = '20230316';
session_data(k).folders = {'115713','120250','120610','121013',...
    '121753','122215','122449',...
    '123254','123706','123924'};
session_data(k).folder_type={'I','I','I','I',...
    'O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am',...
    'am','am','am',...
    'am','am','am'};

%day8
k=8;
session_data(k).Day = '20230322';
session_data(k).folders = {'120714','121243','121611','121931',...
    '122703','123008','123225',...
    '123738','124014','124236'};
session_data(k).folder_type={'I','I','I','I',...
    'O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am',...
    'am','am','am',...
    'am','am','am'};

%day9
k=9;
session_data(k).Day = '20230323';
session_data(k).folders = {'121940','122551','123155','123616','124027',...
    '124808','125147','130225','130821',...
    '131616','132215'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am'};


%day10
k=10;
session_data(k).Day = '20230329';
session_data(k).folders = {'113849','114315','114721','115156','115658',...
    '120445','120854','121324','121821',...
    '122522','123025'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am'};


%day11
k=11;
session_data(k).Day = '20230330';
session_data(k).folders = {'122117','123029','124354','125220','125810',...
    '130524','131124','131630','132104',...
    '132647','133231'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am'};


%day12 also covert mime 20231101
k=12;
session_data(k).Day = '20231101';
session_data(k).folders = {'150146','150711','151304','151907','152512',...
    '153413','153959','154552','155027',...
    '155816','160355','160855'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am','am'};

%day13 also covert mime
k=13;
session_data(k).Day = '20231103';
session_data(k).folders = {'144108','144832','145350','145951','151106',...
    '152228','152729','153240','153723',...
    '154260','154658','155156'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am','am'};


%day14 also covert mime
k=14;
session_data(k).Day = '20231106';
session_data(k).folders = {'144344','145053','145532','150023','150448','151002',...
    '151624','152019','152328','152655',...
    '153433','153749','154154'};
session_data(k).folder_type={'I','I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am',...
    'am','am','am','am',...
    'am','am','am'};

%day15 also covert mime
k=15;
session_data(k).Day = '20231108';
session_data(k).folders = {'144602','145219','145634','150123','150607','151039',...
    '151604','151958','152305','152758',...
    '153316','153542','153815'};
session_data(k).folder_type={'I','I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am'...
    'am','am','am','am',...
    'am','am','am'};

%day16 also covert mime
k=16;
session_data(k).Day = '20231110';
session_data(k).folders = {'151210','152025','152352','152755','153205',...
    '153829','154338','154618','154906',...
    '155654','155935','160204'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am','am'};

%day17 also covert mime
k=17;
session_data(k).Day = '20231113';
session_data(k).folders = {'143300','143901','144235','144645','145013',...
    '145733','150011','150507','150818',...
    '151337','151628','151931'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am','am'};


%day18 also covert mime
k=18;
session_data(k).Day = '20231115';
session_data(k).folders = {'144845','145418','145745','150314','150646',...
    '151230','151519','151819','152110',...
    '152707','152950','153216'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am','am'};


%day19 also covert mime
k=19;
session_data(k).Day = '20231117';
session_data(k).folders = {'151638','152330','152731','153105','153525',...
    '154040','154341','154645','155001',...
    '155407','155657','160025'};
session_data(k).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am',...
    'am','am','am','am',...
    'am','am','am'};

%day20 PNP CONTROL STARTS -> error ignore, but use data
k=20;
session_data(k).Day = '20231120';
session_data(k).folders = {'143242','143825','144059','152530'};
session_data(k).folder_type={'B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am'};


%day21 PNP CONTROL STARTS
k=21;
session_data(k).Day = '20231122';
session_data(k).folders = {'144133','144831','145050','152546',...
    '152908','153225','153525'};
session_data(k).folder_type={'B','B','B','B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am','am'};

%day22 PNP CONTROL Day 2
k=22;
session_data(k).Day = '20231127';
session_data(k).folders = {'142420','142957','143245','152030',...
    '152415','152749'};
session_data(k).folder_type={'B','B','B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am'};

%day23 PNP CONTROL Day 3
k=23;
session_data(k).Day = '20231129';
session_data(k).folders = {'143141','143636','143936'};
session_data(k).folder_type={'B','B','B'};
session_data(k).AM_PM = {'am','am','am'};

%day24 PNP CONTROL Day 4
k=24;
session_data(k).Day = '20231201';
session_data(k).folders = {'142050','142505','142641','142920'};
session_data(k).folder_type={'B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am'};

%day25 PNP CONTROL Day 5
k=25;
session_data(k).Day = '20231207';
session_data(k).folders = {'154441'};
session_data(k).folder_type={'B'};
session_data(k).AM_PM = {'am'};

%day26 PNP CONTROL Day 6
k=26;
session_data(k).Day = '20231210';
session_data(k).folders = {'153818','154259','154654','154930','155159'};
session_data(k).folder_type={'B','B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am'};

%day27 PNP CONTROL Day 7
k=27;
session_data(k).Day = '20231213';
session_data(k).folders = {'144936','145457','145726'};
session_data(k).folder_type={'B','B','B'};
session_data(k).AM_PM = {'am','am','am'};

%day28 PNP CONTROL Day 8
k=28;
session_data(k).Day = '20231215';
session_data(k).folders = {'143903','144741','145028'};
session_data(k).folder_type={'B','B','B'};
session_data(k).AM_PM = {'am','am','am'};

%day29 PNP CONTROL Day 9
k=29;
session_data(k).Day = '20231220';
session_data(k).folders = {'143326','143844','144121'};
session_data(k).folder_type={'B','B','B'};
session_data(k).AM_PM = {'am','am','am'};

%day30 PNP CONTROL Day 10
k=30;
session_data(k).Day = '20231228';
session_data(k).folders = {'132651','133809','134328','134839','135205'};
session_data(k).folder_type={'B','B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am'};

%day31 PNP CONTROL Day 11
k=31;
session_data(k).Day = '20231229';
session_data(k).folders = {'132822','133400','133737','134343','134836'};
session_data(k).folder_type={'B','B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am'};

%day32 PNP CONTROL Day 12
k=32;
session_data(k).Day = '20240104';
session_data(k).folders = {'133046','133554','135401','135854','140117'};
session_data(k).folder_type={'B','B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am'};



save session_data_B3 session_data -v7.3

%% PLASTICITY AND AE FRAMEWORK FOR B2 ARROW DATA (MAIN)
% good days: 20210324

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data_B2
dist_online_total=[];
dist_imag_total=[];
var_imag_total=[];
mean_imag_total=[];
var_online_total=[];
mean_online_total=[];
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

    %%%%%%imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'CenterOut',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596-002131.mat')
    condn_data = load_data_for_MLP_B2(files,ecog_grid);

    %save the data
    filename = ['B2_condn_data_Imagined_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % build the AE based on MLP and only for hG
    %[net,Xtrain,Ytrain] = build_mlp_AE_B2(condn_data);
    %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);

    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

    % get activations in deepest layer but averaged over a trial
    %     imag=1;
    %     [TrialZ_imag,dist_imagined,mean_imagined,var_imagined,idx_imag] = ...
    %         get_latent_regression_B2(files,net,imag,ecog_grid);
    %     dist_imag_total = [dist_imag_total;dist_imagined];
    %     mean_imag_total=[mean_imag_total;pdist(mean_imagined)];
    %     var_imag_total=[var_imag_total;var_imagined'];

    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'DiscreteArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP_B2(files,ecog_grid);

    % save the data
    filename = ['B2_condn_data_Online_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];


    % get activations in deepest layer
    %     imag=0;
    %     [TrialZ_online,dist_online,mean_online,var_online,idx_online] = ...
    %         get_latent_regression_B2(files,net,imag,ecog_grid);
    %     dist_online_total = [dist_online_total;dist_online];
    %     mean_online_total=[mean_online_total;pdist(mean_online)];
    %     var_online_total=[var_online_total;var_online'];

    %%%%%%batch data
    if ~isempty(batch_idx)
        folders = session_data(i).folders(batch_idx);
        day_date = session_data(i).Day;
        files=[];
        for ii=1:length(folders)
            folderpath = fullfile(root_path, day_date,'DiscreteArrow',folders{ii},'BCI_Fixed');
            files = [files;findfiles('',folderpath)'];
        end


        %load the data
        condn_data = load_data_for_MLP_B2(files,ecog_grid);

        % save the data
        filename = ['B2_condn_data_Batch_Day' num2str(i)];
        save(filename, 'condn_data', '-v7.3')

        % get the mahab distance in the full dataset
        Donline = mahal2_full(condn_data);
        Donline = triu(Donline);
        Donline = Donline(Donline>0);
        mahab_full_batch = [mahab_full_batch Donline];
    else
        mahab_full_batch = [mahab_full_batch zeros(6,1)];
    end

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

save mahab_dist_full_B2 -v7.3

figure;boxplot([mahab_full_imagined mahab_full_online mahab_full_batch])

figure;plot(1:6,mean(mahab_full_imagined));
hold on
plot(1:6,mean(mahab_full_online))
plot(2:5,mean(mahab_full_batch(:,2:5)))

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


% stats on distances, early days and late days online mahalanobis distances
early_days_online = dist_online_total(1:4,:);
early_days_online=early_days_online(:);
late_days_online = dist_online_total(5:end,:);
late_days_online=late_days_online(:);
early_days_online(end+1:length(late_days_online))=NaN;
figure;boxplot([early_days_online late_days_online ]);


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
ylim([5 65]) %0 to 40 for regression
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
ylabel('Mean Diff in latent space')
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


a=randn(3)
idx = randi(3,[3,3])
a(idx)



% save res_python_B2_latent -v7.3

%%%% regressions for mahab on full as a function of day
clear tmp
tmp(:,1) = (median(mahab_full_imagined,1));
tmp(:,2) = (median(mahab_full_online,1));
tmp(:,3) = (median(mahab_full_batch,1));
tmp = tmp(2:5,:);
% for i=1:size(tmp,2)
%     tmp(:,i) = smooth(tmp(:,i));
% end

% plotting the regression for Mahab distance increases as a function of day
figure;
xlim([0 5])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:4,tmp(:,1),'.b','MarkerSize',20)
y = tmp(:,1);
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
plot(1:4,yhat,'b','LineWidth',1)
% online
plot(1:4,tmp(:,2),'.k','MarkerSize',20)
y = tmp(:,2);
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat = x*B;
plot(1:4,yhat,'k','LineWidth',1)
% batch
plot(1:4,tmp(:,3),'.r','MarkerSize',20)
y = tmp(:,3);
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat = x*B;
plot(1:4,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xlim([0.5  4.5])
xticks([1:4])
ylim([0 10])
%yticks([5:5:35])
%ylim([5 35])



% plotting the robust regression for Mahab distance increases as a function of day
figure;
xlim([0 5])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:4,tmp(:,1),'.b','MarkerSize',20)
[bhat p1 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,1),2);
yhat = x*bhat;
plot(1:4,yhat,'b','LineWidth',1)
% online
plot(1:4,tmp(:,2),'.k','MarkerSize',20)
[bhat p2 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,2),2);
yhat = x*bhat;
plot(1:4,yhat,'k','LineWidth',1)
% batch
plot(1:4,tmp(:,3),'.r','MarkerSize',20)
[bhat p3 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,3),2);
yhat = x*bhat;
plot(1:4,yhat,'r','LineWidth',1)


% bootstrapped test for regression coefficient
y = tmp(:,3);
[B2,BINT,R,RINT,STATS2] = regress(y,x);
bhat_boot=[];
for i=1:2000
    ytmp = y(randperm(numel(y)));
    [B_tmp,~,Rtmp,~,STATS_tmp] = regress(ytmp,x);
    bhat_boot(i) = B_tmp(2);
end
figure;hist(bhat_boot)
vline(B2(2),'r')
sum(bhat_boot>B2(2))/length(bhat_boot)

% mixed effect model for batch and Init seed slopes
day_name=[];
mahab_dist=[];
for i=1:size(tmp,1)
    day_name = [day_name;i;i];
    mahab_dist = [mahab_dist;tmp(i,2:3)'];
end
data = table(day_name,mahab_dist);
glm = fitglme(data,'mahab_dist ~ 1 + day_name');
stat = glm.Coefficients.tStat(2);
stat_boot=[];
for i=1:1000
    day_name_tmp = day_name(randperm(numel(day_name)));
    data_tmp = table(day_name_tmp,mahab_dist);
    glm_tmp = fitglm(data_tmp,'mahab_dist ~ 1 + day_name_tmp');
    stat_boot(i) = glm_tmp.Coefficients.tStat(2);
end
figure;hist(stat_boot)
vline(stat)
sum(stat_boot>stat)/length(stat_boot)

% getting the mixed effect model stats for latent variance in  B2
%save latent_variance_B2 -v7.3
% fitting the data into a RM anova format
tmp=[tmp];
Data_RM_B2_latentVar=[];
for i=1:length(tmp)
    exp_type = [1 2 3]';
    subject = [i i i]';
    lat_var = tmp(i,:)';
    Data_RM_B2_latentVar = [Data_RM_B2_latentVar;[exp_type subject lat_dist]];
end
save Data_RM_B2_latentVar Data_RM_B2_latentVar -v7.3

% putting into a mixed effct model format
day_name=[];
latent_var=[];
exp_type=[];
subject=[];
I = 1:4:length(tmp);
for i=1:length(I)
    tmp1 = tmp(I(i):I(i)+3,:);
    for j=1:length(tmp1)
        exp_type = [exp_type;[1 2 3]'];
        latent_var = [latent_var;tmp1(j,:)'];
        day_name = [day_name;[i i i]'];
        subject = [subject;[1 1 1]'];
    end
end
data=table(exp_type,latent_var,day_name,subject);
glme = fitglme(data,'latent_var ~ 1 + exp_type +(1|day_name)+(1|subject)')

% doing paired t-tests
I = sort([find(data.exp_type==2);find(data.exp_type==3)]);
data1=data(I,:);
glme = fitglme(data1,'latent_var ~ 1 + exp_type +(1|day_name)+(1|subject)')

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
iterations=5;
plot_true=false;
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

    %load the data
    load('ECOG_Grid_8596-002131.mat')
    condn_data = load_data_for_MLP_TrialLevel_B2(files,ecog_grid);

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = accuracy_imagined_data_B2(condn_data, iterations);
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
        folderpath = fullfile(root_path, day_date,'DiscreteArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_online = accuracy_online_data_B2(files);
    if plot_true
        figure;imagesc(acc_online)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_online_days(:,i) = diag(acc_online);


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

        % get the classification accuracy
        acc_batch = accuracy_online_data_B2(files);
        if plot_true
            figure;imagesc(acc_batch)
            colormap bone
            clim([0 1])
            set(gcf,'color','w')
        end
        acc_batch_days(:,i) = diag(acc_batch);
    else
        acc_batch_days(:,i) = NaN*ones(4,1);
    end
end

load ('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\b1_acc_rel_imagined_prop.mat')
save hDOF_6days_accuracy_results_New_B2 -v7.3
%save hDOF_10days_accuracy_results -v7.3


%acc_online_days = (acc_online_days + acc_batch_days)/2;
figure;
ylim([0.0 0.65])
xlim([0.5 6.5])
hold on
plot(nanmean(acc_imagined_days,1))
plot(nanmean(acc_online_days,1))
plot(nanmean(acc_batch_days,1),'k')

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
m3b = sort(bootstrp(1000,@nanmean,m3));
m33 = nanmean(acc_batch_days,1);
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
    plot(x(i),y(i),'o','MarkerSize',20,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
for i=1:size(acc_batch_days,2)
    plot(1+0.1*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
    plot(2+0.1*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
    plot(3+0.1*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
end
xlim([.5 3.5])
ylim([0.10 0.55])
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
yticks(0:.1:1)
set(gca,'FontSize',12)
h=hline(.25,'--');
h.LineWidth=1;
xlabel('Decoder Type')
ylabel('Accuracy')

tmp = [ m11' m22' m33'];
figure;boxplot(tmp)

tmp = [ m1 m2 m3];
figure;boxplot(tmp)

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

acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=1;
plot_true=false;
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

    %load the data
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0);

    % get cross-val classification accuracy
    [acc_imagined,train_permutations,acc_imagined_bin] =...
        accuracy_imagined_data(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined_bin,1));
    %disp(mean(diag(acc_imagined)))
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
    [acc_online,acc_online_bin] = accuracy_online_data(files);
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
    acc_online_days(:,i) = diag(acc_online_bin);


    %%%%%% cross_val classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    [acc_batch,acc_batch_bin] = accuracy_online_data(files);
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
    acc_batch_days(:,i) = diag(acc_batch_bin);

end

%load ('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\b1_acc_rel_imagined_prop.mat')
%save hDOF_6days_accuracy_results_New_B2 -v7.3
%save hDOF_11days_accuracy_results_B3 -v7.3
%save hDOF_11days_accuracy_results_B3_corrected -v7.3 % not good
%save hDOF_11days_accuracy_results_B3_v2 -v7.3 % new after old data got deleted: best of the lot 
%save hDOF_11days_accuracy_results_B3_v3 -v7.3 % new after old data got deleted
%save hDOF_11days_accuracy_results_B3_v4 -v7.3 % new and correcting for errors in accuracy computation

nanmean(acc_imagined_days,1)'
mean(ans)


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

%% (MAIN) COMBINING AND PLOTTING DECODING ACC FOR B1 AND B3 OL -> CL1, CL2

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
a=load('hDOF_11days_accuracy_results_B3_v4');

% load B1 data 
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%b=load('hDOF_10days_accuracy_results_New');
b=load('hDOF_10days_accuracy_results_New_New');

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
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end
aa = find(idx==2);
x=(1:3) + 0.1*randn(length(aa),3);
h=scatter(x,[m11(aa)' m22(aa)' m33(aa)'],'filled');
for i=1:3
    h(i).MarkerFaceColor = 'b';
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


% Signed rank test
[P,H,STATS] = signrank(mean(acc_batch_days,1),mean(acc_online_days,1));
[P,H,STATS] = signrank(mean(acc_imagined_days,1),mean(acc_online_days,1));
[P,H,STATS] = signrank(mean(acc_batch_days,1),mean(acc_imagined_days,1));


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


% run LMM non parametric test
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

% boxplots comparing OL, CL1 and CL2 
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


%% B3 PLASTICITY AND AE FRAMEWORK (MAIN MAIN)
% good days: 20210324

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data_B3
dist_online_total=[];
dist_imag_total=[];
var_imag_total=[];
mean_imag_total=[];
var_online_total=[];
mean_online_total=[];
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
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_B3(files,ecog_grid);



    %save the data
    filename = ['B3_condn_data_Imagined_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % build the AE based on MLP and only for hG
    %[net,Xtrain,Ytrain] = build_mlp_AE_B2(condn_data);
    %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);

    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

    % get activations in deepest layer but averaged over a trial
    %     imag=1;
    %     [TrialZ_imag,dist_imagined,mean_imagined,var_imagined,idx_imag] = ...
    %         get_latent_regression_B2(files,net,imag,ecog_grid);
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
    condn_data = load_data_for_MLP_B3(files,ecog_grid);

    % save the data
    filename = ['B3_condn_data_Online_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];


    % get activations in deepest layer
    %     imag=0;
    %     [TrialZ_online,dist_online,mean_online,var_online,idx_online] = ...
    %         get_latent_regression_B2(files,net,imag,ecog_grid);
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
    condn_data = load_data_for_MLP_B3(files,ecog_grid);

    % save the data
    filename = ['B3_condn_data_Batch_Day' num2str(i)];
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

save mahab_dist_full_B3 -v7.3

figure;boxplot([mahab_full_imagined mahab_full_online mahab_full_batch])


X= [ ones(size(mahab_full_imagined,2),1) [1:size(mahab_full_imagined,2)]'];
figure;plot(1:size(mahab_full_imagined,2),mean(mahab_full_imagined),'.','MarkerSize',20);
hold on
plot(1:size(mahab_full_imagined,2),mean(mahab_full_online),'r')
plot(1:size(mahab_full_imagined,2),mean(mahab_full_batch),'k')

% plotting the regression for Mahab distance increases as a function of day
tmp=[mean(mahab_full_imagined)' mean(mahab_full_online)' mean(mahab_full_batch)'];
figure;
xlim([0 size(tmp,1)+0.5])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:size(tmp,1),tmp(:,1),'.b','MarkerSize',20)
y = tmp(:,1);
[B1,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B1;
plot(1:size(tmp,1),yhat,'b','LineWidth',1)
% online
plot(1:size(tmp,1),tmp(:,2),'.k','MarkerSize',20)
y = tmp(:,2);
[B2,BINT,R,RINT,STATS2] = regress(y,x);
yhat = x*B2;
plot(1:size(tmp,1),yhat,'k','LineWidth',1)
% batch
plot(1:size(tmp,1),tmp(:,3),'.r','MarkerSize',20)
y = tmp(:,3);
[B3,BINT,R,RINT,STATS3] = regress(y,x);
yhat = x*B3;
plot(1:size(tmp,1),yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:11])
xlabel('Days')
ylabel('Mahab Distance')
title('B3 original high dimensional data')
legend({'','OL','','CL1','','CL2'})



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



%% (MAIN) Looking at the regression of projecting across day data on manifold

% get the stats from python
clc;clear

%B2
y=[0.06356695, 0.07478359, 0.09559412, 0.12340624, 0.13129963]';
x=[ones(length(y),1) [1:length(y)]'];

[B,BINT,R,RINT,STATS1] = regress(y,x);
lm = fitlm(x(:,2),y);

%B1
y=[5.95474288, 6.8286061 , 8.39052186, 8.17585576, 8.27175004,...
    9.24597975, 8.60957232, 9.03721714, 8.79579779]';
x=[ones(length(y),1) [1:length(y)]'];

[B,BINT,R,RINT,STATS2] = regress(y,x);
STATS2
lm = fitlm(x(:,2),y)

% for lm get the data from python
mahab=data(:,1);
days = data(:,2);
data=table(days,mahab);

glm = fitglme(data,'mahab ~ 1+ days');

save latent_across_day_proj_B1 -v7.3


%% ANALYSIS OF THE 3D PATH TASK AND ITS PERFORMANCE ETC.

% get all the trials that had the 3D path task with some measure of task
% performance

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'

task_name = 'Robot3DPath';
folders = dir(root_path);

field1 = 'name'; value1 = '';
field2 = 'folder'; value2 = '';
field3 = 'date'; value3='';
field4 = 'bytes'; value4 = [];
field5 = 'isdir';value5 =[];
field6 = 'datenum';value6=[];

folders1=struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6);
k=1;
for i=1:length(folders)
    if ~isempty(str2num(folders(i).name))
        folders1(k) = folders(i);
        k=k+1;
    end
end

folders=folders1;

files=[];
for i=1:length(folders)
    folder_path = fullfile(root_path,folders(i).name,task_name);
    if exist(folder_path)
        tmp_files=findfiles('.mat',folder_path,1)';
        for j=1:length(tmp_files)
            if length(regexp(tmp_files{j},'Data'))>0
                files=[files;tmp_files(j)];
            end
        end
    end
end

figure;
plot3(TrialData.CursorState(1,:),TrialData.CursorState(2,:),TrialData.CursorState(3,:),'.')

%% plotting robot trajectories for center out (MAIN)

clc;clear
foldername = '20210326';
task_name = 'Robot';
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
addpath(genpath('C:\Users\nikic\Documents\MATLAB\svg_plot'))

fullpath = fullfile(root_path,foldername,task_name);
files = findfiles('.mat',fullpath,1);
files1=[];
for i=1:length(files)
    if length(regexp(files{i},'Data'))>0
        files1=[files1;files(i)'];
    end
end
files=files1;

% get all the good robot 3D trial data and plot them
col = {'r','g','b','c','m','y'};
col=turbo(6);
files_suc=[];
figure;
hold on
xlim([-250,250])
ylim([-250,250])
zlim([-250,250])
recon_error = [];% wrt to how well straight line trajecotry is reconstructed
trial_error = [];% just the plain error with respect to target location
for i=1:length(files)
    load(files{i})
    if TrialData.TargetID == TrialData.SelectedTargetID
        kin = TrialData.CursorState;
        task_state = TrialData.TaskState;
        kinidx = find(task_state==3);
        kin = kin(:,kinidx);
        target = TrialData.TargetPosition;
        targetID = TrialData.TargetID;
        fs = TrialData.Params.UpdateRate;
        if size(kin,2)*(1/fs) < 12
            files_suc = [files_suc;files(i)];
            %plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col{targetID});
            plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col(targetID,:));
        end

        % get the errors in terms of deviation from the ideal path
        idx = find(target==0); % get the axes where errors shoudln't happen
        idx_target = find(target~=0);
        tmp_error = [];
        kin = kin(1:3,:);
        for j=1:size(kin,2)
            if sum(kin(:,j)) ~= 0
                break
            end
        end
        kin = kin(:,j:end);
        for j=1:size(kin,2)
            if (sign(target(idx_target)) * sign(kin(idx_target,j))) == -1
                f=2;
            else
                f=1;
            end
            e = f*(sum((target(idx)' - kin(idx,j)).^2));
            tmp_error = [tmp_error;e];
        end
        recon_error =[recon_error; sqrt(sum(tmp_error(1:14)))];
        trial_error = [trial_error sqrt(sum(sum((kin(:,1:10) - target').^2)))];
        %end
    end
end

set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('X-axis')
ylabel('Y-axis')
zlabel('Z-axis')
set(gca,'LineWidth',1.0)
grid on

save recon_error_IBID recon_error -v7.3

%plot2svg('3DTraj-grid.svg');


%% plotting diagonal robot trajectories (MAIN)



clc;clear
foldername = '20210115';
task_name = 'Robot';
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
addpath(genpath('C:\Users\nikic\Documents\MATLAB\svg_plot'))

files={'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114418\BCI_Fixed\Data0002.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115/Robot\114613\BCI_Fixed\Data0001.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114613\BCI_Fixed\Data0002.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114613\BCI_Fixed\Data0004.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114740\BCI_Fixed\Data0001.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0001.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0002.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0003.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0005.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0006.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0007.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0008.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0009.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0010.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0011.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0012.mat'};

% get all the good robot 3D trial data and plot them
col = {'r','g','b','c','m','y'};
col=turbo(4);
files_suc=[];
figure;
hold on
xlim([-250,250])
ylim([-250,250])
zlim([-250,250])
tid=[];
% also get the velocities in the x and y directions
vel=[];
err_vel=[];
for i=1:length(files)
    load(files{i})
    tid = [tid TrialData.TargetID];
    if TrialData.TargetID == TrialData.SelectedTargetID
        kin = TrialData.CursorState;
        task_state = TrialData.TaskState;
        kinidx = find(task_state==3);
        kin = kin(:,kinidx);
        target = TrialData.TargetPosition;
        targetID = TrialData.TargetID-6;
        fs = TrialData.Params.UpdateRate;
        if size(kin,2)*(1/fs) < 12
            files_suc = [files_suc;files(i)];
            %plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col{targetID});
            plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col(targetID,:));
            vel = [vel kin(4:6,:)];
        end
        % get the velocities relative to the ideal velocity towards the
        % target
        pos = TrialData.TargetPosition(1:2)'
        start_pos = kin(1:2,1);
        ideal_vector  = pos-start_pos;
        ideal_vector = ideal_vector./norm(ideal_vector);
        tmp_vel = kin(4:6,1:end);
        idx = abs(sum(tmp_vel))>0;
        tmp_vel = tmp_vel(1:2,idx);
        for j=1:length(tmp_vel)
            tmp_vel(:,j)=tmp_vel(:,j)./norm(tmp_vel(:,j));
        end
        %%% cos angle
        %angles_err = acos(ideal_vector'*tmp_vel);
        %err_vel =[err_vel angles_err];
        %%% angle to target
        %ideal_angle = atan2(ideal_vector(2)/ideal_vector(1));
        %angles_err = atan2(tmp_vel(2,:)./tmp_vel(1,:));
        ideal_angle = atan2(ideal_vector(1),ideal_vector(2));
        angles_err = atan2(tmp_vel(1,:),tmp_vel(2,:));
        angles_err_rel = angles_err - ideal_angle;
        err_vel =[err_vel angles_err_rel];
    end
end

% histogram of the errors in decoded velocities with the ideal velocity
figure;rose(err_vel)
figure;hist(err_vel*180/pi,20)
vline(45)

% circular statistics test
addpath(genpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a'))
mu = circ_mean(err_vel') % get the mean
[pval, z] = circ_rtest(err_vel); % is it uniformly distributed
[h mu ul ll]  = circ_mtest(err_vel', 0) % does it have a specific mean
[ll mu ul]*180/pi
pval = circ_medtest(err_vel',0)


%%%%%%% USING CHAT GPT %%%%

% Set of angles in radians
angles = err_vel';

% Reference direction (null hypothesis)
mu_0 = 0; % You can set this to your desired reference direction

% Compute the circular mean
mu = circ_mean(angles);

% Perform the one-sample test and get the p-value
p_value = circ_test(angles - mu_0);

% Display the results
fprintf('Circular mean: %f\n', mu);
fprintf('P-value: %f\n', p_value);

% Compare with significance level
alpha = 0.05; % Set your desired significance level
if p_value < alpha
    fprintf('Reject the null hypothesis: The mean direction is significantly different.\n');
else
    fprintf('Fail to reject the null hypothesis: The mean direction is not significantly different.\n');
end


%%%%%%%%%%%



%grid off
set(gcf,'Color','w')
set(gca,'FontSize',12)
plot2svg('3DTraj_diag1.svg');

% pca on the velocity data
[c,s,l] = pca(vel');
figure;
stem(cumsum(l)./sum(l))
figure;stem(c(:,1))
xlim([0 4])
figure;stem(c(:,2))
xlim([0 4])
figure;stem(c(:,3))
xlim([0 4])
figure;



% pca on null velocity data, 2dim
vel_null=[];
for i=1:100
    if rand(1)>0.5
        tmp = [randn(1,10)*20;zeros(1,10)];
    else
        tmp = [zeros(1,10);randn(1,10)*20];
    end
    vel_null = [vel_null tmp];
end
[c,s,l] = pca(vel_null');
figure;stem(c(:,1),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
figure;stem(c(:,2),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off

% pca on the first x and y velocity alone
C = cov((vel(1:2,:))');
figure;imagesc(C)
[c,s,l] = pca(vel(1:2,:)');
figure;stem(c(:,1),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
figure;stem(c(:,2),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
% nullhypothesis via MP test
kindata1=vel(1:2,:)';
kindata1=kindata1-mean(kindata1);
q = size(kindata1,1)/size(kindata1,2);
q = sqrt(q);
sigma2 = var(kindata1(:)-mean(kindata1(:)));
lambda_thresh = sigma2*((1+1/q)^2);
figure;stem(l);
hline(lambda_thresh)




% get histogram of angles relative to the x and y axes
velxy = vel(1:2,:);
angles = [];
for i=1:size(velxy,2)
    velxy(:,i) = velxy(:,i)./norm(velxy(:,i));
    tmp = (velxy(:,i));
    angles(i) = atan(tmp(2)/tmp(1));
end
errx = acos([1 0]*abs(velxy));
erry = acos([0 1]*abs(velxy));
err = [errx erry];
err = err(~isnan(err));
angles = angles(~isnan(angles));

% null distribution to compare these angles towards:
% Fit an exponential distribution centered at 0 and pi/2 with variance
% equal to the actual data
mu = std(angles);
tmp=0:0.01:pi/2;
y = exppdf(tmp,mu);
%y = conv(y,fliplr(y),'same');
y=y+fliplr(y);
y=y./sum(y);
figure;plot(tmp,y)
figure;hist(angles)



%plot
[t,r]=rose(err_vel,20);
figure
polarplot(t,r,'LineWidth',1,'Color','k');
pax=gca;
%pax.RLim = [0 20];
thetaticks(0:30:360);
pax.ThetaAxisUnits='radians';
pax.FontSize=16;
set(gcf,'Color','w')
%pax.RTick = [5 10 15 20 ];
pax.GridAlpha = 0.25;
pax.MinorGridAlpha = 0.25;
pax.ThetaMinorGrid = 'off';
%pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
pax.ThetaTickLabel = ''
%pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
pax.RTickLabel = {' ',' '};
pax.RAxisLocation=1;
pax.RAxis.LineWidth=1;
pax.ThetaAxis.LineWidth=1;
pax.LineWidth=1;
%pax.ThetaLim = [0 pi/2];
temp = exp(1i*err_vel);
r1 = abs(mean(temp))*1 * max(r);
phi = angle(mean(temp));
hold on;
polarplot([phi-0.01 phi],[0 r1],'LineWidth',1.5,'Color','r')
%polarplot([0.7854-0.01 0.7854],[0 r1],'LineWidth',1.5,'Color','m')
%polarplot([0 0],[0 0.25e3],'LineWidth',1.5,'Color','k')
%polarplot([pi/2 pi/2 ],[0 0.25e3],'LineWidth',1.5,'Color','k')
set(gcf,'PaperPositionMode','auto')
set(gcf,'Position',[680.0,865,120.0,113.0])


% null model of what distributions should be like
err_null = [zeros(1,400) 90*pi/180*ones(1,400)];
[t,r]=rose(err_null,50);
figure
polarplot(t,r,'LineWidth',1,'Color','k');
pax=gca;
%pax.RLim = [0 20];
thetaticks(0:30:360);
pax.ThetaAxisUnits='radians';
pax.FontSize=16;
set(gcf,'Color','w')
%pax.RTick = [5 10 15 20 ];
pax.GridAlpha = 0.25;
pax.MinorGridAlpha = 0.25;
pax.ThetaMinorGrid = 'off';
%pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
pax.ThetaTickLabel = ''
%pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
pax.RTickLabel = {' ',' '};
pax.RAxisLocation=1;
pax.RAxis.LineWidth=1;
pax.ThetaAxis.LineWidth=1;
pax.LineWidth=1;
pax.ThetaLim = [0 90*pi/180];
hold on;
polarplot([0 0],[0 0.4e3],'LineWidth',1.5,'Color','k')
polarplot([pi/2 pi/2 ],[0 0.4e3],'LineWidth',1.5,'Color','k')
set(gcf,'PaperPositionMode','auto')
set(gcf,'Position',[680.0,865,120.0,113.0])


% now plotting a few example trials along with position, user input and
% velocity profile

idx=1; %1 and 7
load(files{idx})
kin = TrialData.CursorState;
task_state = TrialData.TaskState;
kinidx = find(task_state==3);
kin = kin(:,kinidx);
decodes = TrialData.FilteredClickerState;
fs=TrialData.Params.UpdateRate;
tt = [0:length(decodes)-1]*(1/fs);
col = turbo(8);

% plot the trajectory
figure
hold on
for i=1:length(decodes)
    c = col(decodes(i)+1,:);
    plot3(kin(1,i),-kin(2,i),kin(3,i),'.','MarkerSize',40,'Color',c);
end
xlim([-300 300])
ylim([-300 300])
zlim([-300 300])
target = TrialData.TargetPosition;
plot3(target(1),-target(2),target(3),'ok','MarkerSize',50)
%plot(-150,150,'o','MarkerSize',50,'Color','k')
set(gcf,'Color','w')
xticks([-200:200:200])
yticks([-200:200:200])
zticks([-200:200:200])

% plot as a straight line
figure;
hold on
plot3(kin(1,:),-kin(2,:),kin(3,:),'LineWidth',2,'Color','b');
xlim([-300 300])
ylim([-300 300])
zlim([-300 300])
target = TrialData.TargetPosition;
plot3(target(1),-target(2),target(3),'ok','MarkerSize',50)
%plot(-150,150,'o','MarkerSize',50,'Color','k')
set(gcf,'Color','w')
xticks([-200:200:200])
yticks([-200:200:200])
zticks([-200:200:200])
xlabel('X axis')
ylabel('Y axis')
zlabel('Z axis')
set(gcf,'Color','w')
set(gca,'FontSize',14)
%view(40,40)



% plot the decodes
figure;
set(gcf,'Color','w')
subplot(2,1,1)
hold on
for i=0:7
    h=barh(i,length(decodes),1);
    h.FaceColor = col(i+1,:);
    %h.FaceAlpha = 0.8;
    h.FaceAlpha = 1;
end
stem(decodes,'filled','LineWidth',1,'Color','k')
ii = [9 17 25 33 41 49];
xticks(ii)
xticklabels(tt(ii))
yticks ''
axis tight
% plot the velocity profile
subplot(2,1,2)
hold on
plot(kin(4,:),'r','LineWidth',1)
plot(kin(5,:),'b','LineWidth',1)
plot(kin(6,:),'g','LineWidth',1)
xticks(ii)
xticklabels(tt(ii))
axis tight

%%%%% plotting the dynamics and decodes for the virtual r2g lateral task
filename = fullfile(root_path,'20211013\RobotLateralR2G\135901\BCI_Fixed\Data0001.mat');
load(filename)
grip_mode = TrialData.OpMode;
decodes = TrialData.FilteredClickerState;
translation_decodes = decodes(grip_mode==0);
gripper_decodes = decodes(grip_mode==1);

% plot the decodes
fs=TrialData.Params.UpdateRate;
tt = [0:length(decodes)-1]*(1/fs);
col = turbo(8);
figure;
set(gcf,'Color','w')
hold on
for i=0:7
    h=barh(i,length(decodes),1);
    h.FaceColor = col(i+1,:);
    h.FaceAlpha = 0.5;
end
stem(decodes,'filled','LineWidth',1,'Color','k')
ii = [26 51 76 101 126 151 176 201 226 ];
xticks(ii)
xticklabels(tt(ii))
yticks ''
axis tight

%% (MAIN) B1 ROBOT PATH TASK STATS
% get the path efficiency i.e., distance from ideal path as compared to
% random walks of same length

clc;clear;close all




%% (MAIN) B3 looking at decoding performance from imagined -> online -> batch
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
iterations=50;
plot_true=false;
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

    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files);
    % save the data
    filename = ['condn_data_ImaginedTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = accuracy_imagined_data(condn_data, iterations);
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
    acc_online = accuracy_online_data(files);
    if plot_true
        figure;imagesc(acc_online)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_online_days(:,i) = diag(acc_online);


    %%%%%% cross_val classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_batch = accuracy_online_data(files);
    if plot_true
        figure;imagesc(acc_batch)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_batch_days(:,i) = diag(acc_batch);
end

save hDOF_10days_accuracy_results_New -v7.3
%save hDOF_10days_accuracy_results -v7.3


%acc_online_days = (acc_online_days + acc_batch_days)/2;
figure;
ylim([0.2 1])
xlim([0.5 10.5])
hold on
plot(mean(acc_imagined_days,1))
plot(median(acc_online_days,1))
plot(median(acc_batch_days,1),'k')

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


%% (MAIN B1) PnP experiment-> how many days required for good bit rates?
% use cumulative increasing number of days to build a MLP
% test the model performance on the PnP data and get bit rates as a
% function of the number of training days

clc;
clear
close all

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data
%session_data = session_data([1:9 11]); % removing bad days
condn_data_total={};
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    %folders_online = logical((strcmp(session_data(i).folder_type,'B')) + (strcmp(session_data(i).folder_type,'O')));
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
    condn_data_imag = load_data_for_MLP(files);


    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    condn_data_online = load_data_for_MLP(files);


    %%%%%%batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    condn_data_batch = load_data_for_MLP(files);

    % append all together
    condn_data_tmp=cell(1,7);
    for k=1:length(condn_data_batch)
        tmp1 = condn_data_imag{k};
        tmp2 = condn_data_online{k};
        tmp3 = condn_data_batch{k};
        tmp = cat(1,tmp1,tmp2,tmp3);
        condn_data_tmp{k}=tmp;
    end
    condn_data_total{i}=condn_data_tmp;
end

condn_data_total_prePnPB1 = condn_data_total;
save condn_data_total_prePnPB1 condn_data_total_prePnPB1 -v7.3


% building a decoder with cumulatively increasing number of days and
% testing performance on the PnP task in terms of overall accuracy and
% bitrates
bit_rates=[];
res_acc=[];
for i=1:length(condn_data_total)
    disp(['Analyzing PnP with Day ' num2str(i) ' of ' num2str(length(condn_data_total))])
    days=[1:i];
    condn_data=cell(1,7);
    for j=1:length(days)
        tmp = condn_data_total{days(j)};
        for k=1:length(tmp)
            tmp0 = condn_data{k};
            tmp1 = tmp{k};
            tmp0 = cat(1,tmp0,tmp1);
            condn_data{k}=tmp0;
        end
    end

    % build a PnP decoder
    net = get_mlp(condn_data,64);

    % test it out on the PnP experimental data
    conf_matrix_overall = get_simulated_acc_PnP(net);

    % store average
    conf_matrix_overall1 = squeeze(mean(conf_matrix_overall,3));
    res_acc(i) = median(diag(conf_matrix_overall1));
end


figure;plot(res_acc)
figure;plot(res_acc,'.','MarkerSize',20)

save simulate_PnP_Exp1_withLesserDays_B1 -v7.3


%% PLOTTING STATS FOR REAL ROBOT R2G PERFORMANCE

clc;clear



%% STATS TO SHOW MAIN EFFECT OF VARIANCE REDUCTIONS

b2_var = [0.01832357, 0.01053406, 0.00996372];
b2_mean = [0.0978772 , 0.16672725, 0.22486098];
b1_var = [38.49924139,  7.21904851,  5.23573381];
b1_mean = [7.25656431, 9.20766884, 9.67933391];

b1_mean_effsize = b1_mean(2:3)/b1_mean(1)*1;
b2_mean_effsize = b2_mean(2:3)/b2_mean(1)*1;
b1_var_effsize = b1_var(1)./b1_var(2:3);
b2_var_effsize = b2_var(1)./b2_var(2:3);


sum(b2_var_effsize + b1_var_effsize)/4
sum(b1_mean_effsize + b2_mean_effsize)/4

%% STATS OF THE REAL ROBOT TASK FIG 8 WITH SPLITTING TASK INTO TWO HALVES (MAIN)

clc;clear
close all

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
%load('wall_1_9.mat')
load('wall_1_9_new')

PnP_days_wall_task = [2,14,19,21,34,35,40,210,0];
dist_total = cell2mat(dist);
min_d = min(dist_total);
max_d = max(dist_total);

% plotting the accuracy of the wall task split by two sections, with linear
% fit
%%%%% plotting first half
acc1=success1_rate;
figure;hold on
x = PnP_days_wall_task(1:9);
t=x;
y = acc1(1:9);
plot(t,y,'ob','MarkerSize',15)
ylim([0 1])
hold on
% logistic fit
[b,p] = logistic_reg(x,y);
xhat=[ones(length(x),1) x'];
yhat = 1./(1 + exp(-(xhat*b)));
plot(t,yhat,'--b')

%%%%% plotting second half 
figure;
hold on
x = PnP_days_wall_task(1:8);t=x;
acc2 = success2_rate;
y = acc2(1:8);
plot(t,y,'ok','MarkerSize',15)
ylim([0 1])
% logistic fit
[b,p] = logistic_reg(x,y);
xhat=[ones(length(x),1) x'];
yhat = 1./(1 + exp(-(xhat*b)));
plot(t,yhat,'--k')
xlim([0 40])
xticks([0:5:40])
yticks([0:.2:1])
set(gcf,'Color','w')

% plotting first and second half
acc1=success1_rate;
figure;hold on
plot(acc1,'ok','MarkerSize',15)
plot(acc1,'k','LineWidth',1)
xticks(1:9)
xticklabels(PnP_days_wall_task)
ylim([0 1])
yticks([0:.2:1])
set(gcf,'Color','w')
xlim([0.5 9])

acc2=success2_rate;
figure;hold on
plot(acc2,'ok','MarkerSize',15)
plot(acc2,'k','LineWidth',1)
xticks(1:9)
xticklabels(PnP_days_wall_task)
ylim([0 1])
yticks([0:.2:1])
set(gcf,'Color','w')
xlim([0.5 9])

%%%% plotting both tasks with full fit
figure;
subplot(2,1,1)
% first part
hold on
y = success1_rate(1:8);
x = PnP_days_wall_task(1:8);
plot(x,y,'ok','MarkerSize',15);
[b,p] = logistic_reg(x,y);
xhat=[ones(length(x),1) x'];
yhat = 1./(1 + exp(-(xhat*b)));
plot(x,yhat,'--k')
xlim([0 220])
xticks([0:10:220])
yticks([0:.2:1])
set(gcf,'Color','w')
ylim([0 1])
xlabel('Days PnP')
ylabel('Success Rate')
legend({'R2G','Logistic Fit'})
set(gca,'FontSize',12)

% second part
subplot(2,1,2);hold on
y = success2_rate(1:8);
x = PnP_days_wall_task(1:8);
plot(x,y,'ob','MarkerSize',15);
[b,p] = logistic_reg(x,y);
xhat=[ones(length(x),1) x'];
yhat = 1./(1 + exp(-(xhat*b)));
plot(x,yhat,'--b')
xlim([0 220])
xticks([0:10:220])
yticks([0:.2:1])
set(gcf,'Color','w')
ylim([0 1])
ylabel('Success Rate')
xlabel('Days PnP')
legend({'Full Task','Logistic Fit'})
set(gca,'FontSize',12)

% get half life
y = success2_rate(1:8);
x = PnP_days_wall_task(1:8);
figure;hold on
plot(x,y,'or','MarkerSize',15);
y=y(2:end-2);x=x(2:end-2);
[b,p] = logistic_reg(x,y);
t=1:220;
x=t;
xhat=[ones(length(x),1) x'];
yhat = 1./(1 + exp(-(xhat*b)));
plot(t,yhat,'--r')
xticks([0:5:50])
xticks([0:40:750])
yticks([0:.2:1])
ylim([0 1])
xlim([0 750])
set(gcf,'Color','w')
set(gca,'FontSize',12)
legend({'Full Task','Logistic Fit (1st month)'})
[aa,bb] = min(abs(yhat - yhat(1)/2));
vline(bb,'g')
ylabel('Success Rate')
xlabel('Days PnP')

t=1:750;
x=t;
xhat=[ones(length(x),1) x'];
yhat = 1./(1 + exp(-(xhat*b)));
plot(t,yhat,'--m')

xlim([0 10])
xticks(1:9)
xticklabels(PnP_days_wall_task)
ylim([0 1.05])
yticks([0:.1:1.01])
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Accuracy')
box off



%% STATS OF THE REAL ROBOT TASKS (MAIN)


clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
%load('wall_1_9.mat')
load('wall_1_9_new')
ttc=ttc2;
success_rate = success2_rate;

PnP_days_wall_task = [2,14,19,21,34,35,40,210,0];

% get the overall distances from target for the wall task
dist_total = cell2mat(dist);
min_d = min(dist_total);
max_d = max(dist_total);

% to scale size based on distance from target, it is (value - min) / (max-min)
% for different dim-> circle, triangle and cross

% batch update time on 20230331
% load the folders and add duration of trials needed for batch update
folders={'104704','105331','110952','111222','111432','111651'};
dayname = '20230331';
total_time=[];
for i=1:length(folders)
    filename = fullfile(root_path,dayname,'RealRobotBatch',folders{i},'BCI_Fixed');
    files=findfiles('',filename)';
    for j=1:length(files)
        load(files{i});
        total_time=[total_time TrialData.Time(end)-TrialData.Time(1)];
    end
end


figure;
%set(gca,'Color',[.5 .5 .5 0.7])
% ylabel is time to target, xlabel is date, size of data-pt. is distance,
% shape of data-pt. is dim type, color is day
hold on
col=turbo(length(ttc));
%col=['r','b','m']; % in case of plotting the task difficulty by color
mm=[];
for i=1:length(ttc)
    tmp=ttc{i};
    dim_tmp = dim{i};
    dist_tmp = dist{i};
    idx=length(tmp);
    idx = i+0.1*randn(idx,1);
    for j=1:length(idx)
        if dim_tmp(j)==1
            shape = '+';
        elseif dim_tmp(j)==2
            shape = '*';
        elseif dim_tmp(j)==3
            shape = 'd';
        end
        msize = 10 + 10*((dist_tmp(j) - min_d)/(max_d - min_d));
        mm=[mm msize];
        plot(idx(j),tmp(j),'Marker',shape,'MarkerSize',msize,'color',col(i,:),...
            'LineWidth',2)
    end
end
xlim([0 10])
xticks(1:9)
xticklabels(PnP_days_wall_task)
ylim([0 150])
yticks([0:20:160])
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Time to target')

% plotting the task difficulty by color, and scale the size
figure;
hold on
col=turbo(length(ttc));
col={'r','b','k'}; % in case of plotting the task difficulty by color
mm=[];
time2target=[];
days_list=[];
median_t2t=[];
for i=1:length(ttc)
    tmp=ttc{i};
    median_t2t(i) = median(tmp);
    time2target=[time2target tmp];
    days_list=[days_list PnP_days_wall_task(i)*ones(1,length(tmp))];
    dim_tmp = dim{i};
    dist_tmp = dist{i};
    idx=length(tmp);
    idx = i+0.1*randn(idx,1);
    for j=1:length(idx)
        msize = 5 + 5*((dist_tmp(j) - min_d)/(max_d - min_d));
        mm=[mm msize];
        plot(idx(j),tmp(j),'Marker','o',...%'MarkerFaceColor', col{dim_tmp(j)},...
            'MarkerSize',msize,'color',col{dim_tmp(j)},...
            'LineWidth',2)
    end
end
xlim([0 10])
xticks(1:9)
xticklabels(PnP_days_wall_task)
ylim([0 150])
yticks([0:20:160])
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Time to target')

% plot the boxplot of time to target
figure;
boxplot(cell2mat(ttc))
ylim([0 150])
yticks([0:20:160])
set(gcf,'Color','w')
xlim([0.85 1.15])
set(gcf,'Color','w')
set(gca,'FontSize',12)
ylabel('Time to target')
box off


% plot the accuracy
acc=success_rate;
figure;hold on
plot(acc(1:9),'ok','MarkerSize',15)
plot(acc,'k','LineWidth',1)
xlim([0 10])
xticks(1:9)
xticklabels(PnP_days_wall_task)
ylim([0 1.05])
yticks([0:.1:1.01])
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Accuracy')
box off

% linear regression
X=[ones(8,1) (1:8)'];
[B,BINT,R,RINT,STATS] = regress(acc(1:8)',X);

% exponential fit with tau parameter on time to target
y=time2target(1:end-4);
x=days_list(1:end-4);
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
time_to_50per = -log(2)*tau;
tt=1:1000;
yhat = a*exp(b*tt) ;
plot(tt,yhat,'r','LineWidth',1)
vline(round(time_to_50per))

% exponential fit with tau parameter on accuracy
%PnP_days_wall_task = [2    14    19    21    34    35    40   210     0];
%acc=[1	0.750000000000000	1	0.875000000000000	0.888888888888889	0.666666666666667	0.800000000000000	0.714285714285714	1];
y = acc(1:8);
x = PnP_days_wall_task(1:8);t=x;
figure;plot(t,y,'ok','MarkerSize',20);
ylim([0 1])
hold on
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
tt=1:1000;
yhat = a*exp(b*tt) ;
plot(tt,yhat,'r','LineWidth',1)
vline(round(time_to_50per))

% computing the exponential curve
tau = (-1/B(2));
A = exp(B(1));
y=exp(y);
x=x(:,2);
yhat = A*exp(-x/tau);
figure;hold on
plot(x,y,'ok','MarkerSize',15)
plot(x,yhat,'k','LineWidth',1)
ylim([0 1])

% testing the exponential curve
t=1:1:150;tau=20;const=2;
y = const*(exp(-t/tau)) + rand(size(t))*0.05;
figure;plot(t,y,'.','MarkerSize',10);
hold on
f=fit(t(:),y(:),'exp1',Algorithm='Levenberg-Marquardt');
a=f.a;
b=f.b;
%c=f.c;
%d=f.d;
%yhat = a*exp(b*t) + c*exp(d*t);
yhat = a*exp(b*t);
plot(t,yhat,'k','LineWidth',1)



[A,tau,yhat,tt] = exp_fit(t,y);
figure;plot(t,y,'.','MarkerSize',10);
hold on
plot(tt,yhat,'k','LineWidth',1)


% compare performance in first month to rest of session
t1 = cell2mat(ttc(1:4));
t1b = sort(bootstrp(1000,@median,t1));
[t1b(25) median(t1) t1b(975)]

t1 = cell2mat(ttc(5:8));
t1b = sort(bootstrp(1000,@median,t1));
[t1b(25) median(t1) t1b(975)]

a1= acc(1:4);
median(a1)
a2= acc(5:8);
median(a2)


%%%% STATS for the top down rotate task 
clc;clear;
load('topDown_rotate_v2.mat');
PnP_days_rotate_task = [77,81,102,203];
% plot the success rate
acc=successRate;
figure;hold on
plot(acc,'ok','MarkerSize',15)
plot(acc,'k','LineWidth',1)
xlim([0.5 4.5])
xticks(1:4)
xticklabels(PnP_days_rotate_task)
ylim([0 1.05])
yticks([0:.1:1.01])
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Accuracy')
box off

% plotting the time to target
figure;
hold on
col=turbo(length(ttc));
col={'r','b','k'}; % in case of plotting the task difficulty by color
for i=1:length(ttc)
    tmp=ttc{i};
    idx=length(tmp);
    idx = i+0.1*randn(idx,1);
    for j=1:length(idx)
        msize = 7.5;
        plot(idx(j),tmp(j),'Marker','o',...%'MarkerFaceColor', col{dim_tmp(j)},...
            'MarkerSize',msize,'color','k',...
            'LineWidth',2)
    end
end
xlim([0.5 4.5])
xticks(1:4)
xticklabels(PnP_days_rotate_task)
ylim([0 1.05])
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylim([0 150])
yticks([0:20:160])
set(gcf,'Color','w')
xlabel('Days - PnP')
ylabel('Time to target')


% plot the boxplot of time to target
figure;
boxplot(cell2mat(ttc))
ylim([0 150])
yticks([0:20:160])
set(gcf,'Color','w')
xlim([0.85 1.15])
set(gcf,'Color','w')
set(gca,'FontSize',12)
ylabel('Time to target')
box off


% performace metrics
t1 = cell2mat(ttc);
t1b = sort(bootstrp(1000,@median,t1));
[t1b(25) median(t1) t1b(975)]

%% COACTIVATION / MULTISTATE building a classifier (MAIN -B1)
% test on held out trials
% the features are the output of the last layer of the LSTM. Collate these
% features at each time-step across the trial, and label trial either right
% hand, leg, head or coactivation RH+leg, RH+head. Train a MLP on training
% trials and test on held out trials.

clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'

folders={'20221129','20221206','20221214','20221215','20230111','20230118',...
    '20230120'};

% load the files especially if robot3dArrow. If Imagined and if
% TrialData.Target is between 10 and 13, then store it
filedata=[];
k=1;
for i=1:length(folders)
    disp(i)
    folderpath = fullfile(root_path,folders{i},'Robot3DArrow');
    D = dir(folderpath);
    for j=3:length(D)
        filepath = fullfile(folderpath,D(j).name);
        D1 =dir(filepath);
        datapath = fullfile(filepath,D1(3).name);
        files = findfiles('',datapath)';
        for ii=1:length(files)
            load(files{ii});
            target = TrialData.TargetID;
            if sum(target == [1 2 4])>0
                filedata(k).TargetID = target;
                filedata(k).filename = files{ii};
                filedata(k).filetype = 1;
                k=k+1;
            elseif (target >= 10) && (target <=11)
                filedata(k).TargetID = target;
                filedata(k).filename = files{ii};
                filedata(k).filetype = 0;
                k=k+1;
            end
        end
    end
end

% load the rnn
load net_bilstm_20220824
net_bilstm = net_bilstm_20220824;
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

% get the neural features per trial
for i=1:length(filedata)
    disp(i/length(filedata)*100)
    filepath = filedata(i).filename;
    [lstm_output]...
        = get_lstm_performance_multistate_features(filepath,net_bilstm,Params,lpFilt);
    filedata(i).lstm_output = lstm_output;
end

% now build a classifier to discriminate between the neural features
% a 7X10X10X5 MLP will do the trick
acc_overall=[];
for iter=1:10

    tid = unique([filedata(1:end).TargetID]);
    train_idx=[];
    test_idx=[];
    condn_data={};
    for i=1:length(tid)
        idx = find([filedata(1:end).TargetID]==tid(i));
        train_idx1 = randperm(length(idx),round(0.9*length(idx)));
        I = ones(length(idx),1);
        I(train_idx1)=0;
        test_idx1 =  find(I==1);
        train_idx = [train_idx idx(train_idx1)];
        test_idx = [test_idx idx(test_idx1')];
        % get the training data
        train_idx1 = idx(train_idx1);
        tmp_data=[];
        for j=1:length(train_idx1)
            %a = filedata(train_idx1(j)).TargetID;
            disp(filedata(train_idx1(j)).TargetID)
            a = filedata(train_idx1(j)).lstm_output;
            if size(a,2)>20
                a = a(:,5:15);
            end
            tmp_data = [tmp_data a];
        end
        condn_data{i}=tmp_data';
    end

    % now build a classifier
    A = condn_data{1};
    B = condn_data{2};
    C = condn_data{3};
    D = condn_data{4};
    E = condn_data{5};
    D = D(randperm(size(D,1),round(0.6*size(D,1))),:);
    E = E(randperm(size(E,1),round(0.6*size(E,1))),:);

    clear N
    N = [A' B' C' D' E' ];
    T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
        5*ones(size(E,1),1)];

    T = zeros(size(T1,1),5);
    [aa bb]=find(T1==1);[aa(1) aa(end)]
    T(aa(1):aa(end),1)=1;
    [aa bb]=find(T1==2);[aa(1) aa(end)]
    T(aa(1):aa(end),2)=1;
    [aa bb]=find(T1==3);[aa(1) aa(end)]
    T(aa(1):aa(end),3)=1;
    [aa bb]=find(T1==4);[aa(1) aa(end)]
    T(aa(1):aa(end),4)=1;
    [aa bb]=find(T1==5);[aa(1) aa(end)]
    T(aa(1):aa(end),5)=1;

    % code to train a neural network
    clear net
    net = patternnet([64 64 ]) ;
    net.performParam.regularization=0.2;
    %net.divideParam.trainRatio=0.85;
    %net.divideParam.valRatio=0.15;
    %net.divideParam.testRatio=0;
    net = train(net,N,T');


    % now run it through the test dataset
    acc=zeros(5);
    for i=1:length(test_idx)
        tmp = filedata(test_idx(i)).lstm_output;
        target = filedata(test_idx(i)).TargetID;
        if target==1
            target=1;
        elseif target==2
            target=2;
        elseif target==4
            target=3;
        elseif target==10
            target=4;
        elseif target==11
            target=5;
        end

        out=net(tmp);
        [aa bb]=max(out);
        decode = mode(bb);
        acc(target,decode) = acc(target,decode)+1;
    end

    for i=1:size(acc,1)
        acc(i,:) = acc(i,:)./sum(acc(i,:));
    end

    figure;
    imagesc(acc)
    set(gcf,'Color','w')
    set(gca,'FontSize',14)
    xticks(1:5)
    yticks(1:5)
    xticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. Leg'})
    yticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. leg'})
    colormap bone
    caxis([0 1])
    colorbar
    title([num2str(100*mean(diag(acc))) '% Accuracy'])
    acc_overall(iter,:,:)=acc;

end

acc = squeeze(mean(acc_overall,1));
figure;
imagesc(acc)
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:5)
yticks(1:5)
xticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. Leg'})
yticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. leg'})
colormap bone
caxis([0 1])
colorbar
title([num2str(100*mean(diag(acc))) '% Accuracy'])


%% ERPs imagined vs. Online action 7DoF: Same somatotopy

clc;clear
addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20210616'};
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

%files=files(1:84)

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
    t=(1/fs)*[1:15];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');


    % now stick all the data together
    %trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];

    % correction
    %     if length(state1)<8
    %         data  =[data(:,1) data];
    %     end

    % store the time to target data
    %     time_to_target(2,TrialData.TargetID) = time_to_target(2,TrialData.TargetID)+1;
    %     if trial_dur<=3
    %         time_to_target(1,TrialData.TargetID) = time_to_target(1,TrialData.TargetID)+1;
    %     end

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


%time_to_target(1,:)./time_to_target(2,:)

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
    erps =  squeeze(D1(i,:,:)); % change this to the action to generate ERPs

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
    idx=10:25;
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
sgtitle('Rt Thumb - Online')

%% SIG CH VIA ERP FOR EACH DAY AND OVER ALL MOVEMENTS FOR OL,CL1,CL2


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data

% init variables
D1_imag_days=[];
D2_imag_days=[];
D3_imag_days=[];
D4_imag_days=[];
D5_imag_days=[];
D6_imag_days=[];
D7_imag_days=[];
D1_CL1_days=[];
D2_CL1_days=[];
D3_CL1_days=[];
D4_CL1_days=[];
D5_CL1_days=[];
D6_CL1_days=[];
D7_CL1_days=[];
D1_CL2_days=[];
D2_CL2_days=[];
D3_CL2_days=[];
D4_CL2_days=[];
D5_CL2_days=[];
D6_CL2_days=[];
D7_CL2_days=[];

% loop over days
session_data = session_data([1:9 11]); % removing bad days
for i=1:length(session_data)


    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    %folders_online = logical((strcmp(session_data(i).folder_type,'B')) + (strcmp(session_data(i).folder_type,'O')));

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
    D1_imag_days=cat(3,D1_imag_days,pmask1);
    D2_imag_days=cat(3,D2_imag_days,pmask2);
    D3_imag_days=cat(3,D3_imag_days,pmask3);
    D4_imag_days=cat(3,D4_imag_days,pmask4);
    D5_imag_days=cat(3,D5_imag_days,pmask5);
    D6_imag_days=cat(3,D6_imag_days,pmask6);
    D7_imag_days=cat(3,D7_imag_days,pmask7);


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
    [D1,D2,D3,D4,D5,D6,D7,D8,D9,tim] = load_erp_data_7DoF(files);

    % run the ERPs and get the significant channels
    pmask1 = sig_ch_erps(D1,TrialData,tim);
    pmask2 = sig_ch_erps(D2,TrialData,tim);
    pmask3 = sig_ch_erps(D3,TrialData,tim);
    pmask4 = sig_ch_erps(D4,TrialData,tim);
    pmask5 = sig_ch_erps(D5,TrialData,tim);
    pmask6 = sig_ch_erps(D6,TrialData,tim);
    pmask7 = sig_ch_erps(D7,TrialData,tim);
    D1_CL1_days=cat(3,D1_CL1_days,pmask1);
    D2_CL1_days=cat(3,D2_CL1_days,pmask2);
    D3_CL1_days=cat(3,D3_CL1_days,pmask3);
    D4_CL1_days=cat(3,D4_CL1_days,pmask4);
    D5_CL1_days=cat(3,D5_CL1_days,pmask5);
    D6_CL1_days=cat(3,D6_CL1_days,pmask6);
    D7_CL1_days=cat(3,D7_CL1_days,pmask7);


    %%%%%% CL2 data ERPs
    disp(['Processing Day ' num2str(i) ' CL2 Files '])
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    [D1,D2,D3,D4,D5,D6,D7,D8,D9,tim] = load_erp_data_7DoF(files);

    % run the ERPs and get the significant channels
    pmask1 = sig_ch_erps(D1,TrialData,tim);
    pmask2 = sig_ch_erps(D2,TrialData,tim);
    pmask3 = sig_ch_erps(D3,TrialData,tim);
    pmask4 = sig_ch_erps(D4,TrialData,tim);
    pmask5 = sig_ch_erps(D5,TrialData,tim);
    pmask6 = sig_ch_erps(D6,TrialData,tim);
    pmask7 = sig_ch_erps(D7,TrialData,tim);
    D1_CL2_days=cat(3,D1_CL2_days,pmask1);
    D2_CL2_days=cat(3,D2_CL2_days,pmask2);
    D3_CL2_days=cat(3,D3_CL2_days,pmask3);
    D4_CL2_days=cat(3,D4_CL2_days,pmask4);
    D5_CL2_days=cat(3,D5_CL2_days,pmask5);
    D6_CL2_days=cat(3,D6_CL2_days,pmask6);
    D7_CL2_days=cat(3,D7_CL2_days,pmask7);
end

save sig_ch_ERPs_B1_7DoF_beta -v7.3


% plotting the stats across days
corr_val=[];
mvmt={'Rt Thumb','Lt Leg','Lt Thumb','Head','Tong','Lips','Both middle'};
for i=1:7 % plot the imag maps along with the correlation across days
    varname = genvarname(['D' num2str(i) '_imag_days']);
    a1 = squeeze(sum(eval(varname),3));
    varname = genvarname(['D' num2str(i) '_CL1_days']);
    a2 = squeeze(sum(eval(varname),3));
    varname = genvarname(['D' num2str(i) '_CL2_days']);
    a3 = squeeze(sum(eval(varname),3));
    figure;
    subplot(1,3,1)
    imagesc(a1)
    %colormap bone
    axis off
    box on
    clim([0 10])
    subplot(1,3,2)
    imagesc(a2)
    %colormap bone
    axis off
    box on
    clim([0 10])
    subplot(1,3,3)
    imagesc(a3)
    %colormap bone
    axis off
    box on
    set(gcf,'Color','w')
    clim([0 10])
    sgtitle(mvmt{i})

    % correlation
    %     corr_val(i,:) = [corr(a1(:),a2(:),'Type','Pearson') ...
    %         corr(a1(:),a3(:),'Type','Pearson'),...
    %         corr(a2(:),a3(:),'Type','Pearson')];

    % distance
    D1 = pdist([a1(:)';a2(:)'],'cosine');
    D2 = pdist([a1(:)';a3(:)'],'cosine');
    D3 = pdist([a2(:)';a3(:)'],'cosine');
    corr_val(i,:) = [D1 D2 D3];
end
mean(corr_val(:))


%% plotting robot trajectories for B3

clc;clear
filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20231101\Robot3D';
cd(filepath)


folders={'161817','162148','162555'};
figure;hold on
cmap=turbo(6);
for i=3%length(folders)
    foldername = fullfile(filepath,folders{i},'BCI_Fixed');
    files=findfiles('',foldername)';
    for j=1:length(files)
        load(files{j});
        kin=TrialData.CursorState(1:3,:);
        tid=TrialData.TargetID;
        %if size(kin,2)*(1/5) <15
            plot3(kin(1,:),kin(2,:),kin(3,:),'color',cmap(tid,:),'LineWidth',2)
        %end
    end
end
axis tight
grid on

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

% plotting decoder acc across days
figure;hold on
acc=[];
acch=[];
acc_good_days = acc_days(good_days);
cmap = turbo(length(acc_good_days));
for i=1:length(acc_good_days)
    tmp  = acc_good_days{i};
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


%% chcking porjection and CKA metric

X = randn(4000,759);

w1=randn(759,275);
w2=randn(759,275);

x = X*w1;
y = X*w2;

% cka between x and  y
x=x-mean(x);
y=y-mean(y);
a=norm(x*x','fro');
b=norm(y*y','fro');
c=norm(x*y','fro');
d=c/(a*b)




